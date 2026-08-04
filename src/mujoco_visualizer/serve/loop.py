"""The simulation thread: one thread owning both physics and rendering.

The EGL/GL context is thread-affine, so physics and rendering must share a thread. That is
affordable here -- the fly model steps at 0.65x real-time on one core and rendering is
~8.6 ms of a ~48 ms frame -- and it keeps MjData single-owner: Flask's request threads only
ever touch the command queue and the frame slot.

Frame publishing is latest-only. A client that cannot keep up skips to the newest frame
instead of draining a backlog, so a slow link degrades frame rate rather than falling
progressively further behind while server memory grows.
"""

import threading
import time
from typing import Dict, List, Optional, Tuple

from mujoco_visualizer.serve.protocol import coalesce
from mujoco_visualizer.serve.session import Diverged


class SimLoop(threading.Thread):
    """Drive a Session at a target frame rate, applying queued commands each tick."""

    def __init__(
        self,
        session,
        *,
        fps_cap: float = 20.0,
        substeps_per_frame: int = 260,
        idle_pause_s: Optional[float] = 60.0,
        max_queue: int = 4096,
    ):
        super().__init__(name="SimLoop", daemon=True)
        self._session = session
        self._fps_cap = float(fps_cap)
        self._substeps = int(substeps_per_frame)
        self._idle_pause_s = idle_pause_s
        self._max_queue = int(max_queue)

        self._queue: List[Dict] = []
        self._qlock = threading.Lock()

        self._frame_lock = threading.Condition()
        self._seq = 0
        self._jpeg: Optional[bytes] = None
        self._meta: Dict = {}

        self._playing = False
        self._pending_steps = 0
        self._stop_event = threading.Event()
        self._error: Optional[Dict] = None

        self._clients = 0
        self._last_client_at = time.monotonic()

        self._rtf = 0.0
        self._controller_debt = 0.0

    # -- public surface --------------------------------------------------------

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def substeps_per_frame(self) -> int:
        return self._substeps

    @property
    def rtf(self) -> float:
        return self._rtf

    @property
    def error(self) -> Optional[Dict]:
        return self._error

    def submit(self, cmd: Dict) -> None:
        """Queue a validated command. Silently drops past ``max_queue`` -- an unbounded
        queue just converts a client flood into server memory growth."""
        with self._qlock:
            if len(self._queue) < self._max_queue:
                self._queue.append(cmd)

    def latest(self) -> Optional[Tuple[int, bytes, Dict]]:
        with self._frame_lock:
            if self._jpeg is None:
                return None
            return self._seq, self._jpeg, dict(self._meta)

    def wait_for_frame(
        self, last_seq: int, timeout: float = 1.0
    ) -> Optional[Tuple[int, bytes, Dict]]:
        """Block until a frame newer than *last_seq* exists, then return the NEWEST one.

        Must also block before the very first frame is ever published: with the
        "never seen a frame" sentinel ``last_seq=-1``, ``self._seq`` starts at 0, so the
        wait condition here has to match the post-wait check exactly (``_jpeg is None or
        _seq <= last_seq``) -- otherwise a caller polling in a loop (the expected usage)
        busy-spins at native call rate until the first frame exists.
        """
        with self._frame_lock:
            if self._jpeg is None or self._seq <= last_seq:
                self._frame_lock.wait(timeout)
            if self._jpeg is None or self._seq <= last_seq:
                return None
            return self._seq, self._jpeg, dict(self._meta)

    def client_joined(self) -> None:
        with self._qlock:
            self._clients += 1
            self._last_client_at = time.monotonic()

    def client_left(self) -> None:
        with self._qlock:
            self._clients = max(0, self._clients - 1)
            self._last_client_at = time.monotonic()

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_lock:
            self._frame_lock.notify_all()

    # -- the tick --------------------------------------------------------------

    def _drain(self) -> List[Dict]:
        with self._qlock:
            cmds, self._queue = self._queue, []
        return coalesce(cmds)

    def _apply(self, cmd: Dict) -> None:
        kind = cmd["t"]
        if kind == "ctrl":
            self._session.set_ctrl(cmd["set"])
        elif kind == "ctrl_group":
            self._session.set_group_gain(cmd["group"], cmd["gain"])
        elif kind == "mode":
            self._session.set_ctrl_mode(cmd["ctrl"])
        elif kind == "speed":
            self._substeps = cmd["substeps_per_frame"]
        elif kind == "camera":
            self._session.set_camera(**{k: v for k, v in cmd.items() if k != "t"})
        elif kind == "render":
            self._session.apply_render(cmd["set"])
        elif kind == "settings":
            self._session.load_settings(cmd["load"])
        elif kind == "stream":
            if "fps" in cmd:
                self._fps_cap = cmd["fps"]
            if "quality" in cmd:
                self._session.jpeg_quality = cmd["quality"]
            width = cmd.get("width", self._session.width)
            height = cmd.get("height", self._session.height)
            if (width, height) != (self._session.width, self._session.height):
                self._session.resize(width, height)
        elif kind == "sim":
            action = cmd["cmd"]
            if action == "play":
                self._playing = True
                self._error = None
            elif action == "pause":
                self._playing = False
            elif action == "step":
                self._pending_steps += int(cmd["n"])
            elif action == "reset":
                self._playing = False
                self._error = None
                self._session.reset()

    def _advance_controller_for(self, n_steps: int) -> None:
        """Call the controller once per ``1/rate_hz`` of simulated time covered by
        *n_steps* physics steps -- not once per step."""
        rate_hz = getattr(self._session, "controller_rate_hz", None)
        if not rate_hz:
            return
        dt = self._session.model.opt.timestep if hasattr(self._session, "model") else 1e-4
        self._controller_debt += n_steps * dt * rate_hz
        while self._controller_debt >= 1.0:
            self._session.advance_controller()
            self._controller_debt -= 1.0

    def _publish(self) -> None:
        frame = self._session.render()
        jpeg = self._session.encode(frame)
        meta = {
            "t": "frame_meta",
            "sim_time": float(self._session.data.time),
            "rtf": round(self._rtf, 3),
            "w": self._session.width,
            "h": self._session.height,
            "playing": self._playing,
            "warn": self._session.warnings(),
            "readout": self._session.readout(),
        }
        with self._frame_lock:
            self._seq += 1
            meta["seq"] = self._seq
            self._jpeg = jpeg
            self._meta = meta
            self._frame_lock.notify_all()

    def _maybe_idle_pause(self) -> None:
        if self._idle_pause_s is None or not self._playing:
            return
        with self._qlock:
            idle = self._clients == 0 and (
                time.monotonic() - self._last_client_at > self._idle_pause_s
            )
        if idle:
            self._playing = False

    def run(self) -> None:
        try:
            while not self._stop_event.is_set():
                tick_started = time.monotonic()
                # Everything inside a tick is wrapped as defense-in-depth: nothing in
                # here -- however unexpected -- may propagate out of run() and kill this
                # thread. The specific error kinds below (command/diverged/controller/
                # render) are the ones we understand and want a precise label+pause
                # policy for; this outer catch is the backstop for anything else (e.g. a
                # bug reachable through _drain()/coalesce(), or a degenerate fps_cap).
                slack = 0.05
                try:
                    # _drain() runs coalesce(), which indexes cmd["t"] on every queued
                    # command -- a malformed dict (no "t") raises KeyError here, before
                    # any individual command is ever applied. That is exactly a
                    # client-input problem, not a physics problem, so it gets the same
                    # "command" kind and no-pause treatment as a bad command caught
                    # below: one malformed message from one client must not halt
                    # playback for every other viewer sharing this loop.
                    try:
                        cmds = self._drain()
                    except Exception as exc:
                        cmds = []
                        self._error = {
                            "t": "error",
                            "kind": "command",
                            "msg": str(exc),
                            "paused": False,
                        }

                    for cmd in cmds:
                        try:
                            self._apply(cmd)
                        except Exception as exc:
                            # A bad command (unknown actuator/group/mode, etc.) is a
                            # client-input problem, not evidence the physics state is
                            # untrustworthy -- so unlike diverged/controller/render this
                            # must NOT pause playback. Pausing here would let a single
                            # malformed or version-skewed message from one client freeze
                            # the shared session for every other viewer: a denial of
                            # service via one bad message.
                            self._error = {
                                "t": "error",
                                "kind": "command",
                                "msg": str(exc),
                                "paused": False,
                            }

                    self._maybe_idle_pause()

                    n_steps = 0
                    if self._pending_steps > 0:
                        n_steps = self._substeps * self._pending_steps
                        self._pending_steps = 0
                    elif self._playing:
                        n_steps = self._substeps

                    if n_steps:
                        sim_before = float(self._session.data.time)
                        try:
                            self._advance_controller_for(n_steps)
                            self._session.step(n_steps)
                        except Diverged as exc:
                            self._playing = False
                            self._error = {
                                "t": "error",
                                "kind": "diverged",
                                "msg": str(exc),
                                "paused": True,
                            }
                        except Exception as exc:
                            self._playing = False
                            self._error = {
                                "t": "error",
                                "kind": "controller",
                                "msg": str(exc),
                                "paused": True,
                            }
                        else:
                            advanced = float(self._session.data.time) - sim_before
                            elapsed = max(time.monotonic() - tick_started, 1e-9)
                            # EMA so the reported factor is readable rather than jittery
                            self._rtf = 0.8 * self._rtf + 0.2 * (advanced / elapsed)

                    try:
                        self._publish()
                    except Exception as exc:
                        self._playing = False
                        self._error = {
                            "t": "error",
                            "kind": "render",
                            "msg": str(exc),
                            "paused": True,
                        }

                    fps_cap = self._fps_cap if self._fps_cap > 0 else 1.0
                    budget = 1.0 / fps_cap
                    slack = budget - (time.monotonic() - tick_started)
                except Exception as exc:
                    self._playing = False
                    self._error = {
                        "t": "error",
                        "kind": "internal",
                        "msg": str(exc),
                        "paused": True,
                    }

                if slack > 0:
                    self._stop_event.wait(slack)
        finally:
            self._session.close()
