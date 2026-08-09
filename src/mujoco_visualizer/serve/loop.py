"""The simulation thread: one thread owning both physics and rendering.

The EGL/GL context is thread-affine, so physics and rendering must share a thread. That is
affordable here -- the fly model steps at 0.65x real-time on one core and rendering is
~8.6 ms of a ~48 ms frame -- and it keeps MjData single-owner: Flask's request threads only
ever touch the command queue and the frame slot.

Frame publishing is latest-only. A client that cannot keep up skips to the newest frame
instead of draining a backlog, so a slow link degrades frame rate rather than falling
progressively further behind while server memory grows.
"""

import math
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

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
        source=None,
        frame_dt: float = 1e-3,
        export_factory=None,
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
        # Scene description published from THIS thread alongside each frame. Flask request
        # threads read it from here (see scene()) instead of calling session.scene_message()
        # themselves, which would touch live Session state -- vis_state, mutated by this
        # thread via apply_render/load_settings/set_camera -- from a request thread.
        # Seeded here so /ws and /api/scene have an answer before the first frame exists;
        # SimLoop is constructed on the simulation thread, so this call is on-thread.
        self._scene: Dict = session.scene_message()

        self._playing = False
        self._pending_steps = 0
        self._stop_event = threading.Event()
        self._error: Optional[Dict] = None

        self._clients = 0
        self._last_client_at = time.monotonic()

        self._rtf = 0.0
        # Physics steps still owed to the current control step. <= 0 means one is due now.
        # Starts at 0 so the first tick advances the controller before its first physics step.
        self._ctrl_countdown = 0.0

        # -- replay state. All frame indices are ORIGINAL rollout frames, never strided
        # ones: if `stride` changed what `in`/`out` meant, the trim handles would move
        # under the user every time they changed the slow-motion factor.
        self._source = source
        self._frame_dt = float(frame_dt)
        self._clip = 0
        self._frame = 0
        self._in = 0
        self._out = (source.clip_length(0) - 1) if source is not None else 0
        self._stride = 1
        self._loop_playback = True
        self._ghost = False
        # The frame whose pose was actually last written by _advance_replay -- what
        # replay_state()/_publish report. NOT the same as self._frame once playback has
        # advanced past it: self._frame means "what the next tick will draw", and reporting
        # that instead (as opposed to what is on screen right now) is exactly the bug this
        # field exists to avoid. Seeded to match self._frame so a read before the first tick
        # (nothing written yet) is still sane.
        self._published_frame = 0
        # Set when a command moved the playhead while paused, so the tick renders the new
        # pose once instead of waiting for Play. Also seeded True here (whenever a source is
        # attached) so the very first tick renders frame 0 immediately, before Play is ever
        # pressed -- deliberate UX, not an incidental side effect of the paused-scrub case.
        self._replay_dirty = source is not None

        # Injected so the loop is testable with no GL: production passes a factory that
        # builds a real ExportJob (see serve/app.py and the fly launcher).
        self._export_factory = export_factory
        self._export_job = None

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

    @property
    def replay_mode(self) -> bool:
        return self._source is not None

    def replay_state(self) -> Dict:
        """Snapshot of the replay playhead. Read from the sim thread and from _publish.

        ``frame`` is ``self._published_frame`` -- the frame whose pose was last actually
        written -- not ``self._frame``, which by the time playback has advanced past it means
        "what the next tick will draw". A UI slider bound to this field must always match
        what is on screen, in both the playing and paused/scrubbed cases.

        ``frame_dt`` is published so a client can convert frames to recorded time (and hence
        report a slow-motion factor) instead of assuming a sample rate. A UI that hardcodes
        one is wrong, silently, for every source not sampled at that rate.
        """
        return {
            "clip": self._clip,
            "frame": self._published_frame,
            "in": self._in,
            "out": self._out,
            "stride": self._stride,
            "playing": self._playing,
            "loop": self._loop_playback,
            "ghost": self._ghost,
            "frame_dt": self._frame_dt,
            "n_clips": 0 if self._source is None else self._source.n_clips,
            "length": 0 if self._source is None else self._source.clip_length(self._clip),
        }

    def submit(self, cmd: Dict) -> None:
        """Queue a validated command. Silently drops past ``max_queue`` -- an unbounded
        queue just converts a client flood into server memory growth."""
        with self._qlock:
            if len(self._queue) < self._max_queue:
                self._queue.append(cmd)

    def scene(self) -> Dict:
        """The most recently published scene message.

        Built on the simulation thread (see :meth:`_publish`) and handed out under the frame
        lock, so a Flask request thread never reads ``Session`` state that this thread is
        concurrently mutating. Each publish stores a fresh, fully-snapshotted dict and nothing
        ever mutates a published one, so the returned object is safe to serialise as-is.

        Keeps answering after :meth:`stop` / ``Session.close()``: the last published
        description is still a truthful description of the model that was being shown.
        """
        with self._frame_lock:
            return self._scene

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
        elif kind == "replay":
            self._apply_replay(cmd)
        elif kind == "export":
            self._start_export(cmd)
        elif kind == "export_cancel":
            if self._export_job is not None:
                self._export_job.cancel()
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
                if self.replay_mode:
                    # Session.reset() snaps qpos to the rest-pose keyframe, but the playhead is
                    # untouched -- so without this the next publish reports the old frame while
                    # the canvas shows the rest pose, and nothing re-renders the replay frame
                    # until some later command happens to arrive. The same "reported frame !=
                    # rendered pose" failure _advance_replay's write-then-publish order exists
                    # to prevent, reached through a different door.
                    #
                    # The playhead is deliberately NOT moved: reset is about simulation state,
                    # not about the cursor, and yanking a user's scrub position back to the
                    # trim-in point is not what "reset the physics" asks for. The visible
                    # effect in replay mode is therefore that the rest pose is replaced by the
                    # current frame again on the next tick.
                    self._replay_dirty = True

    def _physics_steps_per_control_step(self) -> Optional[float]:
        """Physics steps covered by one control step, or None when there is no controller.

        Fractional on purpose: at the defaults this is exactly 10 (dt=1e-4, rate_hz=1000), but
        a controller whose rate is not a divisor of the physics rate must not be silently
        rounded to one -- that would change its effective rate, and ``rate_hz`` is a property
        of the trained policy, not a knob.
        """
        rate_hz = getattr(self._session, "controller_rate_hz", None)
        if not rate_hz:
            return None
        dt = self._session.model.opt.timestep if hasattr(self._session, "model") else 1e-4
        control_periods_per_step = dt * float(rate_hz)
        if control_periods_per_step <= 0.0:
            return None
        return 1.0 / control_periods_per_step

    def _advance_and_step(self, n_steps: int) -> None:
        """Interleave the controller with physics across one tick's *n_steps* steps.

        The controller still runs at its own ``rate_hz``, NOT once per physics step -- that
        property is unchanged. What changed is where its output is consumed. Advancing it 26
        times (260 substeps x 1e-4 s x 1000 Hz) and only then calling ``Session.step(260)``
        threw away 25 of the 26 outputs, because ``Session.step`` composes ``ctrl`` once and
        then runs every physics step with that single value: control effectively updated at
        ~38 Hz instead of the trained 1 kHz. With ``dyntype=MUSCLE`` (``tau_act`` 2 ms),
        holding activation constant for 26 ms is not the trained mechanics -- and
        ``WarpBackend.warning`` is None, i.e. this path advertises itself as the trained
        dynamics. Stepping in control-sized chunks costs essentially nothing: physics is ~96%
        of the tick either way.

        The fractional remainder is CARRIED in ``self._ctrl_countdown`` rather than rounded,
        so no physics step is lost or double-counted and the long-run controller rate is
        exactly ``rate_hz``.

        Divergence: each inner ``Session.step`` performs its own warning-counter delta, so the
        delta is measured PER INNER CHUNK rather than across the whole tick. A divergence
        therefore rolls back to the last good CHUNK (tighter than before) and propagates out of
        this method immediately, so ``run()`` reports it exactly once and the remaining chunks
        of a doomed tick are not ground through.
        """
        per_control_step = self._physics_steps_per_control_step()
        if per_control_step is None:
            self._session.step(n_steps)  # no controller: one call, exactly as before
            return

        remaining = int(n_steps)
        while remaining > 0:
            # A `while`, not an `if`: a controller faster than the physics rate owes more than
            # one advance per physics step.
            while self._ctrl_countdown <= 0.0:
                self._session.advance_controller()
                self._ctrl_countdown += per_control_step
            chunk = min(remaining, max(1, int(math.ceil(self._ctrl_countdown))))
            self._session.step(chunk)
            self._ctrl_countdown -= chunk
            remaining -= chunk

    def _apply_replay(self, cmd: Dict) -> None:
        """Move the playhead / retarget the source. Raises on out-of-range values.

        Raising is what gets this reported as an ``error`` with ``kind='command'`` and
        ``paused=False`` by ``run()``'s per-command handler -- the same treatment every other
        bad command gets, and deliberately not a pause.
        """
        if self._source is None:
            raise ValueError("this session has no trajectory source; replay is unavailable")

        if "clip" in cmd:
            clip = int(cmd["clip"])
            if not 0 <= clip < self._source.n_clips:
                raise IndexError(
                    f"clip {clip} out of range (have {self._source.n_clips} clips)"
                )
            if clip != self._clip:
                self._clip = clip
                # A new clip has its own length, so a trim from the previous one is
                # meaningless -- and a stale `out` past this clip's end would raise on the
                # very next advance.
                self._in = 0
                self._out = self._source.clip_length(clip) - 1
                self._frame = min(self._frame, self._out)

        length = self._source.clip_length(self._clip)

        if "trim" in cmd:
            lo, hi = cmd["trim"]
            # protocol.py already enforces ordering and non-negativity before a command
            # reaches this loop; this is defence-in-depth for callers that construct a
            # replay command by hand (e.g. an export job) rather than through protocol.py.
            if lo > hi:
                raise ValueError(f"trim must be ordered [in, out]; got in={lo} > out={hi}")
            if hi >= length:
                raise IndexError(
                    f"trim out={hi} past the end of clip {self._clip} (length {length})"
                )
            self._in, self._out = lo, hi
            self._frame = min(max(self._frame, lo), hi)

        if "stride" in cmd:
            stride = int(cmd["stride"])
            # Same defence-in-depth rationale as trim above: protocol.py already enforces
            # stride >= 1.
            if stride < 1:
                raise ValueError(f"stride must be >= 1, got {stride}")
            self._stride = stride

        if "loop" in cmd:
            self._loop_playback = bool(cmd["loop"])

        if "frame" in cmd:
            frame = int(cmd["frame"])
            if not 0 <= frame < length:
                raise IndexError(
                    f"frame {frame} out of range for clip {self._clip} (length {length})"
                )
            self._frame = frame

        if "ghost" in cmd and bool(cmd["ghost"]) != self._ghost:
            new_ghost = bool(cmd["ghost"])
            # Swap FIRST, flip flags only after it succeeds. If swap_model raises (e.g. a
            # GL failure mid mesh-upload), nothing below has run: self._ghost and the
            # source's own flag are still exactly what they were, so the session, the
            # source, and this loop's own bookkeeping all still agree -- the failure is
            # reported (by run()'s per-command handler) and playback continues on the
            # unchanged, still-coherent model/source pair, rather than being left with
            # flags flipped and the swap only half-done. ~400-560 ms: every mesh
            # re-uploads. Coalescing means at most one call per tick either way.
            self._session.swap_model("alt" if new_ghost else "primary")
            self._ghost = new_ghost
            # The source and the model must agree on qpos width in every tick from here
            # on: a ghost-off-width source (e.g. 101 DOF) paired with the ghost-on model
            # (e.g. 202 DOF, policy+reference concatenated) is exactly the mismatch that
            # makes Session.set_qpos raise on the very next frame. Duck-typed, like
            # `controller_rate_hz` above: a source with no `ghost` attribute (e.g. the
            # generic ArrayTrajectorySource) is untouched and keeps working.
            if hasattr(self._source, "ghost"):
                self._source.ghost = self._ghost

        if "play" in cmd:
            self._playing = bool(cmd["play"])
            if self._playing:
                self._error = None

        self._replay_dirty = True

    def _start_export(self, cmd: Dict) -> None:
        """Slice the frames on THIS thread, then hand them to a job thread.

        The slice is what keeps the job independent: at most 1588 x 202 float64 = 2.6 MB for a
        full ghost clip (``TrajectorySource.qpos`` returns float64, which is what MjData's qpos
        is), so copying is free, and afterwards the job needs neither the source nor this
        Session. Frame indices are original rollout frames.

        The factory is handed the RESOLVED request, not the client's ``cmd``: ``clip`` is not a
        wire field at all (the clip being replayed is this loop's state, and letting a client
        name a different one for export would be a new way for the file to disagree with the
        pixels), and ``trim``/``stride`` are optional on the wire but default to this loop's
        current values. A factory that has to re-derive them can only guess -- which is how
        every auto-named export ended up called ``clip000_...``, overwriting the last one, with
        ``"clip": null`` in its provenance sidecar.
        """
        if self._export_factory is None:
            raise ValueError("this session cannot export: no export_factory was configured")
        if self._source is None:
            raise ValueError("this session has no trajectory source; nothing to export")
        if self._export_job is not None and self._export_job.is_alive():
            raise ValueError(
                "one export at a time; cancel the running export first "
                f"({self._export_job.progress()['done']}/"
                f"{self._export_job.progress()['total']} frames done)"
            )

        lo, hi = cmd.get("trim", (self._in, self._out))
        lo, hi = int(lo), int(hi)
        stride = int(cmd.get("stride", self._stride))
        length = self._source.clip_length(self._clip)
        # Same defence-in-depth as _apply_replay's own trim/stride guards: protocol.py already
        # enforces both, so these only fire for a caller that builds an export command by hand.
        # Without them a reversed trim exports zero frames (np.stack([]) raising far from the
        # cause) and stride <= 0 surfaces as a bare "range() arg 3 must not be zero".
        if lo > hi:
            raise ValueError(f"export trim must be ordered [in, out]; got in={lo} > out={hi}")
        if stride < 1:
            raise ValueError(f"export stride must be >= 1, got {stride}")
        if hi >= length:
            raise IndexError(
                f"export trim out={hi} past the end of clip {self._clip} (length {length})"
            )
        indices = list(range(lo, hi + 1, stride))
        frames = np.stack([self._source.qpos(self._clip, i) for i in indices])
        self._export_job = self._export_factory(
            frames, dict(cmd, clip=self._clip, trim=[lo, hi], stride=stride)
        )
        self._export_job.start()

    def _next_replay_frame(self, frame: int) -> Tuple[int, bool]:
        """Where one stride sends ``frame``, and whether that ran past ``out`` with
        ``loop=False`` -- the boundary at which advancing must stop.

        Pure arithmetic, no side effects (in particular: does NOT touch ``self._playing``),
        so both playback (:meth:`_advance_replay`) and a step
        (:meth:`_step_replay`) go through this one place and the wrap/stop rule can never
        drift between them.
        """
        nxt = frame + self._stride
        if nxt > self._out:
            if self._loop_playback:
                return self._in, False
            return self._out, True
        return nxt, False

    def _advance_replay(self) -> None:
        """Write the current frame, then move the playhead one stride if playing.

        Write-THEN-advance is correct here: a playing tick's frame has already been
        rendered (or was just scrubbed to), so this writes it and only then moves the
        cursor on for the tick after. ``self._published_frame`` is set to the frame just
        written, BEFORE ``self._frame`` potentially moves on below -- callers that read
        state after this returns (i.e. ``_publish``) must see "what was drawn", not "what
        the cursor now points at for next time". Contrast :meth:`_step_replay`, which needs
        the opposite order for the opposite reason.
        """
        self._session.set_qpos(self._source.qpos(self._clip, self._frame))
        self._published_frame = self._frame
        self._replay_dirty = False
        if not self._playing:
            return
        nxt, stop = self._next_replay_frame(self._frame)
        self._frame = nxt
        if stop:
            self._playing = False

    def _step_replay(self, n: int) -> None:
        """Advance the cursor by ``n`` strides, THEN write and report the result --
        deliberately the opposite order from :meth:`_advance_replay`.

        A step means "show me the next frame": the frame currently on screen has already
        been seen, so writing it again first (this used to reuse _advance_replay's
        write-then-advance order) makes a single step press produce no visible change at
        all -- the cursor moves internally but nothing new is ever rendered until some
        later command happens to trigger another write. Moving first fixes that: exactly
        one write, for the frame the cursor lands on.

        ``n`` strides are folded into ONE cursor move and ONE write/publish (never one
        write per intermediate stride, which the caller -- one publish per tick -- could
        not represent anyway). If a non-looping advance hits ``out`` partway through,
        further strides would just repeat ``out``, so the loop stops early rather than
        spinning through them for nothing.

        Does not touch ``self._playing``: a step is a paused-state operation by convention,
        and it is already ``False`` in the ordinary case, so there is nothing to change.
        Forcing it False here would also incorrectly override a `play` command coalesced
        into the very same command batch.
        """
        frame = self._frame
        for _ in range(max(1, int(n))):
            frame, stop = self._next_replay_frame(frame)
            if stop:
                break
        self._frame = frame
        self._session.set_qpos(self._source.qpos(self._clip, self._frame))
        self._published_frame = self._frame
        self._replay_dirty = False

    def _publish_guarded(self, tick_started: float) -> None:
        """Publish, converting a render-side failure into a paused ``render`` error instead
        of letting it escape the tick. Shared by the physics and replay paths so both get
        the same failure handling. ``tick_started`` is unused by the body; it is kept in the
        signature for symmetry with the caller's own timing use of it.
        """
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

    def _publish(self) -> None:
        frame = self._session.render()
        jpeg = self._session.encode(frame)
        if self.replay_mode:
            replay = self.replay_state()
            sim_time = replay["frame"] * self._frame_dt
        else:
            replay = None
            sim_time = float(self._session.data.time)
        meta = {
            "t": "frame_meta",
            "sim_time": sim_time,
            "rtf": round(self._rtf, 3),
            "w": self._session.width,
            "h": self._session.height,
            "playing": self._playing,
            # Per-frame DELTA, not Session.warnings()'s cumulative totals: those are only
            # cleared by reset(), so a banner fed from them is pinned to a stale string
            # forever after the first warning of a session. new_warnings() is stateful and
            # must be called exactly once per frame -- here.
            "warn": self._session.new_warnings(),
            "readout": self._session.readout(),
        }
        if replay is not None:
            # rtf stays 0 in replay mode: nothing advances data.time, and reporting a
            # real-time factor for a file scrub would be a made-up number.
            meta["replay"] = replay
        if self._export_job is not None:
            meta["export"] = self._export_job.progress()
        # Snapshotted on this thread, next to the frame it describes, for the same reason the
        # frame is: request threads must never reach into live Session state.
        scene = self._session.scene_message()
        with self._frame_lock:
            self._seq += 1
            meta["seq"] = self._seq
            self._jpeg = jpeg
            self._meta = meta
            self._scene = scene
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

                    if self.replay_mode:
                        # Replay never steps physics: state comes from the file. `step`
                        # commands nudge the playhead by n strides instead (advance-then-
                        # write -- see _step_replay's docstring for why that is the
                        # opposite order from continuous playback below).
                        if self._pending_steps > 0:
                            n = self._pending_steps
                            self._pending_steps = 0
                            try:
                                self._step_replay(n)
                            except Exception as exc:
                                # The happy path leaves `playing` untouched (see
                                # _step_replay's docstring), but an actual failure here is
                                # exactly the diverged-state situation every other "paused":
                                # True error in this file stops playback for.
                                self._playing = False
                                self._error = {
                                    "t": "error", "kind": "replay",
                                    "msg": str(exc), "paused": True,
                                }
                        elif self._playing or self._replay_dirty:
                            try:
                                self._advance_replay()
                            except Exception as exc:
                                self._playing = False
                                self._error = {
                                    "t": "error", "kind": "replay",
                                    "msg": str(exc), "paused": True,
                                }
                        self._publish_guarded(tick_started)
                        fps_cap = self._fps_cap if self._fps_cap > 0 else 1.0
                        slack = (1.0 / fps_cap) - (time.monotonic() - tick_started)
                        if slack > 0:
                            self._stop_event.wait(slack)
                        continue

                    n_steps = 0
                    if self._pending_steps > 0:
                        n_steps = self._substeps * self._pending_steps
                        self._pending_steps = 0
                    elif self._playing:
                        n_steps = self._substeps

                    if n_steps:
                        sim_before = float(self._session.data.time)
                        try:
                            self._advance_and_step(n_steps)
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

                    self._publish_guarded(tick_started)

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
