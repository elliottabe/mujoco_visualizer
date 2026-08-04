"""Stateful core of the streaming viewer: one model, one data, one persistent Renderer,
stepped through a swappable :class:`~mujoco_visualizer.serve.backends.PhysicsBackend`.

No threads and no networking live here -- :class:`SimLoop` supplies the thread and
``app.py`` the sockets -- so the whole control-and-stepping surface is testable in pytest
with no server running.

``Session`` does not call ``mj_step`` itself: it owns a *backend* (default
:class:`~mujoco_visualizer.serve.backends.CpuBackend`, sharing this Session's own ``MjData``)
and a :class:`~mujoco_visualizer.Visualizer`, and renders from ``self.data`` after syncing it
from the backend each step. That split keeps control composition/clamping/rendering here and
stepping/state in the backend, so a later device-resident backend (Warp/MJX) drops in without
touching this file.
"""

from typing import Dict, Optional, Sequence

import mujoco
import numpy as np
import simplejpeg

from mujoco_visualizer import Visualizer
from mujoco_visualizer.serve.backends import CpuBackend, PhysicsBackend, UnknownKeyframe
from mujoco_visualizer.serve.controls import actuator_group_map, build_control_tree


class Diverged(RuntimeError):
    """Physics produced a non-finite state. Session state is rolled back to the last good
    step before this is raised, so the caller can still render and offer a reset."""


class Session:
    """Owns the simulation and how it is drawn.

    Control composition has two modes. In ``"absolute"`` the per-actuator values set via
    :meth:`set_ctrl` are written to ``data.ctrl`` directly. In ``"additive"`` they are added
    on top of the attached controller's output -- so sliders act as perturbations, mirroring
    how ``scripts/vnc_explorer/sim.build_stim`` already perturbs a running model. Either way
    the result is clamped to each actuator's ``ctrlrange``, because clamping is what is
    wanted mid-drag; rejecting an out-of-range drag would just make the slider feel broken.
    The clamped result is handed to the backend via :meth:`PhysicsBackend.set_ctrl` --
    ``Session`` never writes ``data.ctrl`` itself, so this composition applies unchanged
    regardless of which backend is attached.
    """

    def __init__(
        self,
        *,
        xml_path: Optional[str] = None,
        model: Optional[mujoco.MjModel] = None,
        anatomy=None,
        settings: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 75,
        backend: Optional[PhysicsBackend] = None,
        **viz_kwargs,
    ):
        self.viz = Visualizer(
            xml_path=xml_path,
            model=model,
            anatomy=anatomy,
            settings_json=settings,
            **viz_kwargs,
        )
        self.model = self.viz.model
        self.data = self.viz.data

        # Default backend shares this Session's own MjData, which is exactly what makes its
        # sync_to a no-op: the data it steps IS the data rendered below.
        self.backend: PhysicsBackend = (
            backend if backend is not None else CpuBackend(self.model, self.data)
        )

        self.width = int(width)
        self.height = int(height)
        self.jpeg_quality = int(jpeg_quality)
        self._renderer = self.viz.make_renderer(height=self.height, width=self.width)

        self._tree = build_control_tree(self.model)
        self._group_of = actuator_group_map(self._tree)
        self._id_of = {
            a["name"]: a["id"] for g in self._tree["groups"] for a in g["actuators"]
        }

        self._offset = np.zeros(self.model.nu, dtype=np.float64)
        self._gain: Dict[str, float] = {}
        self._mode = "absolute"
        self._camera = None

        self._controller = None
        self._controller_out: Optional[np.ndarray] = None

        self._lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._limited = self.model.actuator_ctrllimited.astype(bool)

        self._snapshot()

    # -- controller -----------------------------------------------------------

    def attach_controller(self, controller) -> None:
        """Attach an object with ``rate_hz``, ``step(model, data)`` and ``readout()``.

        Attaching also flips to additive mode: with a controller driving ``ctrl``, absolute
        sliders would fight it on every step.
        """
        self._controller = controller
        self._controller_out = None
        self._mode = "additive"

    @property
    def controller_rate_hz(self) -> Optional[float]:
        return None if self._controller is None else float(self._controller.rate_hz)

    def advance_controller(self) -> None:
        """Call the controller once and cache its output. The loop calls this at
        ``rate_hz``, not once per physics step."""
        if self._controller is None:
            return
        out = self._controller.step(self.model, self.data)
        if out is not None:
            self._controller_out = np.asarray(out, dtype=np.float64)

    def readout(self) -> Dict:
        if self._controller is None:
            return {}
        return dict(self._controller.readout())

    # -- control ---------------------------------------------------------------

    def set_ctrl(self, values: Dict[str, float]) -> None:
        """Merge per-actuator values by name. Raises KeyError on an unknown name."""
        for name, value in values.items():
            if name not in self._id_of:
                raise KeyError(f"unknown actuator {name!r}")
            self._offset[self._id_of[name]] = float(value)

    def set_group_gain(self, group: str, gain: float) -> None:
        if group not in {g["id"] for g in self._tree["groups"]}:
            raise KeyError(f"unknown group {group!r}")
        self._gain[group] = float(gain)

    def set_ctrl_mode(self, mode: str) -> None:
        if mode not in ("absolute", "additive"):
            raise ValueError(f"mode must be 'absolute' or 'additive', got {mode!r}")
        self._mode = mode

    def _compose_ctrl(self) -> None:
        scaled = self._offset.copy()
        if self._gain:
            for i in range(self.model.nu):
                g = self._gain.get(self._group_of.get(i, ""), 1.0)
                if g != 1.0:
                    scaled[i] *= g
        if self._mode == "additive" and self._controller_out is not None:
            raw = self._controller_out + scaled
        else:
            raw = scaled
        np.clip(raw, self._lo, self._hi, out=raw, where=self._limited)
        self.backend.set_ctrl(raw)

    # -- stepping --------------------------------------------------------------

    def _snapshot(self) -> None:
        self._good = (self.data.qpos.copy(), self.data.qvel.copy(), float(self.data.time))

    def _restore(self) -> None:
        qpos, qvel, t = self._good
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = t
        mujoco.mj_forward(self.model, self.data)

    def _is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def step(self, n: int) -> None:
        """Advance physics *n* steps via the backend, then sync its state onto ``self.data``.

        Raises :class:`Diverged`, rolling back to the last good step, if the state is
        non-finite -- checked on ``self.data`` *after* the sync, since that is the state
        actually rendered, which is what catches a backend's own dynamics blowing up: a
        muscle model at dt=1e-4 with a user yanking sliders will blow up, and continuing to
        step NaNs makes the rest of the session useless.

        Also checked once *before* stepping: MuJoCo's own ``mj_step`` silently repairs a
        non-finite ``qvel``/``qpos`` it is handed (warns and resets the bad DOF rather than
        propagating the NaN), so an already-corrupted incoming state would otherwise be
        healed out from under this check instead of being caught.
        """
        self._compose_ctrl()
        if not self._is_finite():
            self._restore()
            raise Diverged(
                "physics diverged (non-finite qpos/qvel); rolled back to the last good step"
            )
        self.backend.step(int(n))
        self.backend.sync_to(self.data)
        if not self._is_finite():
            self._restore()
            raise Diverged(
                "physics diverged (non-finite qpos/qvel); rolled back to the last good step"
            )
        self._snapshot()

    def warnings(self) -> Optional[str]:
        """Comma-joined names of MuJoCo warnings raised so far, or None.

        Surfaced per frame rather than treated as fatal: seeing "contact buffer full" as it
        develops is far more useful than only learning about it after a blow-up.
        """
        names = [
            mujoco.mjtWarning(i).name
            for i in range(mujoco.mjtWarning.mjNWARNING)
            if self.data.warning[i].number > 0
        ]
        return ", ".join(names) if names else None

    def set_qpos(self, qpos: Sequence[float]) -> None:
        """Write state directly, no stepping. Used by replay scrubbing."""
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        self._snapshot()

    def reset(self) -> None:
        """Reset to the fly's rest pose, then sync that state from the backend.

        Looks up the ``'default_pose'`` keyframe BY NAME (never index 0 -- on the composed
        fly+floor model index 0 is an unrelated keyframe) and applies NO standing-height
        correction: ``default_pose`` is the experimentally determined muscle rest point with
        the fly *hanging*, deliberately not a standing pose, and it will not hold against a
        floor at ``ctrl=0``. Falls back to ``mj_resetData`` when the model has no such
        keyframe (e.g. the bare test fixtures here), rather than raising.
        """
        try:
            self.backend.reset_to_keyframe("default_pose")
        except UnknownKeyframe:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        self.backend.sync_to(self.data)
        for i in range(mujoco.mjtWarning.mjNWARNING):
            self.data.warning[i].number = 0
        self._controller_out = None
        self._snapshot()

    # -- rendering -------------------------------------------------------------

    def render(self) -> np.ndarray:
        return self.viz.render_with(self._renderer, camera=self._camera)

    def encode(self, frame: np.ndarray) -> bytes:
        return simplejpeg.encode_jpeg(
            np.ascontiguousarray(frame), quality=self.jpeg_quality, colorspace="RGB"
        )

    def resize(self, width: int, height: int) -> None:
        """Rebuild the Renderer at a new resolution.

        Costs ~400 ms on a mesh-heavy model because every mesh re-uploads, so this is a
        deliberate operation the loop pauses around -- never per frame.
        """
        if (int(width), int(height)) == (self.width, self.height):
            return
        old = self._renderer
        self.width, self.height = int(width), int(height)
        self._renderer = self.viz.make_renderer(height=self.height, width=self.width)
        old.close()

    def set_camera(self, named: Optional[str] = None, **kw) -> None:
        """Point the camera. ``named`` selects a named camera/preset; keyword args
        (az/el/dist/lookat) update the free camera in ``vis_state``."""
        if named is not None:
            self._camera = named
            return
        self._camera = None
        cam = self.viz.vis_state.setdefault("camera", {})
        for key, value in kw.items():
            if value is not None:
                cam[key] = value

    def apply_render(self, settings: Dict) -> None:
        """Merge flat render-setting keys into ``vis_state`` (e.g. ``floor.alpha``)."""
        for dotted, value in settings.items():
            node = self.viz.vis_state
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value

    def load_settings(self, name: str) -> None:
        self.viz.load_settings(name)

    # -- description -----------------------------------------------------------

    def scene_message(self) -> Dict:
        """One-time description the client builds its whole UI from."""
        return {
            "t": "scene",
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "timestep": float(self.model.opt.timestep),
            "controls": self._tree,
            "cameras": self.viz.list_cameras(),
            "presets": self.viz.list_presets(),
            "settings": self.viz.vis_state,
            "has_controller": self._controller is not None,
            "ctrl_mode": self._mode,
            "width": self.width,
            "height": self.height,
            "backend": self.backend.label,
            "backend_warning": self.backend.warning,
        }

    def close(self) -> None:
        """Release the Renderer and backend explicitly. EGL teardown raises from ``__del__``
        if left to the garbage collector, so lifetime is always explicit."""
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
        if getattr(self, "backend", None) is not None:
            self.backend.close()
