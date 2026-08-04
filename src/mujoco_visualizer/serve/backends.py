"""Physics backends: the seam that lets :class:`~mujoco_visualizer.serve.session.Session` step
CPU MuJoCo today and a device-resident simulator (Warp/MJX) later without ``Session`` or the
render path changing.

A backend owns *stepping and state*; ``Session`` owns *control composition, rendering, and
description*. Keep that line clean: no ``mj_step`` in ``Session``, no control clamping here.

Only :class:`CpuBackend` lives here. A ``WarpBackend`` needs the vendored ``mujoco_warp``
package, which pulls in JAX -- and this package must never import ``jax``, ``mujoco.mjx``,
``mujoco_warp``, or ``fly_neuromechanics`` -- so it lands in a later task, in the parent repo.
"""

from typing import Optional, Protocol, runtime_checkable

import mujoco
import numpy as np


@runtime_checkable
class PhysicsBackend(Protocol):
    """What :class:`Session` needs from any physics engine.

    ``sync_to`` writes qpos (and qvel) into a host-side ``mujoco.MjData`` for rendering. A
    backend that already steps the caller's own ``MjData`` in place (:class:`CpuBackend`)
    implements it as a no-op; a backend that steps device-resident state (a future
    ``WarpBackend``) uses it to copy the current state back to host memory each frame.
    """

    label: str
    warning: Optional[str]

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        ...

    def step(self, n: int) -> None:
        ...

    def sync_to(self, data: mujoco.MjData) -> None:
        ...

    def reset_to_keyframe(self, name: str) -> None:
        ...

    @property
    def time(self) -> float:
        ...

    def close(self) -> None:
        ...


class UnknownKeyframe(KeyError):
    """Raised by :meth:`PhysicsBackend.reset_to_keyframe` when *name* is not in the model."""


class CpuBackend:
    """Reference backend: plain ``mj_step`` over the caller-supplied ``MjModel``/``MjData``.

    Shares ``Session``'s own ``MjData`` rather than owning a private copy, so :meth:`sync_to`
    is a no-op -- there is nothing to copy; the data stepped here IS the data ``Session``
    renders.

    The fly's tendons are driven by linear force generators here, not by the trained
    force-length/force-velocity muscle model -- a real mechanical difference from the trained
    dynamics, not a cosmetic one. :attr:`warning` is therefore a mandatory non-empty string: a
    viewer that quietly shows the wrong mechanics is worse than one that admits it is a
    simplified stand-in.
    """

    label = "cpu"
    warning = (
        "CPU backend drives tendons with linear force generators, not the trained "
        "force-length/force-velocity muscle model -- this is NOT the trained dynamics."
    )

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

    def set_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl

    def step(self, n: int) -> None:
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)

    def sync_to(self, data: mujoco.MjData) -> None:
        """No-op: this backend steps the caller's own ``MjData`` in place, so there is
        nothing to copy -- copying an array onto itself would be dead work, not a safety net.
        """

    def reset_to_keyframe(self, name: str) -> None:
        """Reset to the keyframe named *name*, looked up BY NAME, never by index.

        Composed models can reorder keyframes relative to their source XML -- e.g. attaching
        the fly onto a floor scene inserts an unnamed keyframe at index 0 ahead of
        ``'default_pose'`` -- so an index-based reset silently picks the wrong pose.
        """
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        if key_id < 0:
            raise UnknownKeyframe(
                f"no keyframe named {name!r}; model has {self.model.nkey} keyframe(s)"
            )
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def close(self) -> None:
        """Nothing to release: this backend holds no resources of its own."""
