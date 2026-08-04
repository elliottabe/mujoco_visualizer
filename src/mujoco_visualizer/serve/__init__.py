"""Browser-streaming server for the visualizer.

MuJoCo picks its GL backend from ``MUJOCO_GL`` at ``import mujoco`` time, so anything that
pulls mujoco in before this runs is stuck on the GLFW backend, which fails headless with
"gladLoadGL error" the moment it renders offscreen. Setting it here, at package import,
is early enough for every module below. ``setdefault`` so an explicit osmesa/glfw wins.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

__all__ = ["build_control_tree", "actuator_group_map"]


def __getattr__(name):
    if name in ("build_control_tree", "actuator_group_map"):
        from mujoco_visualizer.serve import controls
        return getattr(controls, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
