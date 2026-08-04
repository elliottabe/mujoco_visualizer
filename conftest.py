"""Session-wide test setup: headless GL before any test module imports mujoco.

pytest imports every collected test module (and their transitive ``import mujoco``) before
running anything, so setting MUJOCO_GL inside a test module is too late if another module
was collected first. A root conftest is imported before any test module. Mirrors the
fly_neuromech root conftest.py.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
