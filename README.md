# mujoco_visualizer

Generic, config-driven offscreen visualizer for any MuJoCo model. Wraps
`mujoco.Renderer` with: per-category geom recoloring, lighting / floor /
skybox controls, settings JSON, named camera presets, smooth camera pans,
video / image rendering, and optional ipywidgets / dearpygui interfaces.

This is a model-agnostic refactor of the visualizer originally written for the
Janelia fruitfly neuromechanics model. Anatomy (which bodies belong to which
category, named cameras, joint groups for the Pose tab) is supplied via a YAML
or JSON config file — the core knows nothing about any particular model.

## Install

```bash
pip install -e .[io,jupyter,gui]
```

## Quick start

```python
from mujoco_visualizer import Visualizer, load_config

anatomy = load_config("examples/humanoid.yaml")  # or None for auto-detect
viz = Visualizer("humanoid.xml", anatomy=anatomy)
frame = viz.render_frame(viz.model.qpos0, camera="side")
```

See `examples/` for more.
