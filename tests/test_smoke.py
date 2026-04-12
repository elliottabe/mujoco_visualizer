"""Smoke test: build a trivial inline MuJoCo model, render one frame."""
import numpy as np
import mujoco
from mujoco_visualizer import Visualizer

_MODEL_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.7 0.7 0.7 1"/>
    <body name="ball" pos="0 0 0.3">
      <joint name="freejoint" type="free"/>
      <geom name="ball_geom" type="sphere" size="0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_render_inline_model(tmp_path):
    xml_path = tmp_path / "model.xml"
    xml_path.write_text(_MODEL_XML)
    viz = Visualizer(str(xml_path))
    frame = viz.render_frame(viz.model.qpos0, height=120, width=160)
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
    # auto anatomy should expose the ball as a category
    assert "ball" in viz.anatomy.category_names
