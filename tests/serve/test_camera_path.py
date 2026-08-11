"""Camera paths: which camera renders, the path spec, and the per-frame camera list.

The precedence rule in this file's first section exists because there are now three ways to
say "render from this camera" -- an injected MjvCamera, a named override, and vis_state's own
free/named camera -- and before this task nothing stated which wins. A fourth writer arriving
without that written down is how a control ends up silently doing nothing.
"""

import mujoco
import numpy as np
import pytest

from mujoco_visualizer.serve.session import Session

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <camera name="cam_side" pos="1 0 0.5" xyaxes="0 -1 0 0 0 1"/>
    <body name="box" pos="0 0 0.6">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sess():
    s = Session(model=mujoco.MjModel.from_xml_string(_XML), width=64, height=48)
    yield s
    s.close()


def _cam(azimuth=11.0):
    c = mujoco.MjvCamera()
    c.azimuth = azimuth
    c.elevation = -22.0
    c.distance = 0.9
    return c


# -- the precedence rule ---------------------------------------------------------------------


def test_active_camera_prefers_an_injected_object_over_a_named_override(sess):
    sess.set_camera(named="cam_side")
    sess.set_camera_object(_cam())
    active = sess.active_camera()
    assert isinstance(active, mujoco.MjvCamera)
    assert active.azimuth == pytest.approx(11.0)


def test_active_camera_falls_back_to_the_named_override(sess):
    sess.set_camera(named="cam_side")
    assert sess.active_camera() == "cam_side"


def test_active_camera_is_none_for_the_plain_free_camera(sess):
    """None means "the renderer reads vis_state" -- get_camera's no-override branch."""
    sess.set_camera(az=10.0)
    assert sess.active_camera() is None


def test_clearing_the_object_restores_the_named_override(sess):
    sess.set_camera(named="cam_side")
    sess.set_camera_object(_cam())
    sess.set_camera_object(None)
    assert sess.active_camera() == "cam_side"


def test_a_named_selection_clears_an_injected_object(sess):
    """Picking a camera by name is an explicit choice and must win over whatever was injected,
    or the dropdown would appear dead exactly as it did before Plan A's fix wave."""
    sess.set_camera_object(_cam())
    sess.set_camera(named="cam_side")
    assert sess.active_camera() == "cam_side"


def test_a_free_camera_parameter_clears_an_injected_object(sess):
    """A drag IS the request to look somewhere else. Spec D8's disarm rests on this."""
    sess.set_camera_object(_cam())
    sess.set_camera(az=42.0)
    assert sess.active_camera() is None


def test_the_camera_property_never_returns_an_object(sess):
    """`camera` is provenance -- it goes into an export sidecar as JSON. An MjvCamera there
    would raise at json.dumps time, after the render had already finished."""
    sess.set_camera_object(_cam())
    assert sess.camera is None or isinstance(sess.camera, str)


def test_render_uses_the_injected_object(sess):
    """The load-bearing one: precedence must reach the renderer, not just the accessor."""
    sess.set_camera(az=180.0, el=-80.0, dist=2.0)
    far = sess.render().copy()
    sess.set_camera_object(_cam(azimuth=11.0))
    near = sess.render()
    assert not np.array_equal(far, near)
