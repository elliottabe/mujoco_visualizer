"""Camera state reported to the client, screen-space pan, and camera presets.

``frame_meta.camera`` exists because nothing previously told the browser where the camera was.
``viewer.js`` seeds its drag handler with a literal ``az = 90, el = -20``, so the first drag
sent an absolute position unrelated to what was on screen and the view teleported. A client
that is told the current camera can seed from it instead.
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
    <camera name="cam_top" pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
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


def test_camera_state_reports_every_field_the_client_binds_to(sess):
    state = sess.camera_state()
    assert set(state) == {
        "mode", "free_type", "azimuth", "elevation", "distance",
        "lookat", "trackbody", "fixedcamid", "named", "selected",
    }
    assert isinstance(state["lookat"], list) and len(state["lookat"]) == 3
    for key in ("azimuth", "elevation", "distance"):
        assert isinstance(state[key], float)


def test_camera_state_reflects_a_free_camera_change(sess):
    sess.set_camera(az=42.0, el=-11.0, dist=0.75)
    state = sess.camera_state()
    assert state["azimuth"] == pytest.approx(42.0)
    assert state["elevation"] == pytest.approx(-11.0)
    assert state["distance"] == pytest.approx(0.75)
    # An arriving free-camera parameter IS the request to be on the free camera.
    assert state["mode"] == "free"
    assert state["selected"] is None


def test_selected_is_the_override_not_vis_states_named(sess):
    """The load-bearing distinction. ``Session.camera`` (the named override that actually wins)
    and ``vis_state['camera']['named']`` are different values, which is exactly why both are
    reported. A client showing ``named`` would claim the free camera was a model camera."""
    sess.viz.vis_state["camera"]["named"] = "cam_top"
    assert sess.camera_state()["named"] == "cam_top"
    assert sess.camera_state()["selected"] is None      # nothing selected it yet

    sess.set_camera(named="cam_side")
    assert sess.camera_state()["selected"] == "cam_side"
    assert sess.camera_state()["named"] == "cam_top"    # unchanged; the two really do differ


def test_frame_meta_carries_the_camera_block():
    """Published on the sim thread beside ``locks``, so a request thread never reads live
    Session state."""
    from mujoco_visualizer.serve.loop import SimLoop

    s = Session(model=mujoco.MjModel.from_xml_string(_XML), width=64, height=48)
    loop = SimLoop(s)
    try:
        loop._publish()
        _seq, _jpeg, meta = loop.latest()
        assert "camera" in meta
        assert meta["camera"]["distance"] == pytest.approx(
            s.viz.vis_state["camera"]["distance"]
        )
    finally:
        loop.stop()
        s.close()
