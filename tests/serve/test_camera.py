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


# -- screen-space pan ------------------------------------------------------------------------
#
# Resolved server-side: only the server knows the current azimuth/elevation, so a client-side
# implementation would duplicate _cfg_to_mjvcamera's orientation maths and be free to drift
# from it.


def test_camera_basis_is_orthonormal_and_right_is_horizontal():
    from mujoco_visualizer.serve.session import camera_basis

    for az, el in [(0.0, 0.0), (37.0, -22.0), (90.0, 45.0), (270.0, -80.0)]:
        forward, right, up = camera_basis(az, el)
        for vec in (forward, right, up):
            assert np.linalg.norm(vec) == pytest.approx(1.0)
        assert float(np.dot(forward, right)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.dot(forward, up)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.dot(right, up)) == pytest.approx(0.0, abs=1e-9)
        # `right` is the screen-horizontal axis and must stay level with the world, or a pan
        # would slide the view up/down as a side effect of moving sideways.
        assert float(right[2]) == pytest.approx(0.0, abs=1e-9)


def test_pan_translates_lookat_perpendicular_to_the_view_direction(sess):
    """The invariant that makes a pan a pan: the point you are looking AT slides across the
    screen, and the direction you are looking FROM does not change. A displacement with any
    component along `forward` would zoom instead."""
    from mujoco_visualizer.serve.session import camera_basis

    for az, el in [(0.0, -30.0), (63.0, 10.0), (200.0, -70.0)]:
        sess.set_camera(az=az, el=el, dist=0.5, lookat=[0.0, 0.0, 0.0])
        sess.set_camera(pan=[13.0, -7.0])
        moved = np.array(sess.camera_state()["lookat"])
        forward, _right, _up = camera_basis(az, el)
        assert float(np.dot(moved, forward)) == pytest.approx(0.0, abs=1e-9)
        assert np.linalg.norm(moved) > 0.0


def test_pan_at_azimuth_zero_and_ninety_move_different_world_axes(sess):
    """At azimuth 0 the screen-horizontal axis is world +y; at 90 it is world -x. A pure
    horizontal drag must therefore move a different world axis at each, and leave z alone."""
    sess.set_camera(az=0.0, el=0.0, dist=0.5, lookat=[0.0, 0.0, 0.0])
    sess.set_camera(pan=[10.0, 0.0])
    at_zero = np.array(sess.camera_state()["lookat"])
    assert abs(at_zero[1]) > 1e-6
    assert at_zero[0] == pytest.approx(0.0, abs=1e-9)
    assert at_zero[2] == pytest.approx(0.0, abs=1e-9)

    sess.set_camera(az=90.0, el=0.0, dist=0.5, lookat=[0.0, 0.0, 0.0])
    sess.set_camera(pan=[10.0, 0.0])
    at_ninety = np.array(sess.camera_state()["lookat"])
    assert abs(at_ninety[0]) > 1e-6
    assert at_ninety[1] == pytest.approx(0.0, abs=1e-9)
    assert at_ninety[2] == pytest.approx(0.0, abs=1e-9)


def test_pan_scales_with_distance(sess):
    """So a drag moves the same APPARENT amount whether you are zoomed in or out. Without
    this, a pan that feels right at distance 0.3 barely moves at 3.0."""
    sess.set_camera(az=0.0, el=0.0, dist=0.2, lookat=[0.0, 0.0, 0.0])
    sess.set_camera(pan=[10.0, 0.0])
    near = np.linalg.norm(sess.camera_state()["lookat"])

    sess.set_camera(az=0.0, el=0.0, dist=2.0, lookat=[0.0, 0.0, 0.0])
    sess.set_camera(pan=[10.0, 0.0])
    far = np.linalg.norm(sess.camera_state()["lookat"])

    assert far == pytest.approx(near * 10.0, rel=1e-6)


def test_pan_puts_the_camera_on_the_free_camera(sess):
    sess.set_camera(named="cam_side")
    sess.set_camera(pan=[5.0, 5.0])
    assert sess.camera_state()["mode"] == "free"
    assert sess.camera_state()["selected"] is None


def test_protocol_accepts_a_pan_pair():
    from mujoco_visualizer.serve.protocol import parse_command

    cmd = parse_command({"t": "camera", "pan": [12.0, -3.0]})
    assert cmd == {"t": "camera", "pan": [12.0, -3.0]}


@pytest.mark.parametrize("bad", [[1.0], [1.0, 2.0, 3.0], "12,3", [1.0, True], None])
def test_protocol_rejects_a_malformed_pan(bad):
    from mujoco_visualizer.serve.protocol import CommandError, parse_command

    with pytest.raises(CommandError, match="pan"):
        parse_command({"t": "camera", "pan": bad})
