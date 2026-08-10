"""Camera state reported to the client, screen-space pan, and camera presets.

``frame_meta.camera`` exists because nothing previously told the browser where the camera was.
``viewer.js`` seeds its drag handler with a literal ``az = 90, el = -20``, so the first drag
sent an absolute position unrelated to what was on screen and the view teleported. A client
that is told the current camera can seed from it instead.
"""

import json

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


# -- camera presets --------------------------------------------------------------------------
#
# A dedicated message, not dotted `render.set` keys: render.set cannot delete, and
# `camera_presets.<name>.lookat` through Session._descend's setdefault would build
# {"0": x, "1": y, "2": z}, which Visualizer._resolve_preset's [float(v) for v in ...] then
# reads as [0.0, 1.0, 2.0] -- a wrong camera, with no error anywhere.

_PRESET_FIELDS = {
    "azimuth", "elevation", "distance", "lookat", "free_type", "trackbody", "fixedcamid",
}


def test_saving_a_preset_stores_exactly_the_seven_resolvable_fields(sess):
    sess.set_camera(az=12.0, el=-34.0, dist=0.66, lookat=[0.1, 0.2, 0.3])
    sess.save_camera_preset("my_shot")
    stored = sess.viz.vis_state["camera_presets"]["my_shot"]
    assert set(stored) == _PRESET_FIELDS
    assert stored["azimuth"] == pytest.approx(12.0)
    assert stored["lookat"] == pytest.approx([0.1, 0.2, 0.3])
    # `mode` and `named` are about WHICH camera is selected, not what this preset IS.
    assert "mode" not in stored and "named" not in stored


def test_a_saved_preset_is_offered_to_clients_and_resolves_to_a_camera(sess):
    sess.set_camera(az=12.0, el=-34.0, dist=0.66)
    sess.save_camera_preset("my_shot")
    assert "my_shot" in sess.scene_message()["presets"]
    cam = sess.viz.get_camera(override="my_shot")
    assert isinstance(cam, mujoco.MjvCamera)
    assert cam.azimuth == pytest.approx(12.0)


def test_saving_twice_overwrites_rather_than_duplicating(sess):
    sess.set_camera(az=10.0)
    sess.save_camera_preset("my_shot")
    sess.set_camera(az=200.0)
    sess.save_camera_preset("my_shot")
    assert sess.viz.vis_state["camera_presets"]["my_shot"]["azimuth"] == pytest.approx(200.0)
    assert sess.scene_message()["presets"].count("my_shot") == 1


def test_a_preset_is_a_snapshot_not_a_live_view(sess):
    """It must not track later camera movement -- otherwise every preset is the same camera."""
    sess.set_camera(az=10.0, lookat=[0.0, 0.0, 0.0])
    sess.save_camera_preset("frozen")
    sess.set_camera(az=99.0, lookat=[9.0, 9.0, 9.0])
    stored = sess.viz.vis_state["camera_presets"]["frozen"]
    assert stored["azimuth"] == pytest.approx(10.0)
    assert stored["lookat"] == pytest.approx([0.0, 0.0, 0.0])


def test_deleting_a_preset_removes_it(sess):
    sess.save_camera_preset("doomed")
    sess.delete_camera_preset("doomed")
    assert "doomed" not in sess.viz.vis_state["camera_presets"]
    assert "doomed" not in sess.scene_message()["presets"]


def test_deleting_an_unknown_preset_names_what_is_available(sess):
    sess.save_camera_preset("real_one")
    with pytest.raises(ValueError, match="real_one"):
        sess.delete_camera_preset("never_existed")


@pytest.mark.parametrize("bad", ["has space", "../escape", "", "tab\tname"])
def test_an_invalid_preset_name_is_refused(sess, bad):
    with pytest.raises(ValueError):
        sess.save_camera_preset(bad)


def test_presets_ride_the_settings_round_trip(sess, tmp_path):
    """No new persistence code: Visualizer.save_settings already emits camera_presets and
    load_settings merges them, which is why the design adds none.

    Deviation from the brief: it names ``Visualizer.to_settings_dict()`` for building the
    in-memory bundle, but no such method exists on ``Visualizer`` (only ``save_settings``,
    which writes JSON to a path, and ``load_settings``, which accepts a path OR a dict). This
    goes through the real, existing round-trip -- save to a file, read the JSON back -- rather
    than adding a new method to get a dict shortcut, since the file list for this task does not
    include visualizer.py and the whole point of this test is that persistence needs no new
    code anywhere.
    """
    sess.set_camera(az=77.0)
    sess.save_camera_preset("persisted")
    dest = tmp_path / "bundle.json"
    sess.viz.save_settings(str(dest))
    with open(dest) as f:
        bundle = json.load(f)
    assert bundle["camera_presets"]["persisted"]["azimuth"] == pytest.approx(77.0)

    fresh = Session(model=mujoco.MjModel.from_xml_string(_XML), width=64, height=48)
    try:
        fresh.viz.load_settings(bundle)
        assert "persisted" in fresh.viz.vis_state["camera_presets"]
    finally:
        fresh.close()


def test_protocol_accepts_save_and_delete():
    from mujoco_visualizer.serve.protocol import parse_command

    assert parse_command({"t": "camera_preset", "op": "save", "name": "shot_a"}) == {
        "t": "camera_preset", "op": "save", "name": "shot_a",
    }
    assert parse_command({"t": "camera_preset", "op": "delete", "name": "shot_a"}) == {
        "t": "camera_preset", "op": "delete", "name": "shot_a",
    }


@pytest.mark.parametrize(
    "cmd",
    [
        {"t": "camera_preset", "op": "rename", "name": "a"},
        {"t": "camera_preset", "name": "a"},
        {"t": "camera_preset", "op": "save"},
        {"t": "camera_preset", "op": "save", "name": "has space"},
        {"t": "camera_preset", "op": "save", "name": 7},
    ],
)
def test_protocol_rejects_malformed_preset_commands(cmd):
    from mujoco_visualizer.serve.protocol import CommandError, parse_command

    with pytest.raises(CommandError):
        parse_command(cmd)
