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


# -- the path spec and its exclusions --------------------------------------------------------
#
# A path is an ordered list of >=2 free/tracking presets sharing ONE tracking configuration.
# Every exclusion below has the same cause: nothing interpolates. An XML camera resolves to a
# NAME, so there are no az/el/dist to blend; a `fixed` preset has the same problem one level
# down (MuJoCo reads fixedcamid and ignores az/el/dist/lookat); and keyframes that disagree on
# the tracking config make _build_pan_camera snap free_type at the segment midpoint while still
# lerping lookat -- a visible discontinuity halfway through, with no error.


def _save(sess, name, **kw):
    sess.set_camera(**{k: v for k, v in kw.items() if k in ("az", "el", "dist", "lookat")})
    cam = sess.viz.vis_state["camera"]
    for key in ("free_type", "trackbody", "fixedcamid"):
        if key in kw:
            cam[key] = kw[key]
    sess.save_camera_preset(name)


def test_arming_a_path_records_the_spec(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"], weights=[1.0], loop=False)
    assert sess.camera_path == {"cameras": ["a", "b"], "weights": [1.0], "loop": False}


def test_an_empty_camera_list_disarms(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    sess.set_camera_path([])
    assert sess.camera_path is None


def test_camera_list_for_returns_exactly_n_cameras(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    sess.set_camera_path(["a", "b", "c"], weights=[1.0, 2.0])
    for n in (2, 7, 120, 1588):
        cams = sess.camera_list_for(n)
        assert len(cams) == n
        assert all(isinstance(c, mujoco.MjvCamera) for c in cams)


def test_each_segments_first_frame_is_its_start_keyframe(sess):
    """make_pan_cameras emits t=0 at each segment's start, so those frames land exactly on the
    keyframe. The path's FINAL keyframe is approached but never reached -- t = fi/n over
    range(n) never evaluates 1 -- which is a real property of the last frame, not a rounding
    artefact, and is asserted separately below."""
    _save(sess, "a", az=10.0)
    _save(sess, "b", az=200.0)
    sess.set_camera_path(["a", "b"])
    cams = sess.camera_list_for(50)
    assert cams[0].azimuth == pytest.approx(10.0)


def test_the_final_keyframe_is_approached_but_not_reached(sess):
    _save(sess, "a", az=10.0)
    _save(sess, "b", az=200.0)
    sess.set_camera_path(["a", "b"])
    cams = sess.camera_list_for(50)
    assert cams[-1].azimuth != pytest.approx(200.0)
    # 10 -> 200 spans 190 degrees, which is > 180, so _lerp_angle's shortest-arc rule takes
    # the OTHER, 170-degree route (through 0, into negative values) rather than the direct
    # one -- the same mechanism test_azimuth_takes_the_short_way_round_zero exercises. The
    # raw field therefore lands near -160, not near +200, even though -160 % 360 == 200 is
    # the same physical angle; compare mod 360 rather than on the raw (unwrapped) value.
    assert abs((cams[-1].azimuth % 360.0) - 200.0) < 1.0


def test_azimuth_takes_the_short_way_round_zero(sess):
    """_lerp_angle uses the shortest signed difference, so 350 -> 10 must pass through 0, not
    through 180. This is the single likeliest silent defect in an interpolated pan."""
    _save(sess, "late", az=350.0)
    _save(sess, "early", az=10.0)
    sess.set_camera_path(["late", "early"])
    for cam in sess.camera_list_for(60):
        assert cam.azimuth >= 349.9 or cam.azimuth <= 10.1


def test_segment_weights_split_the_frames_proportionally(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    sess.set_camera_path(["a", "b", "c"], weights=[1.0, 3.0])
    cams = sess.camera_list_for(100)
    # Segment 2 starts where azimuth passes its own start keyframe (90). With weights 1:3 the
    # first segment gets about a quarter of the frames.
    first_segment = [i for i, c in enumerate(cams) if c.azimuth < 89.9]
    assert 20 <= len(first_segment) <= 30


def test_loop_appends_a_return_to_the_first_camera(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"], loop=True)
    cams = sess.camera_list_for(80)
    assert len(cams) == 80
    # The looped path passes back down through low azimuth on the way home.
    assert cams[-1].azimuth < cams[len(cams) // 2].azimuth


def test_camera_list_for_caches_and_invalidates_on_the_spec(sess):
    """scene_message-adjacent code paths call this per frame; recomputing 1588 cameras each
    time would be real work for a byte-identical answer."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    first = sess.camera_list_for(40)
    assert sess.camera_list_for(40) is first          # same object, cached
    assert sess.camera_list_for(41) is not first      # different n
    sess.set_camera_path(["b", "a"])
    assert sess.camera_list_for(40) is not first      # different spec


def test_camera_list_for_without_a_path_raises(sess):
    with pytest.raises(ValueError, match="no camera path"):
        sess.camera_list_for(10)


def test_a_deleted_preset_is_reported_by_name(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    sess.delete_camera_preset("b")
    with pytest.raises(ValueError, match="b"):
        sess.camera_list_for(10)


def test_fewer_than_two_cameras_is_refused(sess):
    _save(sess, "a", az=0.0)
    with pytest.raises(ValueError, match="at least two"):
        sess.set_camera_path(["a"])


def test_an_unknown_preset_name_is_refused_and_lists_what_exists(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "real", az=90.0)
    with pytest.raises(ValueError, match="real"):
        sess.set_camera_path(["a", "never_saved"])


def test_an_xml_camera_name_is_refused_with_the_reason(sess):
    _save(sess, "a", az=0.0)
    with pytest.raises(ValueError, match="cannot be interpolated"):
        sess.set_camera_path(["a", "cam_side"])


def test_a_fixed_preset_is_refused_naming_it_and_its_type(sess):
    _save(sess, "a", az=0.0, free_type="free")
    _save(sess, "pinned", az=90.0, free_type="fixed")
    with pytest.raises(ValueError, match="pinned"):
        sess.set_camera_path(["a", "pinned"])


def test_mixed_tracking_configurations_are_refused_naming_both_and_the_difference(sess):
    _save(sess, "loose", az=0.0, free_type="free")
    _save(sess, "tracked", az=90.0, free_type="trackcom", trackbody="box")
    with pytest.raises(ValueError) as excinfo:
        sess.set_camera_path(["loose", "tracked"])
    message = str(excinfo.value)
    assert "loose" in message and "tracked" in message and "free_type" in message


def test_track_and_trackcom_are_not_a_mixed_path(sess):
    """_FREE_TYPE_MAP maps both to mjCAMERA_TRACKING with needs_body=True -- they are aliases,
    so a path using both is not mixed and must be accepted."""
    _save(sess, "old_name", az=0.0, free_type="track", trackbody="box")
    _save(sess, "new_name", az=90.0, free_type="trackcom", trackbody="box")
    sess.set_camera_path(["old_name", "new_name"])
    assert len(sess.camera_list_for(20)) == 20


def test_tracking_presets_on_different_bodies_are_refused(sess):
    """Panning between two bodies would jump trackbodyid at the segment midpoint."""
    _save(sess, "on_box", az=0.0, free_type="trackcom", trackbody="box")
    _save(sess, "on_world", az=90.0, free_type="trackcom", trackbody="world")
    with pytest.raises(ValueError, match="trackbody"):
        sess.set_camera_path(["on_box", "on_world"])


@pytest.mark.parametrize("weights", [[1.0], [1.0, 2.0, 3.0]])
def test_a_wrong_weight_count_is_refused_with_both_numbers(sess, weights):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    with pytest.raises(ValueError) as excinfo:
        sess.set_camera_path(["a", "b", "c"], weights=weights)
    message = str(excinfo.value)
    # The name promises BOTH numbers: what was supplied and what this path actually needs.
    assert str(len(weights)) in message
    assert "2" in message


def test_weights_with_loop_needs_len_cameras_weights_not_len_cameras_minus_one(sess):
    """loop=True appends keyframes[0] back on, so a 3-camera looped path has 3 segments (not
    2) and therefore needs 3 weights -- the exact len(cameras) vs len(cameras) - 1 distinction
    the brief singled out."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    sess.set_camera_path(["a", "b", "c"], weights=[1.0, 2.0, 3.0], loop=True)
    assert len(sess.camera_list_for(30)) == 30


def test_weights_with_loop_and_wrong_count_names_three(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    with pytest.raises(ValueError) as excinfo:
        sess.set_camera_path(["a", "b", "c"], weights=[1.0, 2.0], loop=True)
    assert "3" in str(excinfo.value)


def test_camera_path_property_returns_a_copy(sess):
    """A caller mutating what the property handed back must not silently re-point the live
    path without going through set_camera_path's validation."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"], weights=[1.0])
    spec = sess.camera_path
    spec["cameras"].append("mutated")
    spec["weights"].append(99.0)
    spec["loop"] = True
    assert sess.camera_path == {"cameras": ["a", "b"], "weights": [1.0], "loop": False}


def test_the_cache_key_distinguishes_frame_counts_without_a_rearm(sess):
    """Pins the cache KEY, not set_camera_path's unconditional reset.

    The sibling test above (test_camera_list_for_caches_and_invalidates_on_the_spec) cannot
    tell those apart: set_camera_path clears the cache on every call, so it passes even if
    the key dropped the spec entirely. This one never re-arms between the two lookups, so the
    only thing that can produce two independent lists is the key including n_frames.

    NOTE this does not round-trip back to n=30 after computing n=31: the cache is a single
    most-recently-computed (key, list) SLOT (see camera_list_for's `self._camera_list_cache =
    (key, cameras)`, exactly the brief's own code), not a dict keyed by every n ever asked
    for. Computing n=31 evicts the n=30 entry, so a THIRD call with n=30 is a legitimate fresh
    miss -- not a bug the key-degradation regression this test targets would cause, and not
    something this test should assert away. What it does check is the repeat-call case (same
    n twice in a row, still cached) and the differing-n case (fresh object), both without an
    intervening set_camera_path.
    """
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    thirty = sess.camera_list_for(30)
    assert sess.camera_list_for(30) is thirty          # repeat call, same n: still cached
    thirty_one = sess.camera_list_for(31)
    assert thirty_one is not thirty
    assert len(thirty) == 30
    assert len(thirty_one) == 31


def test_the_cache_key_includes_the_recorded_spec_not_just_frame_count(sess):
    """Mutates the armed spec IN PLACE, bypassing set_camera_path entirely, so its
    unconditional cache reset never fires. If the key still degraded to n_frames alone, this
    would return the stale cached list; a fresh one proves the key reads `cameras` (not just
    `n_frames`) at lookup time.

    Reaches into the private `_camera_path` attribute rather than through the public API:
    there is no public way to mutate an already-armed path in place -- `camera_path` hands
    back a copy for exactly that reason -- so pinning this needs the private seam. Judged
    worth the fragility here since it isolates the KEY's own behaviour from set_camera_path's
    reset, which the sibling tests above cannot do; consistent with `_save`'s own direct
    writes into `vis_state["camera"]`.
    """
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    sess.set_camera_path(["a", "b"])
    first = sess.camera_list_for(30)
    sess._camera_path["cameras"][1] = "c"
    second = sess.camera_list_for(30)
    assert second is not first


# -- the wire message ------------------------------------------------------------------------


def test_protocol_accepts_a_path():
    from mujoco_visualizer.serve.protocol import parse_command

    assert parse_command(
        {"t": "camera_path", "cameras": ["a", "b"], "weights": [1.0, 2.0], "loop": True}
    ) == {"t": "camera_path", "cameras": ["a", "b"], "weights": [1.0, 2.0], "loop": True}


def test_protocol_defaults_weights_and_loop():
    from mujoco_visualizer.serve.protocol import parse_command

    assert parse_command({"t": "camera_path", "cameras": ["a", "b"]}) == {
        "t": "camera_path", "cameras": ["a", "b"], "weights": None, "loop": False,
    }


def test_protocol_accepts_an_empty_list_as_disarm():
    from mujoco_visualizer.serve.protocol import parse_command

    assert parse_command({"t": "camera_path", "cameras": []})["cameras"] == []


@pytest.mark.parametrize(
    "cmd",
    [
        {"t": "camera_path"},                                        # no cameras
        {"t": "camera_path", "cameras": "a,b"},                      # not a list
        {"t": "camera_path", "cameras": ["a", 7]},                   # non-string name
        {"t": "camera_path", "cameras": ["a"]},                      # one name
        {"t": "camera_path", "cameras": ["a", "b"], "weights": "1"},  # weights not a list
        {"t": "camera_path", "cameras": ["a", "b"], "weights": [0.0]},   # non-positive
        {"t": "camera_path", "cameras": ["a", "b"], "weights": [-1.0]},  # negative
        {"t": "camera_path", "cameras": ["a", "b"], "weights": [True]},  # bool is not a number
        {"t": "camera_path", "cameras": ["a", "b"], "loop": "yes"},   # loop not a bool
    ],
)
def test_protocol_rejects_malformed_paths(cmd):
    from mujoco_visualizer.serve.protocol import CommandError, parse_command

    with pytest.raises(CommandError):
        parse_command(cmd)


def test_camera_path_is_last_wins_but_camera_preset_is_not():
    """A path spec is a whole state, so the newest send is the truth -- unlike camera_preset,
    where coalescing would let a save swallow a delete."""
    from mujoco_visualizer.serve import protocol

    assert "camera_path" in protocol._LAST_WINS
    assert "camera_preset" not in protocol._LAST_WINS
