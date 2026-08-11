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


@pytest.mark.parametrize("n", [1, 2, 7, 1588])
@pytest.mark.parametrize(
    "weights, loop",
    [
        (None, False),        # 3 equal segments
        ([1.0, 2.0, 1.0], False),
        (None, True),         # 4 segments: the wrap-around one is appended
        ([1.0, 2.0, 1.0], True),  # weights AND loop together
    ],
)
def test_camera_list_for_returns_exactly_n_cameras(sess, n, weights, loop):
    """Spec section 11 test 1, at its full width: n in {1, 2, 7, 1588} crossed with weights
    ``None`` / ``[1, 2, 1]`` / ``loop=True``.

    The narrowed version of this test (n in {2, 7, 120, 1588}, weights [1, 2] only) is why a
    real defect shipped: `make_pan_cameras`' old `max(1, round(w / total_w * n))` allocation
    returned MORE than n cameras whenever the non-last segments rounded up to n or beyond
    between them, and every case that exposes it needs either a small n or three-plus segments
    with uneven weights -- exactly what the narrowing removed. n = 1 and n = 2 with [1, 2, 1] or
    loop=True are the cases that failed before `allocate_segment_frames`.

    n = 1 and n = 2 are also the cases with FEWER FRAMES THAN SEGMENTS, where some keyframe
    provably cannot be visited. The contract is still exactly n cameras -- one frame renders one
    camera -- so a zero-length segment is the correct outcome there and is asserted as such in
    `test_allocate_segment_frames_*` below.
    """
    # A 3-weight list means 3 segments, which is 4 presets unlooped but only 3 looped (the
    # wrap-around segment is the third). `weights=None` imposes no such constraint.
    names = ["a", "b", "c"] if (weights is not None and loop) else ["a", "b", "c", "d"]
    for name, az in zip(names, (0.0, 90.0, 180.0, 270.0)):
        _save(sess, name, az=az)
    sess.set_camera_path(names, weights=weights, loop=loop)

    cams = sess.camera_list_for(n)
    assert len(cams) == n, (
        f"camera_list_for({n}) returned {len(cams)} cameras for weights={weights}, loop={loop}; "
        "the preview clamps its index into an over-long list while ExportJob refuses it, so the "
        "user previews a shot they cannot export"
    )
    assert all(isinstance(c, mujoco.MjvCamera) for c in cams)


def test_camera_list_for_refuses_a_list_that_is_not_exactly_n(sess, monkeypatch):
    """The contract's own boundary check. `camera_list_for` promises "exactly n_frames" and
    INHERITS that from `make_pan_cameras`' allocation; when the allocation broke it, the failure
    surfaced three layers away as `ExportJob.__init__` complaining about a "camera sequence"
    length -- internals the user cannot act on, for a path whose preview worked. So the promise
    now fails at the boundary that makes it, naming both numbers.

    Forced by stubbing `make_pan_cameras`, because `allocate_segment_frames` makes the real
    allocation exact: the point is that a FUTURE regression there is caught here rather than in
    an export.
    """
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    monkeypatch.setattr(
        sess.viz, "make_pan_cameras",
        lambda *a, **k: [mujoco.MjvCamera() for _ in range(9)],
    )
    with pytest.raises(ValueError, match="camera_list_for") as exc:
        sess.camera_list_for(20)
    assert "9" in str(exc.value) and "20" in str(exc.value), str(exc.value)


# -- the allocation itself ---------------------------------------------------------------------
#
# `make_pan_cameras`' per-segment split is the thing that makes "exactly n cameras" true, so it
# is tested directly as well as through camera_list_for: the interesting cases (hundreds of
# weight/frame combinations) are cheap here and would each cost a full camera build there.


@pytest.mark.parametrize("n_segs", [1, 2, 3, 4, 5])
def test_allocate_segment_frames_always_sums_to_total_frames(n_segs):
    """The invariant, swept over the whole reachable space named in the review: weights 1-6 over
    2-4 segments with n small is where the old rule broke, so sweep wider than that."""
    import itertools

    from mujoco_visualizer.visualizer import allocate_segment_frames

    for weights in itertools.product(range(1, 7), repeat=n_segs):
        for n in list(range(1, 40)) + [120, 351, 399, 1588]:
            counts = allocate_segment_frames([float(w) for w in weights], n)
            assert sum(counts) == n, (weights, n, counts)
            assert len(counts) == n_segs
            assert all(c >= 0 for c in counts), (weights, n, counts)


@pytest.mark.parametrize("n_segs", [2, 3, 4, 5])
def test_every_segment_gets_a_frame_once_there_are_enough_frames(n_segs):
    """The old `max(1, ...)` floor's INTENT -- every keyframe is visited -- kept wherever it is
    affordable, which is whenever n >= the number of segments. This is what
    `test_each_segments_first_frame_is_its_start_keyframe` depends on, and a pathological weight
    ratio must not break it: [1, 100] over 3 frames is [1, 2], not [0, 3]."""
    import itertools

    from mujoco_visualizer.visualizer import allocate_segment_frames

    for weights in itertools.product((1.0, 3.0, 100.0), repeat=n_segs):
        for n in range(n_segs, n_segs + 20):
            counts = allocate_segment_frames(list(weights), n)
            assert min(counts) >= 1, (weights, n, counts)
    assert allocate_segment_frames([1.0, 100.0], 3) == [1, 2]


def test_a_segment_may_get_no_frames_when_there_are_fewer_frames_than_segments():
    """The deliberate concession, and the reason it is the right one: with n < n_segs some
    keyframes CANNOT be visited (one frame renders one camera), so a zero-length segment is
    honest arithmetic. The alternative the old floor chose was returning more cameras than there
    are frames -- a list nothing can render -- which is strictly worse than skipping a keyframe.
    """
    from mujoco_visualizer.visualizer import allocate_segment_frames

    counts = allocate_segment_frames([1.0, 2.0, 1.0], 1)
    assert sum(counts) == 1
    assert counts.count(0) == 2
    counts = allocate_segment_frames([1.0, 1.0, 1.0, 1.0], 2)
    assert sum(counts) == 2
    assert counts.count(0) == 2


def test_allocate_segment_frames_is_proportional_where_it_can_be_exact():
    """Not merely exact -- exact AND still the proportional split when one exists, which is what
    keeps the path editor's readout meaningful rather than just self-consistent."""
    from mujoco_visualizer.visualizer import allocate_segment_frames

    assert allocate_segment_frames([1.0, 3.0], 100) == [25, 75]
    assert allocate_segment_frames([1.0, 1.0, 1.0, 1.0], 40) == [10, 10, 10, 10]
    # The review's own repro: 5 presets, weights [3, 6, 6, 1], n = 20 (a 1588-frame trim at
    # stride 80). The old rule made this 21 frames' worth of cameras.
    counts = allocate_segment_frames([3.0, 6.0, 6.0, 1.0], 20)
    assert sum(counts) == 20, counts
    assert counts == [4, 8, 7, 1], counts


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
    through 180. This is the single likeliest silent defect in an interpolated pan.

    The property is stated on the RAW, unwrapped value, which climbs 350 -> 370: `_lerp_angle`
    returns ``a + diff * t`` with ``diff = +20`` and never wraps its result. The spec originally
    described this as "stays within [350, 360] union [0, 10]", which is wrong about this
    implementation -- the same correction Figure A's expectation needed -- and the assertion that
    encoded it (``cam.azimuth >= 349.9 or cam.azimuth <= 10.1``) was satisfied by anything at or
    above 349.9, so it would have passed an azimuth that ran away to 700. It did catch a sweep
    through 180, which is the headline defect, but it pinned nothing about the magnitude.

    Pinned here instead: monotonically rising, bounded by the two keyframes' unwrapped values,
    and nowhere near 180.
    """
    _save(sess, "late", az=350.0)
    _save(sess, "early", az=10.0)
    sess.set_camera_path(["late", "early"])
    azimuths = [cam.azimuth for cam in sess.camera_list_for(60)]

    assert azimuths[0] == pytest.approx(350.0)
    # Rising, and confined to the 20-degree arc from 350 to 370 -- crossing the wrap point at
    # 360 rather than jumping discontinuously to stay inside [0, 360).
    assert all(b >= a for a, b in zip(azimuths, azimuths[1:])), "not monotonically rising"
    assert min(azimuths) >= 350.0 and max(azimuths) < 370.0, (min(azimuths), max(azimuths))
    assert max(azimuths) > 360.0, (
        "the trace never crossed 360, so it did not actually exercise the wrap point"
    )
    # The headline defect: the long way round, through 180.
    assert all(not (20.0 < a % 360.0 < 340.0) for a in azimuths), (
        "the pan swept the long way round through 180"
    )


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


def test_a_deleted_preset_is_reported_even_through_a_warm_cache(sess):
    """The bug a review round found: the cache above is keyed on the path spec + n_frames
    only, so a call that warmed it BEFORE the deletion (exactly what a live preview does --
    it calls camera_list_for with the same n every published frame) kept serving the stale
    list afterwards, with no exception at all. test_a_deleted_preset_is_reported_by_name
    above never warms the cache first, so it could not have caught this."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    sess.camera_list_for(40)              # warm the cache
    sess.delete_camera_preset("b")
    with pytest.raises(ValueError, match="b"):
        sess.camera_list_for(40)          # SAME n -- must still detect the deletion


def test_a_resaved_preset_is_reflected_even_through_a_warm_cache(sess):
    """The second mutation path a fingerprint-free cache key cannot see: re-saving a preset a
    path already references (e.g. re-shooting the opening framing) moves it to a new
    position without touching `set_camera_path` or `delete_camera_preset` -- the only two
    places that used to invalidate the cache.

    The re-save below writes `vis_state['camera']` directly and calls `save_camera_preset`
    WITHOUT going through `Session.set_camera` -- unlike `_save`'s usual route -- because
    `set_camera` now disarms an armed path (D8; see the "D8" test section below), and this test
    is deliberately isolating the FINGERPRINT invalidation path specifically. Going through
    `set_camera` here would disarm the path as a side effect and `camera_list_for` would raise
    `ValueError: no camera path is armed` for an unrelated reason, rather than testing what this
    test exists to test. A real re-shoot in the browser would go through a drag first (which,
    correctly, disarms), then Save; this test's narrower job is the cache alone.
    """
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    first = sess.camera_list_for(40)
    assert first[0].azimuth == pytest.approx(0.0)   # frame 0 lands exactly on "a"'s keyframe

    sess.viz.vis_state["camera"]["azimuth"] = 222.0
    sess.save_camera_preset("a")          # move the camera and re-save over "a", path still armed
    assert sess.camera_path is not None   # confirms this route -- unlike _save -- does not disarm
    second = sess.camera_list_for(40)     # SAME n -- must not serve the old shot
    assert second is not first
    assert second[0].azimuth == pytest.approx(222.0)


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


# -- the live preview ------------------------------------------------------------------------
#
# Preview and export must index ONE list from ONE camera_list_for(n) call. n comes from the
# trim and stride: n = floor((out - in) / stride) + 1, the same count launch.py's
# range(lo, hi + 1, stride) produces for an export. The preview then indexes it with
# k = clamp(floor((frame - in) / stride), 0, n - 1), so preview frame f shows the very object
# export frame k will render.


def test_path_frame_index_maths():
    from mujoco_visualizer.serve.loop import path_frame_index

    # 701 frames, stride 2 -> 351 exported frames.
    assert path_frame_index(frame=200, trim_in=200, trim_out=900, stride=2) == 0
    assert path_frame_index(frame=202, trim_in=200, trim_out=900, stride=2) == 1
    assert path_frame_index(frame=900, trim_in=200, trim_out=900, stride=2) == 350
    # Outside the trim, clamped rather than negative or past the end -- a scrub can sit outside
    # the export range, and a negative index would silently wrap onto the last camera.
    assert path_frame_index(frame=100, trim_in=200, trim_out=900, stride=2) == 0
    assert path_frame_index(frame=5000, trim_in=200, trim_out=900, stride=2) == 350


def test_path_frame_count_matches_an_export_range():
    from mujoco_visualizer.serve.loop import path_frame_count

    for lo, hi, stride in [(0, 1587, 1), (200, 900, 2), (0, 100, 10), (7, 7, 1)]:
        assert path_frame_count(lo, hi, stride) == len(range(lo, hi + 1, stride))


def test_camera_state_reports_path_frame(sess):
    assert sess.camera_state()["path_frame"] is None


def test_a_deleted_preset_disarms_through_the_live_loop_despite_a_warm_cache(sess):
    """End-to-end reproduction of the review finding: a path that has already been previewed
    at least once (i.e. the realistic case -- SimLoop._publish calls camera_list_for with the
    SAME n_frames every tick) has a warm cache by the time a referenced preset is deleted, so
    the disarm-on-ValueError branch in _publish must still fire through that warm cache, not
    only on a cold one (delete-before-first-preview).

    Drives real Session + SimLoop._publish directly, the same way
    test_frame_meta_carries_the_camera_block (tests/serve/test_camera.py) does, rather than
    reasoning about it from camera_list_for alone.
    """
    from mujoco_visualizer.serve.loop import SimLoop
    from mujoco_visualizer.serve.replay import ArrayTrajectorySource

    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])

    qpos = np.zeros((1, 5, sess.model.nq))
    loop = SimLoop(sess, source=ArrayTrajectorySource(qpos))
    loop._playing = True
    try:
        loop._publish()  # warms camera_list_for's cache at n = path_frame_count(0, 4, 1) = 5
        assert loop.error is None
        assert sess.camera_path is not None

        sess.delete_camera_preset("b")
        loop._publish()  # same n_frames as the warming call above

        assert loop.error is not None
        assert loop.error == {
            "t": "error", "kind": "command", "msg": loop.error["msg"], "paused": False,
        }
        assert "b" in loop.error["msg"]
        assert sess.camera_path is None       # disarmed
        assert sess._camera_object is None    # no stale injected camera left rendering
        assert loop.playing is True           # non-pausing: playback was not stopped
    finally:
        loop.stop()


# -- D8: a camera command disarms an armed path, and it must SURVIVE a republish -------------
#
# Task 8's acceptance script found this broken against the real fly model: an earlier version
# of Session.set_camera cleared only `_camera_object` for the tick the drag command landed on.
# That looked sufficient (`active_camera()` really did return None right afterwards) but was
# not: `_camera_path` stayed armed, so the very next `SimLoop._publish()` tick re-derived
# `cameras[path_frame_index(...)]` and called `set_camera_object` again, silently overwriting
# the drag one frame later. The fix is `Session._disarm_camera_path`, called from every place a
# drag/named-selection/explicit-disarm can arrive, so there is one disarm, not two out-of-sync
# ones. The tests below assert the ACTUAL RESOLVED CAMERA after a second publish, not just that
# a flag went None -- a flag-only assertion is exactly what the original, insufficient fix would
# still have passed.


def test_a_free_camera_command_disarms_an_armed_path_even_through_a_republish(sess):
    """The regression test for the D8 finding. Arms a path, publishes once (so a path camera
    is genuinely injected and cached), applies a `camera` drag through the same dispatch
    production uses (`SimLoop._apply`), publishes AGAIN, and checks that the second publish did
    not resurrect the path's camera."""
    from mujoco_visualizer.serve.loop import SimLoop
    from mujoco_visualizer.serve.replay import ArrayTrajectorySource

    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])

    qpos = np.zeros((1, 5, sess.model.nq))
    loop = SimLoop(sess, source=ArrayTrajectorySource(qpos))
    loop._playing = True
    try:
        loop._publish()  # warms the path: injects cameras[path_frame] for this tick
        assert sess.camera_path is not None
        injected_before_drag = sess.active_camera()
        assert isinstance(injected_before_drag, mujoco.MjvCamera)

        # The drag, through the real command dispatch (not calling set_camera directly), so
        # this exercises exactly the seam a browser's canvas drag reaches.
        loop._apply({"t": "camera", "az": 271.0, "el": -10.0})
        loop._publish()  # the tick that used to silently re-inject the path's camera

        # -- the flags (necessary, but this alone is what the original insufficient fix passed)
        assert sess.camera_path is None
        assert loop._meta["camera"]["path_frame"] is None

        # -- the rendered camera itself (this is the assertion the flags-only version lacked).
        # active_camera() must fall through to None -- no injected object survives -- and the
        # camera actually resolved for rendering must carry the DRAGGED azimuth, not either
        # preset's (0.0 or 90.0).
        assert sess.active_camera() is None
        resolved = sess.viz.get_camera(override=sess.active_camera())  # what render_with() uses
        assert isinstance(resolved, mujoco.MjvCamera)
        assert resolved.azimuth == pytest.approx(271.0)

        # -- and the actual rendered PIXELS differ from what the still-armed path would have
        # produced, proving this is not merely a metadata field: re-inject the path's own
        # camera (the exact object the bug would have re-derived) and confirm the frame the fix
        # produces is visually different from the frame the bug would have produced.
        dragged_frame = sess.render().copy()
        sess.set_camera_object(injected_before_drag)
        path_frame_render = sess.render().copy()
        assert not np.array_equal(dragged_frame, path_frame_render)
    finally:
        loop.stop()


def test_a_named_selection_disarms_an_armed_path(sess):
    """The other reachable disarm site: picking a saved camera/XML camera by name is just as
    much a request to look elsewhere as a drag is, and D8 covers both."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    assert sess.camera_path is not None

    sess.set_camera(named="cam_side")

    assert sess.camera_path is None
    assert sess.active_camera() == "cam_side"


def test_set_camera_path_empty_list_still_disarms():
    """Pins that routing the explicit disarm through the shared `_disarm_camera_path` helper
    did not lose the original, simpler disarm behaviour."""
    sess = Session(model=mujoco.MjModel.from_xml_string(_XML), width=64, height=48)
    try:
        _save(sess, "a", az=0.0)
        _save(sess, "b", az=90.0)
        sess.set_camera_path(["a", "b"])
        assert sess.camera_path is not None

        sess.set_camera_path([])

        assert sess.camera_path is None
        assert sess._camera_object is None
        assert sess._camera_list_cache is None
    finally:
        sess.close()


# -- D8, third writer: the Camera tab's fields arrive as `render.set`, not as `{t:"camera"}` --
#
# azimuth/elevation/distance/lookat/free_type/trackbody/fixedcamid are GENERATED controls, so a
# browser sends `{t:"render", set:{"camera.azimuth": 271}}` and it lands in `apply_render` --
# which bypassed `set_camera` and therefore every disarm site. Measured before the fix: the path
# stayed armed, `camera_state` echoed 271 back so the field looked accepted, the rendered camera
# stayed on the path at 0.0, and a later drag touching only elevation applied the stashed 271
# retroactively. `active_camera()` states the precedence for readers; `apply_render` now states
# it for writers.


def test_a_render_set_to_a_camera_key_disarms_an_armed_path_and_actually_renders(sess):
    """Both halves matter and the second is the one the review measured: disarming is not enough
    if the typed azimuth still does not reach the renderer."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    cams = sess.camera_list_for(20)
    sess.set_camera_object(cams[0])
    assert sess.camera_path is not None
    armed_frame = sess.render().copy()

    sess.apply_render({"camera.azimuth": 271.0})

    assert sess.camera_path is None, "a camera.* render.set must disarm the path"
    assert sess._camera_object is None
    # The camera actually resolved for rendering carries the typed azimuth -- not either
    # preset's (0.0 or 90.0), and not merely a vis_state value nothing reads.
    resolved = sess.viz.get_camera(override=sess.active_camera())
    assert isinstance(resolved, mujoco.MjvCamera)
    assert resolved.azimuth == pytest.approx(271.0)
    # ...and the pixels move, which is the claim "typing in one moves the view" (spec 12).
    assert not np.array_equal(armed_frame, sess.render()), (
        "the typed azimuth changed no pixels -- a control the user can change with visibly no "
        "effect is the defect this fix exists to remove"
    )


@pytest.mark.parametrize(
    "dotted, value",
    [
        ("camera.azimuth", 271.0),
        ("camera.elevation", -12.0),
        ("camera.distance", 1.25),
        ("camera.lookat.0", 0.05),
        ("camera.free_type", "free"),
        ("camera.mode", "free"),
        ("camera.named", ""),
    ],
)
def test_every_camera_group_key_disarms_not_just_the_positional_ones(sess, dotted, value):
    """Keyed on the dotted key's ROOT, deliberately, rather than on a list of individual keys:
    any write into the `camera` group is the user driving the camera, and a per-key allowlist is
    how the next generated camera field becomes the next silent special case."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    sess.apply_render({dotted: value})
    assert sess.camera_path is None, f"{dotted} left the path armed"


def test_a_render_set_to_a_non_camera_key_leaves_an_armed_path_alone(sess):
    """The other side of the rule: a look change is not a camera move. Without this, arming a
    path and then touching any render setting at all would silently drop the path."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    sess.apply_render({"floor.alpha": 0.5, "shadows": False})
    assert sess.camera_path is not None, (
        "a non-camera render setting disarmed the path; only writes into the camera group are "
        "the user taking the camera"
    )
    assert sess.viz.vis_state["floor"]["alpha"] == 0.5


def test_the_disarm_happens_once_per_batch_not_once_per_key(sess):
    """A drag arrives as a multi-key batch. Disarming per key would drop the path list and its
    memoised camera list several times for one user action; the check runs before the merge."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera_path(["a", "b"])
    calls = []
    original = sess._disarm_camera_path
    sess._disarm_camera_path = lambda: (calls.append(1), original())[1]
    sess.apply_render(
        {"camera.azimuth": 30.0, "camera.elevation": -5.0, "camera.distance": 0.7}
    )
    assert len(calls) == 1, f"disarmed {len(calls)} times for one batch"


# -- frame_meta must report the camera that is actually RENDERING ----------------------------
#
# `SimLoop._publish` overwrites only `path_frame` on the block `camera_state()` returns, and that
# block read `vis_state['camera']` -- so while a path drove the camera the client was told the
# wrong position. Measured mid-scrub on a 20 -> 300 degree path: screen az=-20.0 dist=0.900,
# frame_meta az=20.0 dist=0.400. `rollout.js` seeds its orbit from `serverCamera.azimuth` and
# its wheel from `serverCamera.distance`, so the first drag jumped ~40 degrees and the first
# wheel tick snapped the zoom -- the teleport this branch exists to fix, through another door.


def test_frame_meta_reports_the_injected_path_camera_not_vis_state(sess):
    """Driven through the real `SimLoop._publish`, so this is the published block itself rather
    than `camera_state()` reasoned about in isolation."""
    from mujoco_visualizer.serve.loop import SimLoop
    from mujoco_visualizer.serve.replay import ArrayTrajectorySource

    _save(sess, "start", az=20.0, dist=0.4)
    _save(sess, "end", az=300.0, dist=1.4)
    # Put vis_state back on the FIRST keyframe, so a block that reports vis_state reports a
    # plausible-looking but wrong camera (az 20 / dist 0.4) rather than an obviously stale one.
    sess.set_camera(az=20.0, el=-20.0, dist=0.4)
    sess.set_camera_path(["start", "end"])

    qpos = np.zeros((1, 21, sess.model.nq))
    loop = SimLoop(sess, source=ArrayTrajectorySource(qpos))
    loop._playing = True
    try:
        loop._frame = 10
        loop._published_frame = 10
        loop._publish()

        injected = sess.active_camera()
        assert isinstance(injected, mujoco.MjvCamera), "no path camera was injected"
        block = loop._meta["camera"]
        assert block["path_frame"] == 10

        assert block["azimuth"] == pytest.approx(injected.azimuth % 360.0)
        assert block["elevation"] == pytest.approx(injected.elevation)
        assert block["distance"] == pytest.approx(injected.distance)
        assert block["lookat"] == pytest.approx(list(injected.lookat))

        # ...and specifically NOT vis_state's, which is what shipped.
        vis = sess.viz.vis_state["camera"]
        assert block["distance"] != pytest.approx(vis["distance"]), (
            "the published distance still matches vis_state; the client's wheel would snap the "
            "zoom on the first tick"
        )
        assert block["azimuth"] != pytest.approx(vis["azimuth"]), (
            "the published azimuth still matches vis_state; the client's first drag would jump"
        )
    finally:
        loop.stop()


def test_a_reported_path_azimuth_is_normalised_into_the_control_range(sess):
    """`_lerp_angle` follows the shortest signed arc without wrapping, so a 350 -> 10 segment
    genuinely climbs to 370 and a 10 -> 200 one goes negative. `camera.azimuth`'s generated
    control range is 0-360, so a raw value outside it would be clamped by the field -- the
    reported angle is normalised to the same physical angle inside the range."""
    _save(sess, "late", az=350.0)
    _save(sess, "early", az=10.0)
    sess.set_camera_path(["late", "early"])
    cams = sess.camera_list_for(60)
    raw = [c.azimuth for c in cams]
    assert max(raw) > 360.0, "fixture no longer produces an out-of-range azimuth"
    for cam in cams:
        sess.set_camera_object(cam)
        reported = sess.camera_state()["azimuth"]
        assert 0.0 <= reported < 360.0, reported
        assert reported == pytest.approx(cam.azimuth % 360.0)


def test_camera_state_still_reports_vis_state_with_no_path_armed(sess):
    """The fix must not change the free-camera case, which is every non-path frame."""
    sess.set_camera(az=137.0, el=-11.0, dist=0.63)
    block = sess.camera_state()
    assert block["azimuth"] == pytest.approx(137.0)
    assert block["elevation"] == pytest.approx(-11.0)
    assert block["distance"] == pytest.approx(0.63)


def test_saving_a_preset_snapshots_the_free_camera_not_an_injected_path_camera(sess):
    """A preset stores az/el/dist/lookat NEXT TO free_type/trackbody/fixedcamid, and those three
    only ever come from vis_state -- an injected MjvCamera carries resolved ids, not names. So
    preset-saving deliberately keeps reading the free camera even though `camera_state()` now
    reports the rendering one; mixing the two could mint a preset describing no reachable camera
    (path keyframes tracking a body, vis_state on the free camera)."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    sess.set_camera(az=42.0, el=-33.0, dist=0.77)
    sess.set_camera_path(["a", "b"])
    sess.set_camera_object(sess.camera_list_for(20)[10])

    sess.save_camera_preset("snap")

    saved = sess.viz.vis_state["camera_presets"]["snap"]
    assert saved["azimuth"] == pytest.approx(42.0)
    assert saved["distance"] == pytest.approx(0.77)


# -- weight VALUES, not just their count ------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_or_non_finite_weight_is_refused_naming_its_index(sess, bad):
    """`weights=[0.0]` used to reach `make_pan_cameras` and raise ZeroDivisionError, which is NOT
    a ValueError -- so `SimLoop._publish`'s `except ValueError` missed it and `_publish_guarded`
    paused the loop with a `kind:"render"` error (the one treatment spec section 9 says a camera
    failure must never get), repeating every tick because the path stayed armed. Unreachable from
    the wire, since `parse_command` rejects these, but reachable from any Python caller -- and
    this is the layer that owns path validation."""
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    with pytest.raises(ValueError, match=r"segment_weights\[0\]"):
        sess.set_camera_path(["a", "b"], weights=[bad])
    assert sess.camera_path is None


def test_the_offending_weight_index_is_named_not_just_the_first(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    with pytest.raises(ValueError, match=r"segment_weights\[1\]"):
        sess.set_camera_path(["a", "b", "c"], weights=[1.0, 0.0])


def test_a_valid_weight_list_is_still_accepted(sess):
    _save(sess, "a", az=0.0)
    _save(sess, "b", az=90.0)
    _save(sess, "c", az=180.0)
    sess.set_camera_path(["a", "b", "c"], weights=[0.5, 2.5])
    assert sess.camera_path["weights"] == [0.5, 2.5]


# -- the camera list caches ids resolved against the CURRENT model ---------------------------


def test_swap_model_invalidates_the_camera_list_cache():
    """`_build_pan_camera` BAKES `trackbodyid` by resolving the preset's body NAME against
    whichever model was current when the camera was built, and `camera_list_for`'s cache key
    fingerprints preset content and frame count but nothing about the model. A cached list
    surviving a swap therefore points the camera at whatever body happens to occupy that index
    in the new model. Latent today only because `thorax` is body 1 in both the plain and the
    ghost model -- an accident of those two XMLs -- so the fixture here deliberately gives the
    two models DIFFERENT body orders, which is what makes the assertion meaningful."""
    _TRACKED = (
        '<body name="tracked" pos="0 0 0.3">'
        '<joint name="s2" type="slide" axis="0 0 1"/>'
        '<geom name="g2" type="box" size="0.02 0.02 0.02"/></body>'
    )
    # `tracked` is the LAST body in the primary model and the FIRST in the alt, so a stale
    # cached trackbodyid resolves to the wrong body rather than harmlessly to the same one.
    primary_xml = _XML.replace("</worldbody>", _TRACKED + "</worldbody>")
    alt_xml = _XML.replace(
        '<body name="box" pos="0 0 0.6">', _TRACKED + '<body name="box" pos="0 0 0.6">'
    )
    primary = mujoco.MjModel.from_xml_string(primary_xml)
    alt = mujoco.MjModel.from_xml_string(alt_xml)
    s = Session(model=primary, alt_model=alt, width=64, height=48)
    try:
        primary_id = mujoco.mj_name2id(primary, mujoco.mjtObj.mjOBJ_BODY, "tracked")
        alt_id = mujoco.mj_name2id(alt, mujoco.mjtObj.mjOBJ_BODY, "tracked")
        assert primary_id != alt_id, "fixture no longer reorders the bodies"

        _save(s, "a", az=0.0, free_type="track", trackbody="tracked")
        _save(s, "b", az=90.0, free_type="track", trackbody="tracked")
        s.set_camera_path(["a", "b"])
        assert s.camera_list_for(10)[0].trackbodyid == primary_id
        assert s._camera_list_cache is not None

        s.swap_model("alt")

        assert s._camera_list_cache is None, "the swap left a camera list built for the old model"
        assert s.camera_list_for(10)[0].trackbodyid == alt_id, (
            "the rebuilt camera still tracks the old model's body id"
        )
    finally:
        s.close()
