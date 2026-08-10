"""``vis_state['tendons']`` drives muscle-tendon activation colouring/thickening in the LIVE
browser path -- the per-frame counterpart to ``Visualizer.render_video_pan``'s offline muscle
visualisation (see ``tests/test_tendon_activation.py`` for the extracted functions themselves).

Unlike every render-only ``vis_state`` group, this one:

1. Is driven by ``Session._vis_ctrl``, a visualisation-only vector -- populated by
   ``Session.set_qpos``'s ``ctrl`` parameter during replay, and poked directly (as a stand-in
   for that same seam) in the tests below. It is deliberately NOT ``data.ctrl``: an earlier
   version of this feature read ``data.ctrl`` directly, back when the replay path also wrote
   the recorded ctrl there before ``mj_forward`` -- task 13c removed that write (it was
   perturbing the constraint solve for no requested benefit) and moved this feature's own
   source along with it, onto the store that write left behind.
2. Must actively RESTORE the model's own tendon_rgba/tendon_width the moment it is disabled,
   every frame it stays disabled, not just leave the last-drawn activation frozen on screen
   (see ``Session._apply_tendon_activation_vis``'s docstring for why "looks plausible" is
   exactly the failure mode here).
3. Caches an actuator->tendon map that MUST be rebuilt on a reference-ghost model swap, the same
   way ``Session._ctrl_map`` already is -- a stale map addresses the wrong tendons (or goes out
   of range entirely) on a model with a different ``ntendon``.
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
    <site name="anchor_a" pos="-0.3 0 0.4" size="0.01"/>
    <site name="anchor_b" pos="0.3 0 0.4" size="0.01"/>
    <site name="anchor_free" pos="0 0.5 0.4" size="0.01"/>
    <body name="box_a" pos="-0.3 0 0.6">
      <joint name="slide_a" type="slide" axis="0 0 1"/>
      <geom name="box_a_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_a" pos="0 0 0" size="0.01"/>
    </body>
    <body name="box_b" pos="0.3 0 0.6">
      <joint name="slide_b" type="slide" axis="0 0 1"/>
      <geom name="box_b_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_b" pos="0 0 0" size="0.01"/>
    </body>
    <body name="box_free" pos="0 0.5 0.6">
      <joint name="slide_free" type="slide" axis="0 0 1"/>
      <geom name="box_free_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_free" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_a" width="0.003" rgba="1 0 0 1">
      <site site="anchor_a"/><site site="tip_a"/>
    </spatial>
    <spatial name="t_b" width="0.00015" rgba="0 1 0 1">
      <site site="anchor_b"/><site site="tip_b"/>
    </spatial>
    <spatial name="t_free" width="0.002" rgba="0 0 1 1">
      <site site="anchor_free"/><site site="tip_free"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="m_a" tendon="t_a" ctrlrange="-1 1"/>
    <motor name="m_b" tendon="t_b" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sess():
    model = mujoco.MjModel.from_xml_string(_XML)
    s = Session(model=model, width=64, height=48)
    yield s
    s.close()


def _tendon_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)


# -- disabled by default, and rendering does not require it to be touched at all --------------


def test_tendons_disabled_by_default_render_does_not_change_tendon_state(sess):
    orig_rgba = sess.model.tendon_rgba.copy()
    orig_width = sess.model.tendon_width.copy()
    sess._vis_ctrl[:] = [0.9, 0.9]
    sess.render()
    assert list(sess.model.tendon_rgba.flatten()) == pytest.approx(list(orig_rgba.flatten()))
    assert list(sess.model.tendon_width) == pytest.approx(list(orig_width))


# -- enabling drives alpha/width from the visualisation-only ctrl store, and two different
#    ctrl vectors give two different results ----------------------------------------------------


def test_enabling_colours_muscle_tendons_from_vis_ctrl(sess):
    t_a = _tendon_id(sess.model, "t_a")
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [1.0, 0.0]
    sess.render()
    assert sess.model.tendon_rgba[t_a, 3] == pytest.approx(1.0)
    assert sess.model.tendon_width[t_a] == pytest.approx(
        sess.viz.vis_state["tendons"]["max_width"]
    )


def test_two_different_ctrl_vectors_produce_two_different_alpha_and_width(sess):
    """The load-bearing 'activation actually varies with ctrl' reversion test for the LIVE
    path -- see the task report for the verbatim before/after."""
    t_a = _tendon_id(sess.model, "t_a")
    sess.viz.vis_state["tendons"]["enabled"] = True

    sess._vis_ctrl[:] = [0.1, 0.0]
    sess.render()
    low_alpha = float(sess.model.tendon_rgba[t_a, 3])
    low_width = float(sess.model.tendon_width[t_a])

    sess._vis_ctrl[:] = [0.9, 0.0]
    sess.render()
    high_alpha = float(sess.model.tendon_rgba[t_a, 3])
    high_width = float(sess.model.tendon_width[t_a])

    assert high_alpha > low_alpha
    assert high_width > low_width


def test_non_muscle_tendon_is_hidden_while_enabled(sess):
    t_free = _tendon_id(sess.model, "t_free")
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [0.5, 0.5]
    sess.render()
    assert sess.model.tendon_rgba[t_free, 3] == pytest.approx(0.0)


# -- ctrl_full_scale: a TUNABLE reference, not a measured max (fix round 1) --------------------
#
# actuator_ctrlrange is a THEORETICAL ceiling -- on the real fly model it is 1.05 across all 258
# tendon-driving actuators, but a trained policy's actual |ctrl| occupies a small fraction of
# that ceiling (measured on rollout clip 65, frames 200-320: p50 0.0246, p90 0.0555, p99 0.1442,
# max 0.6287). Normalising against 1.05 therefore renders essentially every tendon near minimum
# width/alpha almost all the time. The fix is an overridable vis_state['tendons']['ctrl_
# full_scale'] knob (default: this fixture's own ctrlrange-derived 1.0) -- a caller holding the
# real rollout (the launcher) is expected to replace it with a measured |ctrl| percentile.


def test_ctrl_full_scale_defaults_to_the_ctrlrange_derived_value(sess):
    from mujoco_visualizer.visualizer import default_tendon_ctrl_full_scale

    assert sess.viz.vis_state["tendons"]["ctrl_full_scale"] == pytest.approx(
        default_tendon_ctrl_full_scale(sess.model)
    )
    assert sess.viz.vis_state["tendons"]["ctrl_full_scale"] == pytest.approx(1.0)


def test_overriding_ctrl_full_scale_changes_alpha_and_width_for_the_same_ctrl(sess):
    """The load-bearing 'the knob changes the resulting alpha/width' reversion test for fix
    round 1 -- see the task report for the verbatim before/after of ignoring the override."""
    t_a = _tendon_id(sess.model, "t_a")
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [0.5, 0.0]

    sess.viz.vis_state["tendons"]["ctrl_full_scale"] = 1.0
    sess.render()
    alpha_at_full_scale_1 = float(sess.model.tendon_rgba[t_a, 3])
    width_at_full_scale_1 = float(sess.model.tendon_width[t_a])

    sess.viz.vis_state["tendons"]["ctrl_full_scale"] = 0.5
    sess.render()
    alpha_at_full_scale_half = float(sess.model.tendon_rgba[t_a, 3])
    width_at_full_scale_half = float(sess.model.tendon_width[t_a])

    # ctrl=0.5 normalised against full_scale=1.0 gives raw=0.5; against full_scale=0.5 it
    # saturates to raw=1.0 -- strictly brighter/wider with the smaller reference.
    assert alpha_at_full_scale_half > alpha_at_full_scale_1
    assert width_at_full_scale_half > width_at_full_scale_1
    assert alpha_at_full_scale_half == pytest.approx(1.0)


# -- a rejected-width replay ctrl must render as INERT, not the previous frame's activation -----
#
# Mirrors SimLoop._write_replay_qpos's own CtrlWidthMismatch handling end-to-end through a real
# Session: a rejected-width ctrl is retried through Session.set_qpos with an explicit all-zero
# vector, sized to CtrlWidthMismatch.expected_width -- never left as whatever _vis_ctrl held
# from the last GOOD frame. This is the task-13c counterpart of the restore-on-disable test right
# below: "we could not apply this frame's commands" must render as NO commands (tendons at their
# floor alpha/width), not a stale-but-plausible activation frozen on screen.


def test_ctrl_width_mismatch_retry_leaves_tendons_inert_not_frozen(sess):
    from mujoco_visualizer.serve.session import CtrlWidthMismatch

    t_a = _tendon_id(sess.model, "t_a")
    sess.viz.vis_state["tendons"]["enabled"] = True
    target = sess.model.qpos0.copy()

    # A good frame: activation elevated above the floor -- proves the retry below is actually
    # clearing something, not vacuously matching an already-inert tendon.
    sess.set_qpos(target, ctrl=[0.9, 0.0])
    sess.render()
    elevated_alpha = float(sess.model.tendon_rgba[t_a, 3])
    elevated_width = float(sess.model.tendon_width[t_a])
    assert elevated_alpha > sess.viz.vis_state["tendons"]["min_alpha"]

    # A rejected-width frame -- wrong length for this 2-actuator model -- retried with an
    # explicit zero vector, exactly as SimLoop._write_replay_qpos does on CtrlWidthMismatch.
    with pytest.raises(CtrlWidthMismatch) as excinfo:
        sess.set_qpos(target, ctrl=[0.1, 0.2, 0.3])
    sess.set_qpos(target, ctrl=[0.0] * excinfo.value.expected_width)
    sess.render()

    min_alpha = sess.viz.vis_state["tendons"]["min_alpha"]
    min_width = sess.viz.vis_state["tendons"]["min_width"]
    assert sess.model.tendon_rgba[t_a, 3] == pytest.approx(min_alpha)
    assert sess.model.tendon_width[t_a] == pytest.approx(min_width)
    assert sess.model.tendon_rgba[t_a, 3] != pytest.approx(elevated_alpha)
    assert sess.model.tendon_width[t_a] != pytest.approx(elevated_width)


# -- restore-on-disable: the load-bearing test for this feature ---------------------------------


def test_disabling_restores_the_models_own_tendon_state_not_the_last_activation(sess):
    """The load-bearing 'restore-on-disable' reversion test -- see the task report for the
    verbatim before/after result of skipping the restore-on-disable branch."""
    t_a = _tendon_id(sess.model, "t_a")
    t_free = _tendon_id(sess.model, "t_free")
    orig_rgba = sess.model.tendon_rgba.copy()
    orig_width = sess.model.tendon_width.copy()

    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [0.9, 0.9]
    sess.render()
    # Sanity: activation actually moved the model away from its original values, so the
    # restore below is provably doing something, not vacuously matching by never having moved.
    assert sess.model.tendon_rgba[t_a, 3] != pytest.approx(float(orig_rgba[t_a, 3]))
    assert sess.model.tendon_rgba[t_free, 3] == pytest.approx(0.0)
    assert float(orig_rgba[t_free, 3]) != pytest.approx(0.0)  # t_free starts visible in the XML

    sess.viz.vis_state["tendons"]["enabled"] = False
    sess.render()

    assert list(sess.model.tendon_rgba.flatten()) == pytest.approx(list(orig_rgba.flatten()))
    assert list(sess.model.tendon_width) == pytest.approx(list(orig_width))


def test_disabling_keeps_restoring_on_every_subsequent_frame_not_just_the_first(sess):
    orig_rgba = sess.model.tendon_rgba.copy()
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [0.9, 0.9]
    sess.render()
    sess.viz.vis_state["tendons"]["enabled"] = False
    for _ in range(3):
        sess.render()
        assert list(sess.model.tendon_rgba.flatten()) == pytest.approx(
            list(orig_rgba.flatten())
        )


# -- partial dicts must not raise ---------------------------------------------------------------
#
# apply_render/load_settings both merge one key at a time into an already fully-populated
# dict (see Visualizer.__init__), so a wire-level partial dict can never actually leave
# vis_state['tendons'] itself partial. The scenario that WOULD leave it partial -- a settings
# file saved before some field existed, or any caller that assigns vis_state['tendons']
# wholesale instead of merging -- is what test_apply_render_with_a_single_tendons_key_does_not_
# raise's sibling below exercises directly, bypassing the merge machinery entirely.


def test_a_wholesale_partial_tendons_dict_does_not_raise(sess):
    """Directly replaces vis_state['tendons'] with a one-key dict -- as a settings file
    written before this feature's other four fields existed would, once loaded by a future
    version of load_settings, or as any caller bypassing apply_render's per-key merge would --
    to prove the apply path itself tolerates a partial dict, not merely that apply_render's own
    merge happens to never produce one."""
    sess.viz.vis_state["tendons"] = {"enabled": True}
    sess._vis_ctrl[:] = [0.5, 0.5]
    sess.render()  # must not raise despite max_width/min_width/min_alpha/baseline missing


def test_a_wholesale_tendons_dict_with_only_ctrl_full_scale_does_not_raise(sess):
    """The fix-round-1 sibling of the test above, added at the coordinator's request: a dict
    carrying ONLY the new ``ctrl_full_scale`` key (no ``enabled`` at all) must not raise
    either."""
    sess.viz.vis_state["tendons"] = {"ctrl_full_scale": 0.6}
    sess._vis_ctrl[:] = [0.5, 0.5]
    sess.render()  # must not raise despite enabled/max_width/min_width/min_alpha/baseline missing


def test_apply_render_with_a_single_tendons_key_does_not_raise(sess):
    # Captured BEFORE render(): rendering while enabled mutates model.tendon_width for the
    # muscle tendons, so reading the model's OWN widths back afterwards would no longer show
    # this MJCF's original values -- the unmentioned vis_state fields are what is under test.
    expected_max_width = float(sess.model.tendon_width.max())

    sess.apply_render({"tendons.enabled": True})
    sess._vis_ctrl[:] = [0.5, 0.5]
    sess.render()  # must not raise despite max_width/min_width/min_alpha/baseline unmentioned
    assert sess.viz.vis_state["tendons"]["enabled"] is True
    # Unmentioned fields keep their construction-time defaults, not some filled-in placeholder.
    assert sess.viz.vis_state["tendons"]["max_width"] == pytest.approx(expected_max_width)


def test_apply_render_with_only_baseline_leaves_other_tendons_fields_untouched(sess):
    sess.apply_render({"tendons.min_alpha": 0.33})
    sess.apply_render({"tendons.baseline": 0.2})
    assert sess.viz.vis_state["tendons"]["min_alpha"] == pytest.approx(0.33)
    assert sess.viz.vis_state["tendons"]["baseline"] == pytest.approx(0.2)
    assert sess.viz.vis_state["tendons"]["enabled"] is False  # never mentioned, stayed default


def test_apply_render_with_only_ctrl_full_scale_does_not_raise_and_takes_effect(sess):
    t_a = _tendon_id(sess.model, "t_a")
    sess.apply_render({"tendons.ctrl_full_scale": 0.6})
    assert sess.viz.vis_state["tendons"]["ctrl_full_scale"] == pytest.approx(0.6)
    assert sess.viz.vis_state["tendons"]["enabled"] is False  # never mentioned, stayed default

    sess.apply_render({"tendons.enabled": True})
    sess._vis_ctrl[:] = [0.5, 0.0]
    sess.render()  # must not raise despite max_width/min_width/min_alpha/baseline unmentioned
    assert sess.model.tendon_rgba[t_a, 3] == pytest.approx(0.5 / 0.6)  # ctrl / ctrl_full_scale


# -- protocol: 'tendons' must be an accepted render.set root -------------------------------------


def test_render_set_tendons_key_is_accepted_by_parse_command():
    from mujoco_visualizer.serve.protocol import parse_command

    cmd = parse_command({"t": "render", "set": {"tendons.enabled": True}})
    assert cmd == {"t": "render", "set": {"tendons.enabled": True}}


# -- the reference-ghost swap must rebuild the tendon map, never reuse it -----------------------

_ALT_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <site name="anchor_c" pos="0 0 0.4" size="0.01"/>
    <body name="box_c" pos="0 0 0.6">
      <joint name="slide_c" type="slide" axis="0 0 1"/>
      <geom name="box_c_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_c" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_c" width="0.001" rgba="1 1 0 1">
      <site site="anchor_c"/><site site="tip_c"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="m_c" tendon="t_c" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def tendon_swap_session():
    primary = mujoco.MjModel.from_xml_string(_XML)
    alt = mujoco.MjModel.from_xml_string(_ALT_XML)
    s = Session(model=primary, alt_model=alt, width=64, height=48)
    yield s
    s.close()


def test_swap_model_rebuilds_the_tendon_map_for_the_smaller_alt_model(tendon_swap_session):
    """The load-bearing 'swap rebuild' reversion test. ``alt`` has ntendon=1 where ``primary``
    has ntendon=3 -- a map/snapshot left over from primary would either colour the wrong tendon
    on alt or index straight past the end of its (shorter) tendon_rgba/tendon_width arrays. See
    the task report for the verbatim IndexError this guards against."""
    s = tendon_swap_session
    s.viz.vis_state["tendons"]["enabled"] = True
    s._vis_ctrl[:] = [0.5, 0.5]
    s.render()  # primary: exercise the map once before swapping away from it

    s.swap_model("alt")
    assert s.model.ntendon == 1

    t_c = _tendon_id(s.model, "t_c")
    s._vis_ctrl[:] = [1.0]
    s.render()  # must not raise -- and must colour t_c, the ONLY tendon on this model
    assert s.model.tendon_rgba[t_c, 3] == pytest.approx(1.0)
    assert s.model.tendon_width[t_c] == pytest.approx(
        s.viz.vis_state["tendons"]["max_width"]
    )


def test_swap_model_back_and_forth_keeps_the_tendon_map_correct(tendon_swap_session):
    s = tendon_swap_session
    s.viz.vis_state["tendons"]["enabled"] = True

    s.swap_model("alt")
    t_c = _tendon_id(s.model, "t_c")
    s._vis_ctrl[:] = [1.0]
    s.render()
    assert s.model.tendon_rgba[t_c, 3] == pytest.approx(1.0)

    s.swap_model("primary")
    t_a = _tendon_id(s.model, "t_a")
    s._vis_ctrl[:] = [1.0, 0.0]
    s.render()
    assert s.model.ntendon == 3
    assert s.model.tendon_rgba[t_a, 3] == pytest.approx(1.0)


def test_swap_model_rebuilds_the_restore_snapshot_from_the_new_models_own_values(
    tendon_swap_session,
):
    """Disabling after a swap must restore to the model NOW ACTIVE's own tendon_rgba/width, not
    to a snapshot taken of the model that was active before the swap (wrong shape entirely once
    ntendon differs, and wrong VALUES even when shapes happen to coincide)."""
    s = tendon_swap_session
    alt_orig_rgba = s._models["alt"].tendon_rgba.copy()
    alt_orig_width = s._models["alt"].tendon_width.copy()

    s.swap_model("alt")
    s.viz.vis_state["tendons"]["enabled"] = True
    s._vis_ctrl[:] = [1.0]
    s.render()
    s.viz.vis_state["tendons"]["enabled"] = False
    s.render()

    assert list(s.model.tendon_rgba.flatten()) == pytest.approx(list(alt_orig_rgba.flatten()))
    assert list(s.model.tendon_width) == pytest.approx(list(alt_orig_width))


def test_swap_model_resets_vis_ctrl_so_the_freshly_swapped_model_renders_inert_not_stale(
    tendon_swap_session,
):
    """Fix round 1: the swap-time reset of ``Session._vis_ctrl`` (inside
    ``_rebuild_tendon_state``) had nothing exercising it in isolation. Every swap test above
    pokes ``_vis_ctrl``/``data.ctrl`` freshly AFTER calling ``swap_model``, so none of them can
    tell "reset on swap" apart from "never reset, but the very next write happens to cover it
    anyway" -- dropping the reset keeps the whole suite green. This test renders BEFORE the
    swap (driving activation up) and again immediately AFTER it, with no new frame written on
    the new model in between, which is exactly the gap a dropped reset falls through: without
    it, a freshly swapped model -- one no recorded replay frame has driven at all -- renders a
    bright, plausible tendon left over from the model that was active before the swap."""
    s = tendon_swap_session
    t_a = _tendon_id(s.model, "t_a")
    s.viz.vis_state["tendons"]["enabled"] = True

    target = s.model.qpos0.copy()
    s.set_qpos(target, ctrl=[0.9, 0.0])
    s.render()
    driven_alpha = float(s.model.tendon_rgba[t_a, 3])
    driven_width = float(s.model.tendon_width[t_a])
    min_alpha = s.viz.vis_state["tendons"]["min_alpha"]
    min_width = s.viz.vis_state["tendons"]["min_width"]
    # Sanity: activation actually moved away from the floor, so the assertions below prove the
    # reset happened rather than vacuously matching a tendon that was already at rest.
    assert driven_alpha > min_alpha
    assert driven_width > min_width

    s.swap_model("alt")
    t_c = _tendon_id(s.model, "t_c")
    s.render()  # no replay frame written on the new model -- this must not carry anything over

    assert s.model.tendon_rgba[t_c, 3] == pytest.approx(min_alpha)
    assert s.model.tendon_width[t_c] == pytest.approx(min_width)
