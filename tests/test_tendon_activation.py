"""Muscle-tendon activation colouring/thickening: the offline algorithm that used to live
entirely inline in :meth:`Visualizer.render_video_pan`.

``apply_tendon_activation`` and ``build_actuator_tendon_map`` are the module-level functions
extracted out of that method (see ``visualizer.py``). This file pins:

1. The two functions do the real work (correct map, correct alpha/width numbers).
2. ``render_video_pan`` actually DELEGATES to them -- not merely calls a function with the
   right name while secretly keeping its own inline copy. That distinction only shows up when
   the extracted function is broken and the caller's behaviour breaks with it (see
   ``test_making_apply_tendon_activation_a_noop_breaks_render_video_pan`` below).
3. ``vis_state['tendons']`` reads its width defaults from the model, not from a hardcoded
   constant.
"""

import numpy as np
import mujoco
import pytest

from mujoco_visualizer import Visualizer
from mujoco_visualizer.visualizer import (
    apply_tendon_activation,
    build_actuator_tendon_map,
    default_tendon_ctrl_full_scale,
)

# Two tendon-driving actuators (t_a wide, t_b narrow -- distinctive, not MuJoCo's own default
# tendon width of 0.003 for both) plus a THIRD tendon with no actuator at all, to exercise the
# "hide non-muscle tendons" half of the algorithm.
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
    <motor name="m_joint" joint="slide_free" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def viz(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(_XML)
    v = Visualizer(str(path))
    yield v
    v.close()


def _tendon_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)


# -- build_actuator_tendon_map ----------------------------------------------------------------


def test_build_actuator_tendon_map_only_includes_tendon_driven_actuators(viz):
    act_to_ten, base_rgba = build_actuator_tendon_map(viz.model)
    m_a = mujoco.mj_name2id(viz.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_a")
    m_b = mujoco.mj_name2id(viz.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_b")
    m_joint = mujoco.mj_name2id(viz.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_joint")

    assert set(act_to_ten) == {m_a, m_b}  # m_joint drives a joint, not a tendon
    assert act_to_ten[m_a] == _tendon_id(viz.model, "t_a")
    assert act_to_ten[m_b] == _tendon_id(viz.model, "t_b")
    assert base_rgba.shape == (viz.model.nu, 4)


def test_build_actuator_tendon_map_uses_the_color_fn(viz):
    def color_fn(name):
        return "#00ff00" if name == "m_a" else "#0000ff"

    act_to_ten, base_rgba = build_actuator_tendon_map(viz.model, color_fn)
    m_a = mujoco.mj_name2id(viz.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_a")
    m_b = mujoco.mj_name2id(viz.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_b")
    assert list(base_rgba[m_a]) == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert list(base_rgba[m_b]) == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_build_actuator_tendon_map_falls_back_to_solid_red_with_no_color_fn(viz):
    act_to_ten, base_rgba = build_actuator_tendon_map(viz.model)
    m_a = mujoco.mj_name2id(viz.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_a")
    assert list(base_rgba[m_a]) == pytest.approx([0.85, 0.15, 0.15, 1.0])


def test_build_actuator_tendon_map_does_not_mutate_the_model(viz):
    before_rgba = viz.model.tendon_rgba.copy()
    before_width = viz.model.tendon_width.copy()
    build_actuator_tendon_map(viz.model)
    assert list(viz.model.tendon_rgba.flatten()) == pytest.approx(list(before_rgba.flatten()))
    assert list(viz.model.tendon_width) == pytest.approx(list(before_width))


# -- apply_tendon_activation --------------------------------------------------------------------


def test_apply_tendon_activation_hides_the_non_muscle_tendon(viz):
    act_to_ten, base_rgba = build_actuator_tendon_map(viz.model)
    apply_tendon_activation(viz.model, np.array([0.5, 0.5, 0.5]), act_to_ten, base_rgba)
    t_free = _tendon_id(viz.model, "t_free")
    assert viz.model.tendon_rgba[t_free, 3] == pytest.approx(0.0)


def test_apply_tendon_activation_colours_muscle_tendons_from_ctrl(viz):
    act_to_ten, base_rgba = build_actuator_tendon_map(viz.model)
    apply_tendon_activation(
        viz.model, np.array([1.0, 1.0, 0.0]), act_to_ten, base_rgba, ctrl_max=1.0,
    )
    t_a = _tendon_id(viz.model, "t_a")
    assert viz.model.tendon_rgba[t_a, 3] == pytest.approx(1.0)
    assert viz.model.tendon_width[t_a] == pytest.approx(0.003)  # default max width


def test_apply_tendon_activation_varies_with_ctrl():
    """The load-bearing reversion test for 'activation actually varies with ctrl' -- see the
    task report for the verbatim before/after this guards."""
    model = mujoco.MjModel.from_xml_string(_XML)
    act_to_ten, base_rgba = build_actuator_tendon_map(model)
    t_a = _tendon_id(model, "t_a")

    apply_tendon_activation(model, np.array([0.1, 0.1, 0.0]), act_to_ten, base_rgba, ctrl_max=1.0)
    low_alpha = float(model.tendon_rgba[t_a, 3])
    low_width = float(model.tendon_width[t_a])

    apply_tendon_activation(model, np.array([0.9, 0.9, 0.0]), act_to_ten, base_rgba, ctrl_max=1.0)
    high_alpha = float(model.tendon_rgba[t_a, 3])
    high_width = float(model.tendon_width[t_a])

    assert high_alpha > low_alpha
    assert high_width > low_width


def test_apply_tendon_activation_respects_tendon_baseline():
    """Baseline blends into the SAME ``norm`` that drives BOTH alpha and width (see the
    function's docstring) -- asserting only alpha here would leave the width half of that
    contract unguarded. That gap was real: the reviewer changed the width line from
    ``tendon_min_width + width_range * norm`` to ``... * raw`` (dropping the baseline blend
    from width alone) and every existing test, including an earlier version of this one that
    checked alpha only, stayed green. See the task report's "Fix round 2" section for the
    verbatim failure once width is asserted here too."""
    model = mujoco.MjModel.from_xml_string(_XML)
    act_to_ten, base_rgba = build_actuator_tendon_map(model)
    t_a = _tendon_id(model, "t_a")

    apply_tendon_activation(
        model, np.array([0.0, 0.0, 0.0]), act_to_ten, base_rgba,
        ctrl_max=1.0, tendon_baseline=0.3, tendon_alpha_min=0.0,
    )
    # ctrl == 0 with baseline 0.3 must land at exactly the baseline, not at alpha_min/zero.
    assert model.tendon_rgba[t_a, 3] == pytest.approx(0.3)
    # Width: same norm (0.3), interpolated between the function's own defaults
    # (tendon_min_width=0.0005, tendon_width=0.003) -- not left at tendon_min_width, which is
    # exactly what dropping the baseline blend from the width line alone would produce (norm
    # would still be reported via alpha, but width would use raw=0.0 instead).
    expected_width = 0.0005 + (0.003 - 0.0005) * 0.3
    assert model.tendon_width[t_a] == pytest.approx(expected_width)


def test_apply_tendon_activation_floors_alpha_at_alpha_min():
    model = mujoco.MjModel.from_xml_string(_XML)
    act_to_ten, base_rgba = build_actuator_tendon_map(model)
    t_a = _tendon_id(model, "t_a")

    apply_tendon_activation(
        model, np.array([0.0, 0.0, 0.0]), act_to_ten, base_rgba,
        ctrl_max=1.0, tendon_alpha_min=0.2,
    )
    assert model.tendon_rgba[t_a, 3] == pytest.approx(0.2)


def test_apply_tendon_activation_is_idempotent_no_snapshot_needed_between_calls():
    """apply_tendon_activation re-hides non-muscle tendons on every call -- a caller does not
    need to separately hide them once up front before the first frame."""
    model = mujoco.MjModel.from_xml_string(_XML)
    act_to_ten, base_rgba = build_actuator_tendon_map(model)
    t_free = _tendon_id(model, "t_free")

    for _ in range(3):
        apply_tendon_activation(model, np.array([0.4, 0.6, 0.0]), act_to_ten, base_rgba)
    assert model.tendon_rgba[t_free, 3] == pytest.approx(0.0)


# -- vis_state['tendons'] must read model.tendon_width, never hardcode -------------------------


def test_vis_state_tendons_group_exists_with_all_seven_fields(viz):
    tendons = viz.vis_state["tendons"]
    assert set(tendons) == {
        "enabled", "max_width", "min_width", "min_alpha", "baseline", "ctrl_full_scale",
        "color_by",
    }


def test_vis_state_tendons_width_defaults_come_from_the_models_own_tendon_width(viz):
    """The load-bearing assertion: this MJCF's tendon widths (0.00015 .. 0.003) are exactly
    what render_video_pan's own long-standing keyword defaults already are, which would let a
    hardcode masquerade as 'reading the model' -- so the widths are deliberately DISTINCT from
    those defaults here as well as spanning them, and the true min/max is asserted against
    ``model.tendon_width`` directly, not against a second hardcoded literal in this test."""
    assert viz.vis_state["tendons"]["max_width"] == pytest.approx(float(viz.model.tendon_width.max()))
    assert viz.vis_state["tendons"]["min_width"] == pytest.approx(float(viz.model.tendon_width.min()))
    assert viz.vis_state["tendons"]["max_width"] == pytest.approx(0.003)
    assert viz.vis_state["tendons"]["min_width"] == pytest.approx(0.00015)


def test_vis_state_tendons_defaults_to_disabled(viz):
    assert viz.vis_state["tendons"]["enabled"] is False


# -- ctrl_full_scale: a TUNABLE reference, not a measured max (fix round 1) --------------------
#
# ``actuator_ctrlrange`` is a THEORETICAL ceiling: on the real fly model it is 1.05 across all
# 258 tendon-driving actuators, but a trained policy's actual |ctrl| occupies a small fraction
# of it (see default_tendon_ctrl_full_scale's docstring for the measured percentiles). This
# fixture's actuators (ctrlrange -1..1) give a default of 1.0 -- deliberately not 1.05, so a
# test asserting "1.0" here cannot be satisfied by accidentally hardcoding the real fly's own
# number instead of actually deriving it from ctrlrange.


def test_default_tendon_ctrl_full_scale_matches_the_models_own_ctrlrange(viz):
    act_to_ten, _ = build_actuator_tendon_map(viz.model)
    assert default_tendon_ctrl_full_scale(viz.model, act_to_ten) == pytest.approx(1.0)
    # Also correct with no precomputed map supplied (builds its own internally).
    assert default_tendon_ctrl_full_scale(viz.model) == pytest.approx(1.0)


def test_default_tendon_ctrl_full_scale_uses_the_larger_of_the_two_ctrlrange_bounds():
    xml = """
    <mujoco>
      <worldbody>
        <site name="a" pos="0 0 0.4" size="0.01"/>
        <body name="box" pos="0 0 0.6">
          <joint name="j" type="slide" axis="0 0 1"/>
          <geom type="box" size="0.05 0.05 0.05"/>
          <site name="tip" pos="0 0 0" size="0.01"/>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t"><site site="a"/><site site="tip"/></spatial>
      </tendon>
      <actuator>
        <motor name="m" tendon="t" ctrlrange="-1.05 1"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    # Mirrors the real fly's own asymmetric ctrlrange (-1.05 .. 1) -- the max of the two
    # |bounds| (1.05), not just the upper bound (1) or the lower bound's magnitude alone.
    assert default_tendon_ctrl_full_scale(model) == pytest.approx(1.05)


def test_default_tendon_ctrl_full_scale_falls_back_to_one_with_no_limited_actuators():
    xml = """
    <mujoco>
      <worldbody>
        <site name="a" pos="0 0 0.4" size="0.01"/>
        <body name="box" pos="0 0 0.6">
          <joint name="j" type="slide" axis="0 0 1"/>
          <geom type="box" size="0.05 0.05 0.05"/>
          <site name="tip" pos="0 0 0" size="0.01"/>
        </body>
      </worldbody>
      <tendon>
        <spatial name="t"><site site="a"/><site site="tip"/></spatial>
      </tendon>
      <actuator>
        <motor name="m" tendon="t"/>
      </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    assert bool(model.actuator_ctrllimited[0]) is False  # sanity: genuinely unlimited
    assert default_tendon_ctrl_full_scale(model) == pytest.approx(1.0)


def test_vis_state_tendons_ctrl_full_scale_defaults_to_the_ctrlrange_derived_value(viz):
    """The load-bearing assertion for fix round 1: vis_state must be populated with the
    computed default at construction time, not a bare hardcoded 1.0 that happens to coincide
    with it for this fixture."""
    assert viz.vis_state["tendons"]["ctrl_full_scale"] == pytest.approx(
        default_tendon_ctrl_full_scale(viz.model)
    )
    assert viz.vis_state["tendons"]["ctrl_full_scale"] == pytest.approx(1.0)


def test_a_model_with_no_tendons_still_gets_sane_tendons_defaults(tmp_path):
    path = tmp_path / "bare.xml"
    path.write_text("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
    v = Visualizer(str(path))
    try:
        assert v.vis_state["tendons"]["max_width"] == pytest.approx(0.003)
        assert v.vis_state["tendons"]["min_width"] == pytest.approx(0.0005)
        assert v.vis_state["tendons"]["ctrl_full_scale"] == pytest.approx(1.0)
    finally:
        v.close()


# -- save_settings / load_settings must round-trip the group -----------------------------------


def test_save_load_round_trips_tendons_between_two_independent_visualizers(tmp_path):
    xml_path = tmp_path / "m.xml"
    xml_path.write_text(_XML)

    saver = Visualizer(str(xml_path))
    try:
        saver.vis_state["tendons"]["enabled"] = True
        saver.vis_state["tendons"]["baseline"] = 0.42
        out = tmp_path / "probe.json"
        saver.save_settings(str(out))
    finally:
        saver.close()

    loader = Visualizer(str(xml_path))
    try:
        assert loader.vis_state["tendons"]["enabled"] is False  # sanity: fresh default
        loader.load_settings(str(out))
        assert loader.vis_state["tendons"]["enabled"] is True
        assert loader.vis_state["tendons"]["baseline"] == pytest.approx(0.42)
    finally:
        loader.close()


def test_load_settings_with_a_partial_tendons_dict_does_not_raise_and_keeps_the_rest(viz):
    viz.vis_state["tendons"]["min_alpha"] = 0.9
    viz.load_settings({"tendons": {"enabled": True}})
    assert viz.vis_state["tendons"]["enabled"] is True
    assert viz.vis_state["tendons"]["min_alpha"] == pytest.approx(0.9)  # untouched, not reset


# -- render_video_pan must call the extracted functions, not an inline copy --------------------


def _qposes_and_cameras(viz, n=3):
    qposes = np.tile(viz.model.qpos0, (n, 1))
    cameras = [viz.get_camera(None) for _ in range(n)]
    return qposes, cameras


def _tendon_states_during_render_video_pan(viz, qposes, cameras, ctrls):
    """Run the real ``render_video_pan`` call, but capture ``model.tendon_rgba``/
    ``tendon_width`` at the moment each frame is actually rendered (inside ``render_with``,
    which the muscle-vis block calls immediately after mutating them) -- not the final pixels,
    which depend on camera framing/anti-aliasing at the tiny resolution these tests use, and not
    the model state after the call returns, which render_video_pan always restores to the
    original regardless of what happened frame-to-frame."""
    captured = []
    orig_render_with = viz.render_with

    def _spy(renderer, **kw):
        captured.append((viz.model.tendon_rgba.copy(), viz.model.tendon_width.copy()))
        return orig_render_with(renderer, **kw)

    viz.render_with = _spy
    try:
        viz.render_video_pan(qposes, cameras, height=48, width=64, ctrls=ctrls)
    finally:
        del viz.render_with  # drop the instance override, restoring the bound method
    return captured


def test_render_video_pan_produces_different_frames_for_different_ctrl_vectors(viz):
    """Bullet 2 of the reversion suite, exercised through the real render_video_pan call (not
    just the extracted function in isolation, which test_apply_tendon_activation_varies_with_
    ctrl above already covers). Uses a clip with a low-ctrl frame and a high-ctrl frame in the
    SAME call rather than two separate calls with a constant ctrl each -- render_video_pan
    normalises by the whole clip's own max (see its docstring), so two separate constant clips
    would each trivially normalise their single repeated value to 1.0 and prove nothing."""
    qposes, cameras = _qposes_and_cameras(viz)
    t_a = _tendon_id(viz.model, "t_a")
    ctrls = np.array([[0.05, 0.05, 0.0], [0.5, 0.5, 0.0], [0.95, 0.95, 0.0]])

    states = _tendon_states_during_render_video_pan(viz, qposes, cameras, ctrls)

    low_alpha = states[0][0][t_a, 3]
    high_alpha = states[2][0][t_a, 3]
    assert high_alpha > low_alpha


def test_render_video_pan_restores_original_tendon_state_after_the_clip(viz):
    orig_rgba = viz.model.tendon_rgba.copy()
    orig_width = viz.model.tendon_width.copy()
    qposes, cameras = _qposes_and_cameras(viz)
    ctrls = np.tile([0.9, 0.9, 0.0], (3, 1))

    viz.render_video_pan(qposes, cameras, height=48, width=64, ctrls=ctrls)

    assert list(viz.model.tendon_rgba.flatten()) == pytest.approx(list(orig_rgba.flatten()))
    assert list(viz.model.tendon_width) == pytest.approx(list(orig_width))


def test_making_apply_tendon_activation_a_noop_breaks_render_video_pan(viz, monkeypatch):
    """The load-bearing extraction-is-real test: if render_video_pan kept its own inline copy
    of the algorithm instead of delegating to ``apply_tendon_activation``, patching that name
    in ``visualizer`` to a no-op would have NO effect on what render_video_pan actually draws
    onto the model each frame. See the task report for the verbatim result of running this
    against a deliberately-reverted (inline-copy) version of render_video_pan."""
    import mujoco_visualizer.visualizer as viz_module

    qposes, cameras = _qposes_and_cameras(viz)
    t_a = _tendon_id(viz.model, "t_a")
    ctrls = np.array([[0.05, 0.05, 0.0], [0.5, 0.5, 0.0], [0.95, 0.95, 0.0]])

    real_states = _tendon_states_during_render_video_pan(viz, qposes, cameras, ctrls)
    real_low_alpha = float(real_states[0][0][t_a, 3])  # low-ctrl frame: real alpha << 1.0

    monkeypatch.setattr(viz_module, "apply_tendon_activation", lambda *a, **k: None)
    noop_states = _tendon_states_during_render_video_pan(viz, qposes, cameras, ctrls)
    # With the no-op in place, model.tendon_rgba is never touched during the loop, so every
    # captured frame just shows this XML's own untouched alpha (1.0) -- not this call's ctrl.
    noop_low_alpha = float(noop_states[0][0][t_a, 3])

    assert real_low_alpha != pytest.approx(noop_low_alpha)
