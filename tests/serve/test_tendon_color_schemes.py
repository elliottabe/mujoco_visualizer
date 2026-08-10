"""``vis_state['tendons']['color_by']`` selects a caller-supplied actuator colour scheme.

The registry is a parameter rather than a palette shipped here: the actuator-name matching that
turns 'mu_T1_28a_left' into a colour is specific to one fly model, and this package is
model-agnostic for tendons the same way it already is for lighting, floor and camera.

The two regressions in this file are the reason the feature is not a one-liner:

1. ``_rebuild_tendon_state`` re-snapshots ``_tendon_orig_rgba`` from the LIVE model. Calling it
   to pick up a scheme change, while activation has already overwritten ``model.tendon_rgba``,
   would capture activated colours as the model's "own" values -- and every later disable
   restores to that snapshot, permanently. Scheme changes must go through
   ``_rebuild_tendon_colors``, which does not snapshot.
2. A scheme can arrive through ``load_settings`` as well as ``apply_render``, so the change
   cannot be detected by hooking one write path.
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
  </worldbody>
  <tendon>
    <spatial name="t_a" width="0.003" rgba="1 0 0 1">
      <site site="anchor_a"/><site site="tip_a"/>
    </spatial>
    <spatial name="t_b" width="0.003" rgba="0 1 0 1">
      <site site="anchor_b"/><site site="tip_b"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="m_a" tendon="t_a" ctrlrange="-1 1"/>
    <motor name="m_b" tendon="t_b" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# 'm_a' -> pure blue, 'm_b' -> pure yellow. Chosen so each channel is exactly 0.0 or 1.0 and an
# assertion cannot pass by rounding.
_SCHEMES = {
    "byname": {
        "color": lambda n: "#0000ff" if n == "m_a" else "#ffff00",
        "group": lambda n: "first" if n == "m_a" else "second",
    },
}


@pytest.fixture
def sess():
    model = mujoco.MjModel.from_xml_string(_XML)
    s = Session(model=model, width=64, height=48, actuator_color_schemes=_SCHEMES)
    yield s
    s.close()


def _tid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)


def test_registered_scheme_colours_tendons_from_the_supplied_function(sess):
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess.viz.vis_state["tendons"]["color_by"] = "byname"
    sess._vis_ctrl[:] = [1.0, 1.0]
    sess.render()
    assert list(sess.model.tendon_rgba[_tid(sess.model, "t_a"), :3]) == pytest.approx([0, 0, 1])
    assert list(sess.model.tendon_rgba[_tid(sess.model, "t_b"), :3]) == pytest.approx([1, 1, 0])


def test_unknown_scheme_falls_back_to_red_and_does_not_raise(sess):
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess.viz.vis_state["tendons"]["color_by"] = "no_such_scheme"
    sess._vis_ctrl[:] = [1.0, 1.0]
    sess.render()
    assert list(sess.model.tendon_rgba[_tid(sess.model, "t_a"), :3]) == pytest.approx(
        [0.85, 0.15, 0.15]
    )


def test_a_session_with_no_registry_is_unchanged(sess):
    """The stock viewer path: no schemes registered at all."""
    model = mujoco.MjModel.from_xml_string(_XML)
    plain = Session(model=model, width=64, height=48)
    try:
        plain.viz.vis_state["tendons"]["enabled"] = True
        plain._vis_ctrl[:] = [1.0, 1.0]
        plain.render()
        assert list(plain.model.tendon_rgba[_tid(plain.model, "t_a"), :3]) == pytest.approx(
            [0.85, 0.15, 0.15]
        )
    finally:
        plain.close()


def test_switching_scheme_mid_session_repaints_without_a_reconnect(sess):
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [1.0, 1.0]
    sess.render()
    before = sess.model.tendon_rgba[_tid(sess.model, "t_a"), :3].copy()
    sess.viz.vis_state["tendons"]["color_by"] = "byname"
    sess.render()
    after = sess.model.tendon_rgba[_tid(sess.model, "t_a"), :3].copy()
    assert not np.allclose(before, after)
    assert list(after) == pytest.approx([0, 0, 1])


def test_scheme_change_while_enabled_does_not_corrupt_the_restore_snapshot(sess):
    """THE load-bearing regression. If a scheme change rebuilds tendon state the way
    __init__/swap_model do, the 'original' rgba snapshot captures activation colours and every
    later disable restores to them instead of to the model's own values."""
    original = sess.model.tendon_rgba.copy()

    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [1.0, 1.0]
    sess.render()                                    # tendon_rgba now holds activation colours
    sess.viz.vis_state["tendons"]["color_by"] = "byname"
    sess.render()                                    # the rebuild happens HERE

    sess.viz.vis_state["tendons"]["enabled"] = False
    sess.render()
    assert list(sess.model.tendon_rgba.flatten()) == pytest.approx(list(original.flatten()))


def test_color_by_arriving_through_apply_render_is_honoured(sess):
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess.apply_render({"tendons.color_by": "byname"})
    sess._vis_ctrl[:] = [1.0, 1.0]
    sess.render()
    assert list(sess.model.tendon_rgba[_tid(sess.model, "t_a"), :3]) == pytest.approx([0, 0, 1])


def test_color_by_arriving_through_load_settings_is_honoured(sess, tmp_path):
    """The D4 path: apply_render is not the only writer of vis_state."""
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess.viz.load_settings({"tendons": {"color_by": "byname"}})
    sess._vis_ctrl[:] = [1.0, 1.0]
    sess.render()
    assert list(sess.model.tendon_rgba[_tid(sess.model, "t_a"), :3]) == pytest.approx([0, 0, 1])


def test_scheme_touches_rgb_only_not_alpha_or_width(sess):
    """color_by must be a pure recolour: alpha and width stay activation-derived."""
    sess.viz.vis_state["tendons"]["enabled"] = True
    sess._vis_ctrl[:] = [0.4, 0.4]
    sess.render()
    t_a = _tid(sess.model, "t_a")
    alpha_before = float(sess.model.tendon_rgba[t_a, 3])
    width_before = float(sess.model.tendon_width[t_a])

    sess.viz.vis_state["tendons"]["color_by"] = "byname"
    sess.render()
    assert float(sess.model.tendon_rgba[t_a, 3]) == pytest.approx(alpha_before)
    assert float(sess.model.tendon_width[t_a]) == pytest.approx(width_before)


def test_a_colour_functions_own_alpha_is_ignored():
    """``apply_tendon_activation`` writes ``base_rgba[act] * [1, 1, 1, alpha]``, so the alpha a
    colour function returns is discarded and replaced by the activation-derived one. Two
    schemes differing ONLY in the alpha they return must render identically -- otherwise a
    palette could quietly override activation brightness, which is the one thing colour must
    not do. Uses rgba 4-tuples because a hex string can't express an alpha at all."""
    model = mujoco.MjModel.from_xml_string(_XML)
    dim = Session(model=model, width=64, height=48, actuator_color_schemes={
        "a": {"color": lambda n: (0.0, 0.0, 1.0, 0.05), "group": lambda n: "g"},
    })
    bright = Session(
        model=mujoco.MjModel.from_xml_string(_XML), width=64, height=48,
        actuator_color_schemes={
            "a": {"color": lambda n: (0.0, 0.0, 1.0, 0.95), "group": lambda n: "g"},
        },
    )
    try:
        for s in (dim, bright):
            s.viz.vis_state["tendons"]["enabled"] = True
            s.viz.vis_state["tendons"]["color_by"] = "a"
            s._vis_ctrl[:] = [0.5, 0.5]
            s.render()
        assert list(dim.model.tendon_rgba.flatten()) == pytest.approx(
            list(bright.model.tendon_rgba.flatten())
        )
    finally:
        dim.close()
        bright.close()


# -- actuators no ctrl column drives are excluded, so their tendons are hidden ----------------
#
# On the fly pair this is the reference ghost: alt_model roughly doubles nu (272 -> 544) and
# ntendon (260 -> 520), the tendon map is built from the ACTIVE model so all 544 actuators enter
# it, but _ctrl_map matches by NAME over the 272 primary names and the ghost's carry a suffix --
# so no primary column resolves to them. _vis_ctrl is only ever written through that map, so
# their activation is zero for the life of the session: 260 dim duplicate tendons drawn on top of
# the real ones, carrying no signal. Stated here without reference to ghosts, because that is how
# it is implemented.

# Renames the tendons' `tendon="..."` transmission-target references on the actuators too --
# not just their own `name="..."` -- otherwise the alt model fails to compile at all (a
# `<motor>` left pointing at a tendon name that no longer exists), before Session is ever
# constructed.
_ALT_XML = _XML.replace('name="t_a"', 'name="t_a_alt"').replace(
    'name="t_b"', 'name="t_b_alt"'
).replace('tendon="t_a"', 'tendon="t_a_alt"').replace(
    'tendon="t_b"', 'tendon="t_b_alt"'
).replace('name="m_a"', 'name="m_a_alt"').replace('name="m_b"', 'name="m_b_alt"')


def test_actuators_absent_from_the_ctrl_map_are_excluded_and_their_tendons_hidden():
    """An actuator no primary ctrl column maps to has structurally-zero activation, so its
    tendon carries no information and must not be drawn."""
    primary = mujoco.MjModel.from_xml_string(_XML)
    alt = mujoco.MjModel.from_xml_string(_ALT_XML)
    s = Session(model=primary, alt_model=alt, width=64, height=48,
                actuator_color_schemes=_SCHEMES)
    try:
        s.swap_model("alt")
        # Every actuator on the alt model is named *_alt, so none of the primary names match.
        assert s._tendon_act_to_ten == {}
        s.viz.vis_state["tendons"]["enabled"] = True
        s.render()
        assert list(s.model.tendon_rgba[:, 3]) == pytest.approx([0.0] * s.model.ntendon)
    finally:
        s.close()


def test_a_single_model_session_drops_nothing(sess):
    """The exclusion must be invisible when every primary name matches -- which is every
    stock, single-model session."""
    assert len(sess._tendon_act_to_ten) == 2


# -- legend aggregates -------------------------------------------------------------------------


def test_scene_message_reports_group_colours_and_counts(sess):
    sess.viz.vis_state["tendons"]["color_by"] = "byname"
    scene = sess.scene_message()
    assert scene["tendon_color_groups"] == {
        "first": {"color": "#0000ff", "count": 1},
        "second": {"color": "#ffff00", "count": 1},
    }
    assert scene["tendon_unclassified"] == 0


def test_legend_counts_only_the_actuators_that_are_actually_drawn():
    """Counted over the ctrl-map-filtered act_to_ten, not over model.nu -- otherwise a
    reference-ghost session would report double what is on screen."""
    primary = mujoco.MjModel.from_xml_string(_XML)
    alt = mujoco.MjModel.from_xml_string(_ALT_XML)
    s = Session(model=primary, alt_model=alt, width=64, height=48,
                actuator_color_schemes=_SCHEMES)
    try:
        s.viz.vis_state["tendons"]["color_by"] = "byname"
        s.swap_model("alt")
        scene = s.scene_message()
        assert scene["tendon_color_groups"] == {}
        assert scene["tendon_unclassified"] == 0
    finally:
        s.close()


def test_unclassified_actuators_are_counted_separately(sess):
    sess._actuator_color_schemes["partial"] = {
        "color": lambda n: "#0000ff" if n == "m_a" else "#888888",
        "group": lambda n: "first" if n == "m_a" else "unknown",
    }
    sess.viz.vis_state["tendons"]["color_by"] = "partial"
    scene = sess.scene_message()
    assert scene["tendon_color_groups"] == {"first": {"color": "#0000ff", "count": 1}}
    assert scene["tendon_unclassified"] == 1


def test_uniform_scheme_reports_no_groups(sess):
    sess.viz.vis_state["tendons"]["color_by"] = "uniform"
    scene = sess.scene_message()
    assert scene["tendon_color_groups"] == {}
    assert scene["tendon_unclassified"] == 0
