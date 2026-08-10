"""``vis_state['forces']`` controls MuJoCo's force/torque arrow scaling.

Unlike every other ``vis_state`` group, these five fields live on ``MjModel.vis`` rather than
on the ``MjvOption`` built per-render (see ``Visualizer._build_scene_option``) -- so there is
no flag to flip in an option object; the apply path writes straight onto the live model.

Two things make this group easy to get wrong, and both are asserted directly against
``model.vis.*`` rather than against ``vis_state`` alone (see the module docstring in
``test_vis_flags.py`` for why: a test that only round-trips through the dict it came from
cannot tell a wired field from a dead one):

1. Initial values MUST come from the model actually loaded, not from MuJoCo's own library
   defaults -- the fly model ships ``map.force = 2e-05``, nothing close to MuJoCo's default
   of ``0.005``, and a hardcoded default would silently overwrite whatever the MJCF set the
   moment a Visualizer is constructed.
2. The apply path must be reachable both on an ordinary render and after a ghost-model swap,
   since the swapped-in model carries its OWN MJCF vis values, not the ones the user had
   dialled in on the model being replaced.
"""

import mujoco
import pytest

from mujoco_visualizer import Visualizer
from mujoco_visualizer.serve.session import Session

# A distinctive map.force (MuJoCo's own default is 0.005) so a test that reads this value back
# off vis_state cannot be satisfied by a hardcoded library default landing there by accident.
_DISTINCTIVE_MAP_FORCE = 6.6

_XML = """
<mujoco>
  <visual>
    <map force="{map_force}" torque="0.25"/>
    <scale forcewidth="0.02" contactwidth="0.4" contactheight="0.05"/>
  </visual>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="b1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _xml(map_force=_DISTINCTIVE_MAP_FORCE):
    return _XML.format(map_force=map_force)


@pytest.fixture
def viz(tmp_path):
    path = tmp_path / "m.xml"
    path.write_text(_xml())
    v = Visualizer(str(path))
    yield v
    v.close()


# -- initialisation must read the model, never hardcode -------------------------------------


def test_vis_state_forces_group_exists_with_all_five_fields(viz):
    forces = viz.vis_state["forces"]
    assert set(forces) == {
        "map_force", "map_torque",
        "scale_forcewidth", "scale_contactwidth", "scale_contactheight",
    }


def test_forces_is_initialised_from_the_models_own_mjcf_values(viz):
    """The load-bearing assertion: this MJCF's map.force (6.6) is nothing close to MuJoCo's
    own library default (0.005) for a bare model. If __init__ ever hardcodes the library
    default instead of reading ``self.model.vis``, this fails immediately with 0.005 instead
    of 6.6 -- see the reversion transcript in the task report."""
    assert viz.vis_state["forces"]["map_force"] == pytest.approx(_DISTINCTIVE_MAP_FORCE)
    assert viz.model.vis.map.force == pytest.approx(_DISTINCTIVE_MAP_FORCE)


def test_forces_other_four_fields_also_come_from_the_mjcf_not_library_defaults(viz):
    forces = viz.vis_state["forces"]
    assert forces["map_torque"] == pytest.approx(0.25)
    assert forces["scale_forcewidth"] == pytest.approx(0.02)
    assert forces["scale_contactwidth"] == pytest.approx(0.4)
    assert forces["scale_contactheight"] == pytest.approx(0.05)


def test_a_model_with_no_explicit_visual_map_still_reads_whatever_default_it_has(tmp_path):
    """Sanity check for the read-from-model path on a model that never sets <visual><map>:
    vis_state must equal whatever MuJoCo actually put on model.vis (its library default),
    not error out and not silently diverge from it."""
    path = tmp_path / "bare.xml"
    path.write_text("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
    v = Visualizer(str(path))
    try:
        assert v.vis_state["forces"]["map_force"] == pytest.approx(float(v.model.vis.map.force))
    finally:
        v.close()


# -- the apply path: vis_state -> model.vis --------------------------------------------------


def test_apply_all_writes_map_force_onto_the_live_model(viz):
    viz.vis_state["forces"]["map_force"] = 1.234
    assert viz.model.vis.map.force != pytest.approx(1.234)  # sanity: not already there
    viz._apply_all()
    assert viz.model.vis.map.force == pytest.approx(1.234)


def test_apply_all_writes_all_five_force_fields_onto_the_live_model(viz):
    viz.vis_state["forces"] = {
        "map_force": 3.0, "map_torque": 4.0,
        "scale_forcewidth": 0.07, "scale_contactwidth": 0.55, "scale_contactheight": 0.09,
    }
    viz._apply_all()
    assert viz.model.vis.map.force == pytest.approx(3.0)
    assert viz.model.vis.map.torque == pytest.approx(4.0)
    assert viz.model.vis.scale.forcewidth == pytest.approx(0.07)
    assert viz.model.vis.scale.contactwidth == pytest.approx(0.55)
    assert viz.model.vis.scale.contactheight == pytest.approx(0.09)


def test_render_with_applies_forces_before_rendering(viz):
    """Exercises the real caller (render_with -> _apply_all), not just the private method
    directly -- the same distinction test_vis_flags.py draws for vis_flags."""
    renderer = viz.make_renderer(height=48, width=64)
    try:
        mujoco.mj_forward(viz.model, viz.data)
        viz.vis_state["forces"]["map_force"] = 9.9
        viz.render_with(renderer)
        assert viz.model.vis.map.force == pytest.approx(9.9)
    finally:
        renderer.close()


# -- save_settings / load_settings must round-trip the group ---------------------------------


def test_save_load_round_trips_forces_between_two_independent_visualizers(tmp_path):
    """Saves from one Visualizer and loads into a SECOND, independent one -- reloading into
    the same live object that already holds the saved value in memory would pass trivially
    even if the group were dropped from one list, because load_settings only merges keys the
    loaded file actually has and never clears what's already there (see the task report and
    tests/serve/test_settings_save.py's identical caution for the sibling ghost feature).
    """
    xml_path = tmp_path / "m.xml"
    xml_path.write_text(_xml())

    saver = Visualizer(str(xml_path))
    try:
        saver.vis_state["forces"]["map_force"] = 42.0
        saver.vis_state["forces"]["scale_contactwidth"] = 0.77
        out = tmp_path / "probe.json"
        saver.save_settings(str(out))
    finally:
        saver.close()

    loader = Visualizer(str(xml_path))
    try:
        # Sanity: before loading, the fresh loader still has the MJCF's own value, not the
        # saver's edit -- otherwise the assertion below could pass by coincidence.
        assert loader.vis_state["forces"]["map_force"] != pytest.approx(42.0)

        loader.load_settings(str(out))
        assert loader.vis_state["forces"]["map_force"] == pytest.approx(42.0)
        assert loader.vis_state["forces"]["scale_contactwidth"] == pytest.approx(0.77)
        assert loader.model.vis.map.force == pytest.approx(42.0)
    finally:
        loader.close()


# -- re-application across a ghost-model swap (Session) --------------------------------------

# Two models with DIFFERENT MJCF map.force so a test can tell "the new model's own value"
# apart from "the value carried over from vis_state".
_PRIMARY_FORCES_XML = """
<mujoco>
  <visual><map force="1.0"/></visual>
  <worldbody><body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1"/></body></worldbody>
</mujoco>
"""

_ALT_FORCES_XML = """
<mujoco>
  <visual><map force="2.0"/></visual>
  <worldbody><body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1"/></body></worldbody>
</mujoco>
"""


@pytest.fixture
def forces_swap_session():
    primary = mujoco.MjModel.from_xml_string(_PRIMARY_FORCES_XML)
    alt = mujoco.MjModel.from_xml_string(_ALT_FORCES_XML)
    s = Session(model=primary, alt_model=alt, width=64, height=48)
    try:
        yield s
    finally:
        s.close()


def test_swap_model_re_applies_the_users_forces_onto_the_new_model(forces_swap_session):
    """The trap this test guards: the ALT model's own MJCF sets map.force=2.0, but the user
    had dialled vis_state['forces']['map_force'] to 5.5 on the primary model. Swapping must
    push the USER's value onto the new model, not leave the new model's own MJCF default
    sitting underneath a vis_state dict that merely claims otherwise."""
    s = forces_swap_session
    assert s.model.vis.map.force == pytest.approx(1.0)  # primary's own MJCF value

    s.viz.vis_state["forces"]["map_force"] = 5.5
    s.swap_model("alt")

    assert s.viz.vis_state["forces"]["map_force"] == pytest.approx(5.5)
    # The load-bearing assertion: not vis_state, but the actual live model the renderer draws.
    assert s.model.vis.map.force == pytest.approx(5.5), (
        f"expected the swap to re-apply the user's map.force (5.5) onto the new model, but "
        f"model.vis.map.force is {s.model.vis.map.force!r} -- looks like the alt model's own "
        f"MJCF default (2.0) leaked through unapplied"
    )


def test_swap_model_back_and_forth_keeps_reapplying_forces(forces_swap_session):
    s = forces_swap_session
    s.viz.vis_state["forces"]["map_force"] = 7.0
    s.swap_model("alt")
    assert s.model.vis.map.force == pytest.approx(7.0)
    s.swap_model("primary")
    assert s.model.vis.map.force == pytest.approx(7.0)


# -- the standalone notebook path (render_settings.apply_settings) must also apply forces ----
#
# render_settings.build_scene_option already flips mjVIS_CONTACTFORCE on via vis_flags, so the
# notebook path can turn contact-force arrows ON -- but without forces support in apply_settings
# they render at whatever scale model.vis happens to hold, which for the fly is exactly the
# invisible-arrow bug this feature exists to fix. Without wiring, this is a second, independent
# way to hit the original bug -- not merely a divergence between two APIs.


def test_apply_settings_writes_forces_onto_the_model():
    from mujoco_visualizer.render_settings import apply_settings

    xml = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    assert model.vis.map.force != pytest.approx(3.3)  # sanity: not already there

    apply_settings(model, {"forces": {
        "map_force": 3.3, "map_torque": 4.4,
        "scale_forcewidth": 0.08, "scale_contactwidth": 0.6, "scale_contactheight": 0.11,
    }})

    assert model.vis.map.force == pytest.approx(3.3)
    assert model.vis.map.torque == pytest.approx(4.4)
    assert model.vis.scale.forcewidth == pytest.approx(0.08)
    assert model.vis.scale.contactwidth == pytest.approx(0.6)
    assert model.vis.scale.contactheight == pytest.approx(0.11)


def test_apply_settings_with_no_forces_key_leaves_the_model_untouched():
    """Behaviour preservation: every settings dict written before this feature existed has no
    'forces' key, and apply_settings must not error or invent a value for it."""
    from mujoco_visualizer.render_settings import apply_settings

    xml = "<mujoco><visual><map force='9.0'/></visual><worldbody>" \
          "<geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)

    apply_settings(model, {})

    assert model.vis.map.force == pytest.approx(9.0)


def test_apply_settings_apply_forces_false_skips_the_write():
    """Mirrors the apply_colors/apply_lighting/apply_floor/apply_skybox opt-out pattern already
    on this function -- forces gets the same on/off knob, not special-cased as always-on."""
    from mujoco_visualizer.render_settings import apply_settings

    xml = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    original = float(model.vis.map.force)

    apply_settings(model, {"forces": {
        "map_force": 3.3, "map_torque": 4.4,
        "scale_forcewidth": 0.08, "scale_contactwidth": 0.6, "scale_contactheight": 0.11,
    }}, apply_forces=False)

    assert model.vis.map.force == pytest.approx(original)
