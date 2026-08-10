"""Task 15b: two seams for a sibling module's recorded per-frame force-sensor arrows.

Neither seam draws an arrow or reads sensor data -- that is a later task's job, once the
parent repo's pure ``force_arrows.py`` module (task 15a) is wired in. This file only proves:

1. ``Session.scene_modifiers`` is forwarded to ``Visualizer.render_with``'s existing
   ``modify_scene_fns`` on every :meth:`Session.render` call (the LIVE path only -- see
   ``test_export_path_does_not_forward_scene_modifiers`` for why the export path structurally
   cannot honour this yet).
2. ``vis_state['force_arrows']`` exists with an on/off toggle (default OFF), a model-derived
   ``scale`` (never a hardcoded constant -- a fixed default is invisible on a model it was not
   tuned for, exactly like MuJoCo's own native force-arrow scaling on this project's real fly
   model), and a ``radius``. It is registered in ``_VIS_STATE_ROOTS``, round-trips through
   save/load, and tolerates a partial/wholesale-replaced dict without raising.
"""

import mujoco
import pytest

from mujoco_visualizer import Visualizer
from mujoco_visualizer.serve.session import Session
from mujoco_visualizer.serve.protocol import parse_command
from mujoco_visualizer.visualizer import add_arrow_to_scene, default_force_arrow_scale

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.2">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""

# Two models with genuinely different mass (the second body doubles the total, at default
# density) and the same extent -- so default_force_arrow_scale changes across a swap, which is
# exactly what the model-swap tests below need to be able to detect a stale value. Geoms MUST
# sit inside a <body> (not directly in worldbody) -- an unwrapped worldbody geom is static and
# contributes zero to model.body_mass, which is exactly the trap that first made these tests
# assert 1.0 == 1.0 (the zero-mass fallback) instead of exercising the real formula.
_ONE_BOX = """
<mujoco><worldbody>
  <body><joint type="slide"/><geom type="box" size=".1 .1 .1"/></body>
</worldbody></mujoco>
"""
_TWO_BOX = """
<mujoco><worldbody>
  <body><joint type="slide"/><geom type="box" size=".1 .1 .1"/></body>
  <body pos="0 .3 0"><joint type="slide"/><geom type="box" size=".1 .1 .1"/></body>
</worldbody></mujoco>
"""


@pytest.fixture
def sess():
    model = mujoco.MjModel.from_xml_string(_XML)
    s = Session(model=model, width=64, height=48)
    yield s
    s.close()


# -- 1. Session.scene_modifiers reaches the renderer -----------------------------------------


@pytest.mark.gl
def test_a_registered_modifier_actually_draws_into_the_renderers_scene(sess):
    """Asserting the modifier was merely STORED would prove registration, not that it ran --
    this asserts on the renderer's own ``scene.ngeom`` after a real render() call."""
    sess.render()
    baseline_ngeom = sess._renderer.scene.ngeom

    calls = []

    def modifier(scene, data=None, frame_idx=0):
        calls.append(frame_idx)
        add_arrow_to_scene(scene, [0.0, 0.0, 0.0], [0.0, 0.0, 0.1])

    sess.scene_modifiers.append(modifier)
    sess.render()

    assert calls == [0], "the modifier registered on Session.scene_modifiers never ran"
    assert sess._renderer.scene.ngeom == baseline_ngeom + 1, (
        "the modifier ran but did not actually add a geom to the scene the renderer used"
    )


@pytest.mark.gl
def test_removing_the_modifier_stops_the_extra_geom(sess):
    """Companion to the test above: clearing the list must make the arrow disappear, not just
    stop growing -- mjv_updateScene rebuilds the scene from the model every call (verified
    separately), so this also confirms nothing here fights that reset."""
    sess.render()
    baseline_ngeom = sess._renderer.scene.ngeom

    def modifier(scene, data=None, frame_idx=0):
        add_arrow_to_scene(scene, [0.0, 0.0, 0.0], [0.0, 0.0, 0.1])

    sess.scene_modifiers.append(modifier)
    sess.render()
    assert sess._renderer.scene.ngeom == baseline_ngeom + 1

    sess.scene_modifiers.clear()
    sess.render()
    assert sess._renderer.scene.ngeom == baseline_ngeom, (
        "a geom from a since-removed modifier is still showing up -- stale scene state"
    )


@pytest.mark.gl
def test_constructor_argument_seeds_scene_modifiers():
    model = mujoco.MjModel.from_xml_string(_XML)
    calls = []

    def modifier(scene, data=None, frame_idx=0):
        calls.append(1)

    s = Session(model=model, width=32, height=32, scene_modifiers=[modifier])
    try:
        assert s.scene_modifiers == [modifier]
        s.render()
        assert calls == [1]
    finally:
        s.close()


def test_scene_modifiers_defaults_to_an_empty_list(sess):
    assert sess.scene_modifiers == []


@pytest.mark.gl
def test_export_now_forwards_scene_modifiers_and_they_reach_the_rendered_pixels(tmp_path):
    """Task 15c closed the gap the previous version of this test pinned (see git history for
    ``test_export_path_does_not_forward_scene_modifiers``): ``ExportJob`` now accepts
    ``modify_scene_fns`` and forwards it to ``Visualizer.render_with``, exactly like
    ``Session.render`` already does for ``Session.scene_modifiers`` on the live path.

    Asserting the signature merely gained the parameter would prove registration, not that the
    callable ever reached a rendered frame -- so this renders the SAME model/qpos twice, once
    with a modifier that draws a large arrow into the scene and once without, and asserts the
    two PNG outputs differ in actual pixels. That is the only way to tell "the callable is
    stored" apart from "the callable is invoked and its geometry rasterised"."""
    import mujoco
    import numpy as np

    from mujoco_visualizer.serve.export import ExportJob
    from mujoco_visualizer.visualizer import add_arrow_to_scene

    model = mujoco.MjModel.from_xml_string(_XML)
    qpos = model.qpos0.copy().reshape(1, -1)

    def modifier(scene, data=None, frame_idx=0):
        add_arrow_to_scene(scene, [0.0, 0.0, 0.4], [0.0, 0.0, 1.0], radius=0.08)

    plain_dir = tmp_path / "plain"
    plain = ExportJob(
        model, None, {}, qpos, path=plain_dir, fmt="png", width=64, height=48, fps=10,
    )
    plain.start()
    plain.join(timeout=120)
    assert plain.progress()["state"] == "done", plain.progress()

    modded_dir = tmp_path / "modded"
    modded = ExportJob(
        model, None, {}, qpos, path=modded_dir, fmt="png", width=64, height=48, fps=10,
        modify_scene_fns=[modifier],
    )
    modded.start()
    modded.join(timeout=120)
    assert modded.progress()["state"] == "done", modded.progress()

    import imageio.v2 as imageio

    plain_frame = imageio.imread(plain_dir / "frame_00000.png")
    modded_frame = imageio.imread(modded_dir / "frame_00000.png")
    assert not np.array_equal(plain_frame, modded_frame), (
        "ExportJob accepted modify_scene_fns but the callable never reached the rendered pixels"
    )


# -- 2. vis_state['force_arrows'] -------------------------------------------------------------


def test_default_force_arrow_scale_matches_the_documented_formula():
    """The exact formula stated in the task: 0.1 * extent / (total_mass * |gravity_z|)."""
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <option gravity="0 0 -9.81"/>
      <worldbody>
        <body><joint type="slide"/><geom type="box" size=".1 .1 .1" mass="2.0"/></body>
      </worldbody>
    </mujoco>
    """)
    expected = 0.1 * float(model.stat.extent) / (2.0 * 9.81)
    assert default_force_arrow_scale(model) == pytest.approx(expected)


def test_default_force_arrow_scale_differs_across_models_with_different_mass():
    one = mujoco.MjModel.from_xml_string(_ONE_BOX)
    two = mujoco.MjModel.from_xml_string(_TWO_BOX)
    # Same extent isn't guaranteed by construction (the second geom sits away from the
    # origin), so this only asserts what actually must differ: the mass term.
    assert float(two.body_mass.sum()) == pytest.approx(2 * float(one.body_mass.sum()))
    assert default_force_arrow_scale(one) != default_force_arrow_scale(two)


def test_default_force_arrow_scale_falls_back_to_one_on_zero_gravity():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body><joint type="slide"/><geom type="box" size=".1 .1 .1" mass="1.0"/></body>
      </worldbody>
    </mujoco>
    """)
    assert default_force_arrow_scale(model) == 1.0


def test_fresh_vis_state_has_force_arrows_off_with_a_real_positive_scale():
    viz = Visualizer(model=mujoco.MjModel.from_xml_string(_XML))
    try:
        fa = viz.vis_state["force_arrows"]
        assert fa["enabled"] is False
        assert fa["scale"] == pytest.approx(default_force_arrow_scale(viz.model))
        assert fa["scale"] > 0.0, "an invisible (zero) default defeats the whole point"
        assert fa["radius"] > 0.0
    finally:
        viz.close()


def test_default_force_arrow_scale_attribute_is_rebuilt_on_a_model_swap():
    """Model swap requirement: the model-derived FALLBACK is recomputed for whichever model is
    currently active. The live vis_state value is deliberately NOT touched (see the next test)
    -- this one is about the private attribute a future consumer would read instead."""
    one = mujoco.MjModel.from_xml_string(_ONE_BOX)
    two = mujoco.MjModel.from_xml_string(_TWO_BOX)
    viz = Visualizer(model=one)
    try:
        first = viz._default_force_arrow_scale
        assert first == pytest.approx(default_force_arrow_scale(one))
        viz.rebind_model(two)
        second = viz._default_force_arrow_scale
        assert second == pytest.approx(default_force_arrow_scale(two))
        assert second != first
    finally:
        viz.close()


@pytest.mark.gl
def test_swap_model_does_not_overwrite_a_live_force_arrows_scale():
    """Mirrors 'tendons' ctrl_full_scale: a caller's explicit choice survives a model swap
    unchanged, exactly like Session.swap_model already leaves other vis_state groups alone."""
    primary = mujoco.MjModel.from_xml_string(_ONE_BOX)
    alt = mujoco.MjModel.from_xml_string(_TWO_BOX)
    s = Session(model=primary, alt_model=alt, width=32, height=32)
    try:
        s.apply_render({"force_arrows.scale": 0.5})
        assert s.viz.vis_state["force_arrows"]["scale"] == 0.5
        s.swap_model("alt")
        assert s.viz.vis_state["force_arrows"]["scale"] == 0.5, (
            "swap_model silently replaced the caller's explicit scale override"
        )
        # But the model-derived fallback attribute itself did keep up with the swap.
        assert s.viz._default_force_arrow_scale == pytest.approx(
            default_force_arrow_scale(alt)
        )
    finally:
        s.close()


def test_force_arrows_is_a_known_render_set_root():
    cmd = parse_command({"t": "render", "set": {"force_arrows.enabled": True}})
    assert cmd["set"] == {"force_arrows.enabled": True}


def test_force_arrows_dotted_render_set_reaches_vis_state(sess):
    sess.apply_render({"force_arrows.enabled": True})
    assert sess.viz.vis_state["force_arrows"]["enabled"] is True
    # Sibling fields untouched by the dotted (merge-style) write.
    assert sess.viz.vis_state["force_arrows"]["scale"] > 0.0


def test_a_wholesale_replaced_partial_force_arrows_dict_does_not_raise(sess):
    """The hazard this project has shipped before: render.set merges one wire key at a time,
    but a bare (non-dotted) top-level key replaces the WHOLE group -- so a caller sending
    {"force_arrows": {"enabled": True}} (one key, no "scale"/"radius") must not break anything
    downstream that reads this group with .get(...)."""
    sess.apply_render({"force_arrows": {"enabled": True}})
    assert sess.viz.vis_state["force_arrows"] == {"enabled": True}

    # render() itself must not raise -- nothing in this package indexes force_arrows directly.
    sess.render()

    # And the safe-read contract a future consumer relies on still works.
    fa = sess.viz.vis_state["force_arrows"]
    scale = fa.get("scale", sess.viz._default_force_arrow_scale)
    radius = fa.get("radius", 0.003)
    assert scale > 0.0
    assert radius > 0.0


def test_save_load_round_trips_force_arrows(tmp_path):
    """Focused counterpart to test_settings_save.py's whole-vis_state round trip: pins this
    one group by name so a future refactor of that generic test can't quietly stop covering it."""
    user_dir = tmp_path / "user_settings"
    saver = Session(
        model=mujoco.MjModel.from_xml_string(_XML), width=32, height=32,
        user_settings_dir=user_dir,
    )
    try:
        saver.viz.vis_state["force_arrows"] = {
            "enabled": True, "scale": 0.42, "radius": 0.01,
        }
        saver.save_settings_as("force_arrows_probe")
    finally:
        saver.close()

    loader = Session(
        model=mujoco.MjModel.from_xml_string(_XML), width=32, height=32,
        user_settings_dir=user_dir,
    )
    try:
        loader.load_settings("force_arrows_probe")
        assert loader.viz.vis_state["force_arrows"] == {
            "enabled": True, "scale": 0.42, "radius": 0.01,
        }
    finally:
        loader.close()


# -- maxgeom: the guard this seam relies on, unmodified ---------------------------------------


@pytest.mark.gl
def test_add_arrow_to_scene_silently_drops_past_maxgeom(sess):
    """Not new code -- this pins the EXISTING guard's behaviour, since force_arrows leans on
    it: at most 6 legs' worth of arrows is nowhere near exhausting a 10000-maxgeom renderer in
    practice, but a caller stacking many overlays could. Silence-on-overflow (no exception, no
    warning) is the existing, unchanged contract; this test exists so a reviewer sees that
    contract demonstrated rather than taking the docstring's word for it."""
    sess.render()
    scene = sess._renderer.scene
    scene.ngeom = scene.maxgeom  # simulate an already-full scene
    add_arrow_to_scene(scene, [0, 0, 0], [0, 0, 1])
    assert scene.ngeom == scene.maxgeom, "should have silently declined to add past maxgeom"
