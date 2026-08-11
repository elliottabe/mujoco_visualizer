"""``Session.reset_render_settings`` returns the RESET_KEYS roots of ``vis_state`` to their
launch values.

The point of the feature is REPLACE semantics. ``load_settings`` merges
(``{**current, **incoming}`` per root), so re-loading the startup preset leaves every edit the
preset does not mention in place -- it looks like a reset and is not one. The test that pins the
difference is ``test_reset_removes_an_entry_the_baseline_never_had``: without it this whole
feature could ship as a merge and every other test here would still pass.

Everything here constructs a real Session, so it needs headless GL -- provided by the root
``conftest.py``, which sets ``MUJOCO_GL=egl`` before any test module imports mujoco, so no
per-module setup is required here.
"""

import copy

import mujoco
import pytest

from mujoco_visualizer.serve.session import RESET_KEYS, Session

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0 0.1 0" size="0.01"/>
      <body name="b1" pos="0.1 0 0">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m0" joint="j0" ctrlrange="-1 1"/>
    <motor name="m1" joint="j1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# One geom fewer than _XML, for the clip-swap test: swapping to a model with fewer geoms is the
# direction _carry_vis_state_across_swap actually prunes in.
_XML_SMALL = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m0" joint="j0" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sess(tmp_path):
    """A Session with BOTH models prebuilt.

    `swap_model` takes a KEY into the Session's own prebuilt `_models` dict ("primary" /
    "alt"), not a model object -- so the small model has to be supplied up front via
    `alt_model=`. Every test here gets both; only the clip-swap test uses the alt.
    """
    p = tmp_path / "m.xml"
    p.write_text(_XML)
    small = tmp_path / "small.xml"
    small.write_text(_XML_SMALL)
    s = Session(
        xml_path=str(p),
        width=64,
        height=64,
        alt_model=mujoco.MjModel.from_xml_path(str(small)),
    )
    yield s
    s.close()


def _mutations(baseline):
    """One flat `apply_render` key per root in RESET_KEYS, each changing that root's value.

    Derived from *baseline* rather than hardcoded where the key set is model-dependent
    (`colors` is keyed by body name). Returned as a dict of {root: (flat_key, new_value)} so a
    failure names which root did not come back.
    """
    colour_key = next(iter(baseline["colors"]))
    return {
        "colors": (f"colors.{colour_key}", "#123456"),
        "geom_colors": ("geom_colors.0", "#654321"),
        "alpha": ("alpha", 0.25),
        "vis_flags": ("vis_flags.wireframe", not baseline["vis_flags"]["wireframe"]),
        "geom_groups": ("geom_groups.4", not baseline["geom_groups"][4]),
        "site_groups": ("site_groups.5", not baseline["site_groups"][5]),
        "lighting": (
            "lighting.use_dual_lighting",
            not baseline["lighting"]["use_dual_lighting"],
        ),
        "floor": ("floor.reflectance", 0.87),
        "skybox": ("skybox.show", not baseline["skybox"]["show"]),
        "ghost": ("ghost.alpha", 0.77),
        "forces": ("forces.scale_forcewidth", 0.055),
        "tendons": ("tendons.max_width", 0.0123),
        "force_arrows": ("force_arrows.scale", 3.5),
    }


def test_mutations_cover_every_reset_key(sess):
    """A guard on the test data, not on the code: if RESET_KEYS grows a root and `_mutations`
    does not, the round-trip test below would silently stop covering it and still pass."""
    assert set(_mutations(sess._reset_baseline)) == set(RESET_KEYS)


def test_reset_restores_every_root_to_its_launch_value(sess):
    """T2. Each root is perturbed through the real `apply_render` path, then reset."""
    baseline = copy.deepcopy(sess._reset_baseline)
    for _root, (key, value) in _mutations(baseline).items():
        sess.apply_render({key: value})

    for root, (key, _value) in _mutations(baseline).items():
        assert sess.viz.vis_state[root] != baseline[root], (
            f"{root} was not actually changed via {key}, so the reset below proves nothing "
            f"about it"
        )

    assert sess.reset_render_settings() is True
    for root in RESET_KEYS:
        assert sess.viz.vis_state[root] == baseline[root], f"{root} did not return to launch"


def test_reset_does_not_write_geom_render_state(sess):
    """R5. `geom_render_state` is a raw gid->rgba cache baked against one model topology;
    `load_settings` already refuses to apply it, and reset must not either."""
    sentinel = {"999": [0.1, 0.2, 0.3, 0.4]}
    sess.viz.vis_state["geom_render_state"] = sentinel
    sess.reset_render_settings()
    assert sess.viz.vis_state["geom_render_state"] == sentinel


def test_reset_removes_an_entry_the_baseline_never_had(sess):
    """T3. THE test that separates a reset from a load.

    `geom_colors` starts empty, so an added override is an entry the baseline does not contain.
    A merge leaves it; a replace drops it. If this test is ever weakened, the feature can ship
    as `load_settings` under a different name.
    """
    assert sess._reset_baseline["geom_colors"] == {}
    sess.apply_render({"geom_colors.1": "#abcdef"})
    assert sess.viz.vis_state["geom_colors"]

    sess.reset_render_settings()
    assert sess.viz.vis_state["geom_colors"] == {}, (
        "reset merged instead of replacing: an override the launch state never had survived"
    )


def test_reset_leaves_the_camera_alone(sess):
    """T4. The Camera tab owns the camera; the Settings tab owns the look. Reset must not move
    the view, delete a saved camera, or disarm a path -- the opposite choice from
    `load_settings`, which replaces the camera outright."""
    # SHORT wire names (az/el/dist). set_camera translates them to vis_state's long names via
    # _CAMERA_WIRE_KEYS; passing `azimuth=` verbatim lands a key the renderer never reads, so
    # the call would appear to work and change nothing.
    sess.set_camera(az=123.0, el=-44.0, dist=0.66)
    sess.save_camera_preset("shot_a")
    sess.set_camera(az=200.0, el=-10.0, dist=0.9)
    sess.save_camera_preset("shot_b")
    # Positional list of preset names, not a dict.
    sess.set_camera_path(["shot_a", "shot_b"], loop=False)

    camera_before = copy.deepcopy(sess.viz.vis_state["camera"])
    presets_before = copy.deepcopy(sess.viz.vis_state["camera_presets"])
    path_before = copy.deepcopy(sess.camera_path)

    sess.reset_render_settings()

    assert sess.viz.vis_state["camera"] == camera_before, "reset moved the camera"
    assert sess.viz.vis_state["camera_presets"] == presets_before, (
        "reset deleted a saved camera preset"
    )
    assert sess.camera_path == path_before, "reset disarmed the camera path"


def test_reset_is_a_no_op_when_nothing_changed(sess):
    """The return value is the signal the client uses to distinguish "reset did nothing because
    you were already at launch state" from "reset failed"."""
    assert sess.reset_render_settings() is False


def test_the_baseline_survives_a_clip_swap(sess):
    """T5. The hazard: `_carry_vis_state_across_swap` MUTATES its argument in place. Handing it
    the baseline would permanently prune the baseline's own `geom_colors` on the first reset
    after a swap, and every later reset would restore less than it should -- silently.

    Reproduced end to end: put a high-id geom override into the BASELINE, swap DOWN to the alt
    model (fewer geoms, so the carry legitimately prunes it), reset there, swap back UP, and
    reset again. The final reset must still restore the override -- which it can only do if the
    baseline was never the thing the carry pruned.

    The override is written into `_reset_baseline` directly rather than through `apply_render`
    + a re-capture: the baseline is captured once in __init__ and there is no re-capture API, so
    this is the honest way to stand up the precondition. What is under test is the restore path,
    not how the baseline came to hold the entry.
    """
    assert sess._models["alt"].ngeom < sess._models["primary"].ngeom, (
        "the alt model must have FEWER geoms than the primary, or the carry never prunes and "
        "this test cannot detect the in-place mutation it exists for"
    )
    high_id = sess._models["primary"].ngeom - 1
    assert high_id >= sess._models["alt"].ngeom, (
        "the chosen geom id must be out of range on the alt model"
    )
    sess._reset_baseline["geom_colors"] = {high_id: "#0f0f0f"}

    sess.swap_model("alt")
    sess.reset_render_settings()
    assert sess.viz.vis_state["geom_colors"] == {}, (
        "an override past the alt model's ngeom was kept, which is what the carry exists to "
        "prune"
    )

    sess.swap_model("primary")
    sess.reset_render_settings()
    assert sess.viz.vis_state["geom_colors"] == {high_id: "#0f0f0f"}, (
        "the baseline was mutated in place by the earlier reset -- it no longer holds the "
        "override it was captured with"
    )


def test_a_later_edit_cannot_reach_back_into_the_baseline(sess):
    """R6. The real erosion path -- and NOT the one the brief originally described.

    `vis_state.update(restored)` aliases the baseline's inner dicts into `vis_state` unless
    `restored` was deep-copied, and then an ordinary later `apply_render` writes through the
    shared dict and silently rewrites the launch state. Every subsequent reset would restore
    the edited value, with nothing to say the baseline had moved.

    The carry is NOT the hazard here: `_carry_vis_state_across_swap` reassigns
    `vis_state["geom_colors"]` rather than mutating the inner dict, so a shallow copy already
    survives it. Only aliasing through `update` erodes the baseline.
    """
    launch = copy.deepcopy(sess._reset_baseline)
    sess.reset_render_settings()
    sess.apply_render({"floor.reflectance": 0.99, "ghost.alpha": 0.11})
    sess.apply_render({"geom_colors.0": "#abcdef"})
    assert sess._reset_baseline == launch, (
        "an edit made after a reset reached back into the baseline -- vis_state is aliasing "
        "the baseline's inner dicts, so the launch state is no longer what the viewer launched "
        "with"
    )
