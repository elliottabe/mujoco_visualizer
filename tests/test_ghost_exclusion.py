"""Category colours must not reach the excluded (ghost) half of a doubled model.

Asserted on model.geom_rgba rather than on pixels: a pixel test can pass or fail for
framing reasons, while the rgba array is the thing the render reads.
"""
import mujoco
import pytest

from mujoco_visualizer import Visualizer

_XML = """
<mujoco><worldbody><light pos="0 0 2"/>
  <body name="thorax"><geom name="thorax_geom" type="box" size=".1 .1 .1" rgba=".5 .3 .1 1"/></body>
  <body name="thorax-ghost" pos="0 .5 0">
    <geom name="thorax_geom-ghost" type="box" size=".1 .1 .1" rgba=".8 .8 .8 .3"/></body>
</worldbody></mujoco>
"""

ANATOMY = {"categories": [{"name": "thorax", "match": {"body_substring": ["thorax"]}}]}


def _viz(**kw):
    return Visualizer(model=mujoco.MjModel.from_xml_string(_XML), anatomy=ANATOMY, **kw)


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def test_without_a_suffix_both_geoms_take_the_category_colour():
    viz = _viz()
    viz.vis_state["colors"]["thorax"] = "#ff0000"
    viz._apply_all()
    pol = viz.model.geom_rgba[_gid(viz.model, "thorax_geom")]
    ghost = viz.model.geom_rgba[_gid(viz.model, "thorax_geom-ghost")]
    assert pol[0] == pytest.approx(1.0)
    assert ghost[0] == pytest.approx(1.0), "baseline: with no exclusion both are recoloured"


def test_excluded_suffix_keeps_the_ghost_off_the_category_colour():
    viz = _viz(excluded_suffix="-ghost")
    viz.vis_state["colors"]["thorax"] = "#ff0000"
    viz._apply_all()
    pol = viz.model.geom_rgba[_gid(viz.model, "thorax_geom")]
    ghost = viz.model.geom_rgba[_gid(viz.model, "thorax_geom-ghost")]
    assert pol[0] == pytest.approx(1.0), "the policy half must still recolour"
    assert ghost[0] != pytest.approx(1.0), "the ghost half must NOT take the category colour"


def test_ghost_tint_and_alpha_drive_the_excluded_geoms():
    viz = _viz(excluded_suffix="-ghost")
    viz.vis_state["ghost"]["tint"] = "#0000ff"
    viz.vis_state["ghost"]["alpha"] = 0.15
    viz._apply_all()
    ghost = viz.model.geom_rgba[_gid(viz.model, "thorax_geom-ghost")]
    assert ghost[2] == pytest.approx(1.0), "blue channel follows ghost.tint"
    assert ghost[3] == pytest.approx(0.15), "alpha follows ghost.alpha"


def test_ghost_alpha_replaces_rather_than_multiplies_the_global_alpha():
    viz = _viz(excluded_suffix="-ghost")
    viz.vis_state["alpha"] = 0.5
    viz.vis_state["ghost"]["alpha"] = 0.20
    viz._apply_all()
    ghost = viz.model.geom_rgba[_gid(viz.model, "thorax_geom-ghost")]
    assert ghost[3] == pytest.approx(0.20), "0.20, not 0.5*0.20 -- the box value is what you get"


def test_default_vis_state_carries_a_ghost_group():
    viz = _viz()
    assert viz.vis_state["ghost"] == {"tint": "#cccccc", "alpha": 0.3}


def test_ghost_tint_wins_over_a_per_geom_override_on_an_excluded_geom():
    viz = _viz(excluded_suffix="-ghost")
    gid = _gid(viz.model, "thorax_geom-ghost")
    viz.vis_state["geom_colors"][gid] = "#00ff00"
    viz.vis_state["ghost"]["tint"] = "#0000ff"
    viz._apply_all()
    ghost = viz.model.geom_rgba[gid]
    assert ghost[2] == pytest.approx(1.0), "ghost tint (blue) wins"
    assert ghost[1] != pytest.approx(1.0), "the per-geom override (green) must be ignored"
