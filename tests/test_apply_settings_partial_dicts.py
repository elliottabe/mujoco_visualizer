"""``render_settings.apply_settings`` must accept a PARTIAL group dict for every group it
handles, not just ``forces`` (fixed in a previous round -- see ``tests/test_forces_vis.py``).

After that fix, the function had two contradictory contracts: a partial dict works for
``forces``/``colors``/``alpha``/``geom_colors``, and crashes with a bare ``KeyError`` for
``lighting.headlight``, a ``lighting.lights`` entry, ``floor``, and ``skybox`` -- one function,
two rules, and a user who learns partial dicts work from tuning ``forces`` hits a ``KeyError``
the first time they tune the floor. This file pins present-keys-only semantics for those four
remaining groups: whichever field the caller mentions changes; whichever field they don't
mention keeps the model's own current value (never a MuJoCo library default -- see
``visualizer.py``'s ``_apply_forces_vis`` docstring for why filling in a default would just
relocate the hardcode hazard requirement 1 of this feature exists to prevent).

Design decisions recorded here (see the task-12 report's "Fix round 3" section for the fuller
reasoning):

- ``lighting.lights`` is a LIST of dicts, not a flat group. A caller supplying fewer entries
  than the model has lights was already fine (the loop only touches indices actually present)
  -- the bug was a partial dict WITHIN one entry, e.g. ``{'active': True}`` alone.
- A light entry's ``dir_az``/``dir_el`` and floor's ``texrepeat_x``/``texrepeat_y`` are each a
  PAIR that jointly determines one underlying model field (a direction vector, a 2-vector).
  Mentioning only one half of a pair decomposes the model's CURRENT value back into the pair
  (via ``_dir_to_az_el``, the exact inverse of the direction write) before re-composing it with
  the mentioned half substituted in -- not "leave the whole pair untouched", since the mentioned
  half must still take effect.
- ``skybox``'s ``sky_top``/``sky_bot`` have no dedicated model field to read back at all --
  ``_make_sky_pixels`` bakes both into every texel of a rendered cube map. Recovering "the
  current value of the one not mentioned" means sampling the texture (see
  ``visualizer._read_sky_colors``), not indexing a struct. Verified independently below by
  sampling pixels directly in the test rather than trusting the new production helper.
"""

import numpy as np
import mujoco
import pytest

from mujoco_visualizer.render_settings import apply_settings
from mujoco_visualizer.visualizer import _hex_to_rgb, _dir_to_az_el


# -- headlight -------------------------------------------------------------------------------


def test_apply_settings_partial_headlight_dict_updates_only_the_mentioned_field():
    xml = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    orig_ambient = list(map(float, model.vis.headlight.ambient))
    orig_diffuse = list(map(float, model.vis.headlight.diffuse))
    orig_specular = list(map(float, model.vis.headlight.specular))
    orig_active = int(model.vis.headlight.active)
    new_active = 0 if orig_active else 1
    assert new_active != orig_active  # sanity: the mentioned field will actually move

    apply_settings(model, {"lighting": {"headlight": {"active": new_active}}})

    assert int(model.vis.headlight.active) == new_active  # mentioned: changed
    # unmentioned: kept the model's own values, not a MuJoCo library default
    assert list(map(float, model.vis.headlight.ambient)) == pytest.approx(orig_ambient)
    assert list(map(float, model.vis.headlight.diffuse)) == pytest.approx(orig_diffuse)
    assert list(map(float, model.vis.headlight.specular)) == pytest.approx(orig_specular)


# -- lighting.lights (one entry, partial) -----------------------------------------------------

_ONE_LIGHT_XML = """
<mujoco><worldbody>
  <light name="l0" pos="0 0 1" dir="0.3 0.4 -0.8"
         ambient="0.11 0.12 0.13" diffuse="0.21 0.22 0.23" specular="0.31 0.32 0.33"/>
  <geom type="box" size=".1 .1 .1"/>
</worldbody></mujoco>
"""


def test_apply_settings_partial_lights_entry_updates_only_the_mentioned_field():
    model = mujoco.MjModel.from_xml_string(_ONE_LIGHT_XML)
    orig_ambient = list(map(float, model.light_ambient[0]))
    orig_diffuse = list(map(float, model.light_diffuse[0]))
    orig_specular = list(map(float, model.light_specular[0]))
    orig_dir = list(map(float, model.light_dir[0]))
    orig_active = bool(model.light_active[0])
    new_active = not orig_active

    apply_settings(model, {"lighting": {"lights": [{"active": new_active}]}})

    assert bool(model.light_active[0]) == new_active  # mentioned: changed
    # unmentioned: kept the model's own values
    assert list(map(float, model.light_ambient[0])) == pytest.approx(orig_ambient)
    assert list(map(float, model.light_diffuse[0])) == pytest.approx(orig_diffuse)
    assert list(map(float, model.light_specular[0])) == pytest.approx(orig_specular)
    assert list(map(float, model.light_dir[0])) == pytest.approx(orig_dir, abs=1e-6)


def test_apply_settings_lights_entry_dir_az_alone_moves_azimuth_but_keeps_elevation():
    """The trickier half of the design decision above: dir_az/dir_el is a PAIR backing one
    model field (light_dir), so mentioning only dir_az must decompose the model's current
    direction, substitute the new azimuth, and recompose -- not leave the whole direction
    untouched (that would silently drop the mentioned field) and not zero out elevation either
    (that would silently move the unmentioned field)."""
    model = mujoco.MjModel.from_xml_string(_ONE_LIGHT_XML)
    orig_az, orig_el = _dir_to_az_el(model.light_dir[0])
    new_az = (orig_az + 90.0) % 360.0

    apply_settings(model, {"lighting": {"lights": [{"dir_az": new_az}]}})

    got_az, got_el = _dir_to_az_el(model.light_dir[0])
    assert got_az == pytest.approx(new_az, abs=1e-4)     # mentioned: changed
    assert got_el == pytest.approx(orig_el, abs=1e-4)    # unmentioned: kept


# -- floor -----------------------------------------------------------------------------------

_FLOOR_XML = """
<mujoco>
  <asset>
    <material name="floormat" rgba="0.3 0.35 0.4 0.9"
              texrepeat="2 3" reflectance="0.22" shininess="0.44" emission="0.05"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" material="floormat" rgba="0.3 0.35 0.4 0.9"/>
    <geom type="box" size=".1 .1 .1"/>
  </worldbody>
</mujoco>
"""


def test_apply_settings_partial_floor_dict_updates_only_the_mentioned_field():
    model = mujoco.MjModel.from_xml_string(_FLOOR_XML)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_mat_id = int(model.geom_matid[floor_gid])
    orig_alpha = float(model.geom_rgba[floor_gid, 3])
    orig_texrepeat = list(map(float, model.mat_texrepeat[floor_mat_id]))
    orig_reflectance = float(model.mat_reflectance[floor_mat_id])
    orig_shininess = float(model.mat_shininess[floor_mat_id])
    orig_emission = float(model.mat_emission[floor_mat_id])

    apply_settings(model, {"floor": {"color": "#112233"}})

    # mentioned: changed
    assert list(model.geom_rgba[floor_gid, :3]) == pytest.approx(
        _hex_to_rgb("#112233"), abs=1 / 255
    )
    assert list(model.mat_rgba[floor_mat_id, :3]) == pytest.approx(
        _hex_to_rgb("#112233"), abs=1 / 255
    )
    # unmentioned: kept the model's own values, not a library default
    assert model.geom_rgba[floor_gid, 3] == pytest.approx(orig_alpha)
    assert model.mat_rgba[floor_mat_id, 3] == pytest.approx(orig_alpha)
    assert list(map(float, model.mat_texrepeat[floor_mat_id])) == pytest.approx(orig_texrepeat)
    assert model.mat_reflectance[floor_mat_id] == pytest.approx(orig_reflectance)
    assert model.mat_shininess[floor_mat_id] == pytest.approx(orig_shininess)
    assert model.mat_emission[floor_mat_id] == pytest.approx(orig_emission)


def test_apply_settings_floor_texrepeat_x_alone_moves_x_but_keeps_y():
    """The texrepeat_x/texrepeat_y pair mirrors the lights dir_az/dir_el decision: they back
    one 2-vector model field, so mentioning only one axis must not silently drop it (leaving the
    whole pair untouched) or silently reset the other axis."""
    model = mujoco.MjModel.from_xml_string(_FLOOR_XML)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_mat_id = int(model.geom_matid[floor_gid])
    orig_y = float(model.mat_texrepeat[floor_mat_id, 1])

    apply_settings(model, {"floor": {"texrepeat_x": 9.0}})

    assert model.mat_texrepeat[floor_mat_id, 0] == pytest.approx(9.0)      # mentioned
    assert model.mat_texrepeat[floor_mat_id, 1] == pytest.approx(orig_y)   # unmentioned


def test_apply_settings_floor_reflectance_alone_does_not_touch_geom_rgba():
    """A settings dict that mentions no colour key at all must leave geom_rgba/mat_rgba
    completely untouched -- not re-derive rgb from the model and rewrite it with the SAME
    values, which would be an inert no-op most of the time but is still the wrong contract to
    guarantee (and would break the moment a caller relies on geom_rgba's identity/aliasing)."""
    model = mujoco.MjModel.from_xml_string(_FLOOR_XML)
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    floor_mat_id = int(model.geom_matid[floor_gid])
    orig_rgba = model.geom_rgba[floor_gid].copy()

    apply_settings(model, {"floor": {"reflectance": 0.9}})

    assert model.mat_reflectance[floor_mat_id] == pytest.approx(0.9)
    assert list(model.geom_rgba[floor_gid]) == pytest.approx(list(orig_rgba))


# -- skybox -----------------------------------------------------------------------------------
#
# Odd width and odd (height // 6) so the pure-top/pure-bottom texel lands exactly at each
# face's own center (u=v=0) -- see visualizer._read_sky_colors's docstring. This makes the
# reconstruction exact up to the uint8 quantization _make_sky_pixels already introduces, rather
# than the extra few percent of interpolation error an even-sized face would add on top.

_SKYBOX_XML = """
<mujoco>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="1 0 0" rgb2="0 0 1" width="9" height="54"/>
  </asset>
  <worldbody><geom type="box" size=".1 .1 .1"/></worldbody>
</mujoco>
"""


def _sample_sky_top_bot(model, tex_id):
    """Independent pixel-sampling check, written separately from
    ``visualizer._read_sky_colors`` so this test does not simply assert the production helper
    agrees with itself."""
    total_h = int(model.tex_height[tex_id])
    w = int(model.tex_width[tex_id])
    face_h = total_h // 6
    nchan = int(model.tex_nchannel[tex_id])
    adr = int(model.tex_adr[tex_id])
    n_pixels = total_h * w
    flat = np.asarray(model.tex_data[adr:adr + n_pixels * nchan]).reshape(n_pixels, nchan)
    center = (face_h // 2) * w + (w // 2)
    top = flat[2 * face_h * w + center, :3] / 255.0
    bot = flat[3 * face_h * w + center, :3] / 255.0
    return top, bot


def test_apply_settings_partial_skybox_dict_updates_only_the_mentioned_field():
    model = mujoco.MjModel.from_xml_string(_SKYBOX_XML)
    tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "skybox")

    apply_settings(model, {"skybox": {"sky_top": "#336699", "sky_bot": "#0d0d26"}})
    _, bot_before = _sample_sky_top_bot(model, tex_id)

    apply_settings(model, {"skybox": {"sky_top": "#ff8800"}})  # only sky_top mentioned
    top_after, bot_after = _sample_sky_top_bot(model, tex_id)

    assert list(top_after) == pytest.approx(_hex_to_rgb("#ff8800"), abs=2 / 255)  # mentioned
    assert list(bot_after) == pytest.approx(list(bot_before), abs=2 / 255)        # unmentioned


def test_apply_settings_skybox_with_neither_color_key_does_not_crash_or_touch_pixels():
    """A settings dict that mentions the skybox group but neither colour key (e.g. only
    'show', which this function does not otherwise handle) must not crash and must not
    regenerate the texture."""
    model = mujoco.MjModel.from_xml_string(_SKYBOX_XML)
    tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "skybox")
    apply_settings(model, {"skybox": {"sky_top": "#336699", "sky_bot": "#0d0d26"}})
    before = model.tex_data.copy()

    apply_settings(model, {"skybox": {"show": True}})  # must not raise

    assert list(model.tex_data) == list(before)


# -- visualizer._read_sky_colors, in isolation -------------------------------------------------


def test_read_sky_colors_reconstructs_known_top_and_bottom():
    from mujoco_visualizer.visualizer import _make_sky_pixels, _read_sky_colors

    model = mujoco.MjModel.from_xml_string(_SKYBOX_XML)
    tex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "skybox")
    top, bot = [0.2, 0.6, 0.9], [0.05, 0.05, 0.15]
    pixels = _make_sky_pixels(model, tex_id, top, bot)
    model.tex_data[:] = pixels.flatten()

    got_top, got_bot = _read_sky_colors(model, tex_id)
    assert got_top == pytest.approx(top, abs=1 / 255)
    assert got_bot == pytest.approx(bot, abs=1 / 255)


def test_read_sky_colors_returns_none_when_there_is_no_skybox_texture():
    from mujoco_visualizer.visualizer import _read_sky_colors

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
    )
    assert _read_sky_colors(model, -1) is None
