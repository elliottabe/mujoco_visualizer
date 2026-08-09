"""render_settings.py — Standalone utilities to load and apply visualizer settings.

Load a settings JSON and apply it to any MuJoCo model + renderer without
needing the full FlyVisualizer class.  Useful in notebooks, render_ghost(),
and batch rendering pipelines.

Example::

    from visualizer.render_settings import load_settings, apply_settings

    settings = load_settings('Earthy_V1')
    apply_settings(mj_model, settings)
    scene_option = build_scene_option(settings)

    # In a render loop:
    renderer.update_scene(data, camera='track1', scene_option=scene_option)
    apply_scene_flags(renderer.scene, settings)
    frame = renderer.render()
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np

from mujoco_visualizer.visualizer import (
    _az_el_to_dir,
    _hex_to_rgb,
    _make_sky_pixels,
    _rgb_to_hex,
    _FREE_TYPE_MAP,
    _build_pan_camera,
    _cosine_ease,
    _resolve_preset,
    dual_lighting,
    scale_lights,
)
from mujoco_visualizer.config import AnatomyConfig
from mujoco_visualizer.categories import build_geom_categories, _auto_anatomy

# ---------------------------------------------------------------------------
# Settings directory
# ---------------------------------------------------------------------------
_SETTINGS_DIR = Path(__file__).parent / 'settings'

DEFAULT_SETTINGS = 'Default'

# Whitelist for a settings preset NAME reaching this package from network-facing code
# (mujoco_visualizer.serve). Deliberately excludes '/', '.', and everything else a path
# needs -- see serve/protocol.py's 'settings' branch for why this has to be enforced before
# the value ever reaches open()/json.load(), not after.
PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _resolve_settings_path(name_or_path: str) -> Path:
    """Resolve a settings name (e.g. 'Earthy_V1') or path to a file."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    # Try as a name in the settings directory
    candidate = _SETTINGS_DIR / f'{name_or_path}.json'
    if candidate.is_file():
        return candidate
    # Try with .json extension already included
    candidate = _SETTINGS_DIR / name_or_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"Settings not found: '{name_or_path}'. "
        f"Available: {[d['name'] for d in list_available_settings()]}"
    )


def list_available_settings(
    user_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, str]]:
    """Return available settings presets as ``{"name", "origin"}`` entries.

    ``origin`` is ``"bundled"`` for presets shipped inside the installed package
    (this file's own ``settings/`` directory) and ``"user"`` for presets found in
    *user_dir*, when one is given and exists.

    A name present in BOTH is listed twice, once per origin -- this function never
    lets one silently shadow the other. Callers that need a single preset for a
    given name (loading) decide precedence themselves; see
    ``mujoco_visualizer.serve.session.Session.load_settings`` for the policy this
    package's own server uses (the user preset wins).

    Bundled presets are always included, even when *user_dir* is None -- this keeps
    the long-standing zero-argument call (predating per-session user directories)
    returning exactly the set of presets it always did.
    """
    result = [
        {"name": p.stem, "origin": "bundled"}
        for p in sorted(_SETTINGS_DIR.glob('*.json'))
    ]
    if user_dir is not None:
        user_dir = Path(user_dir)
        if user_dir.is_dir():
            result.extend(
                {"name": p.stem, "origin": "user"}
                for p in sorted(user_dir.glob('*.json'))
            )
    return result


def load_settings(name_or_path: str) -> dict:
    """Load a settings JSON by preset name or file path.

    Args:
        name_or_path: Either a preset name (e.g. 'Earthy_V1', 'Purple')
                      or a full/relative path to a .json file.

    Returns:
        Settings dictionary with keys: colors, geom_colors, alpha,
        vis_flags, geom_groups, site_groups, camera, lighting, floor,
        skybox, geom_render_state, camera_presets.
    """
    path = _resolve_settings_path(name_or_path)
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Apply settings to a MuJoCo model
# ---------------------------------------------------------------------------

def apply_settings(
    model: mujoco.MjModel,
    settings: dict,
    *,
    anatomy: Optional[AnatomyConfig] = None,
    apply_colors: bool = True,
    apply_lighting: bool = True,
    apply_floor: bool = True,
    apply_skybox: bool = True,
    body_name_substring: Optional[str] = None,
) -> dict:
    """Apply a settings dict to a MuJoCo model (mutates model in place).

    This handles geom colors, lighting, floor material, and skybox texture.
    Returns internal state needed for correct color reset (keep if you plan
    to call apply_settings again with different settings on the same model).

    Args:
        model: MuJoCo model to modify.
        settings: Settings dict (from load_settings or a JSON).
        apply_colors: Whether to apply geom/body colors.
        apply_lighting: Whether to apply lighting settings.
        apply_floor: Whether to apply floor material settings.
        apply_skybox: Whether to apply skybox gradient.
        body_name_substring: Optional substring filter. When set, only geoms
            whose parent body name contains this substring are recolored
            (and their materials baked). Intended for multi-instance scenes
            built with ``MjSpec.attach_body(..., suffix='_flyN')`` — pass
            e.g. ``body_name_substring='_fly1'`` to color one instance
            without touching the other. Scene-wide settings (lighting,
            floor, skybox, *_inertial hide) are unaffected.

    Returns:
        Dict with cached internal state ('geom_categories', 'orig_geom_rgba',
        'floor_geom_id', 'floor_mat_id', 'skybox_tex_id') for reuse.
    """
    # Build internal state
    if anatomy is None:
        anatomy = _auto_anatomy(model)
    geom_categories = build_geom_categories(model, anatomy)

    # Optional: scope category membership to geoms under a given body subtree,
    # so we can color e.g. fly0 and fly1 independently in a multi-fly scene.
    if body_name_substring is not None:
        scoped: Dict[str, List[int]] = {}
        for cat, idxs in geom_categories.items():
            kept: List[int] = []
            for gid in idxs:
                bid = int(model.geom_bodyid[gid])
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ''
                if body_name_substring in bname:
                    kept.append(gid)
            scoped[cat] = kept
        geom_categories = scoped

    orig_geom_rgba = model.geom_rgba.copy()

    # Bake material rgba into geom_rgba for categorized geoms.
    # Match Visualizer.__init__ semantics: if geom has default rgba use mat
    # directly, otherwise modulate. This keeps `apply_settings` and the GUI
    # visualizer pixel-identical.
    orig_mat_rgba = model.mat_rgba.copy()
    cat_geom_ids = {i for idxs in geom_categories.values() for i in idxs}
    _DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1.0])
    for gi in cat_geom_ids:
        mid = int(model.geom_matid[gi])
        if mid >= 0:
            if np.allclose(orig_geom_rgba[gi], _DEFAULT_GEOM_RGBA):
                model.geom_rgba[gi] = orig_mat_rgba[mid]
            else:
                model.geom_rgba[gi] = np.clip(
                    orig_mat_rgba[mid] * orig_geom_rgba[gi], 0.0, 1.0
                )
            model.geom_matid[gi] = -1
    # Refresh originals after baking (so color application below sees baked)
    orig_geom_rgba = model.geom_rgba.copy()

    # Hide *_inertial helper geoms (e.g. wing_left_inertial bounding box).
    # Use a substring check so we still catch names that were renamed by
    # MjSpec.attach_body(..., suffix=...), e.g. wing_left_inertial_fly1.
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if '_inertial' in gname:
            model.geom_rgba[gid, 3] = 0.0
            orig_geom_rgba[gid, 3] = 0.0
    orig_geom_rgba = model.geom_rgba.copy()

    # NOTE: we intentionally do NOT apply settings['geom_render_state'] here.
    # That dict is a raw geom-id -> rgba cache baked against whatever model the
    # preset was saved from; applying it to any other model (e.g. the same fly
    # composed with a different floor.xml) collides on gid and overwrites
    # unrelated geoms like the floor. The name-based 'colors'/'geom_colors'
    # loop below is topology-independent and is the correct path.

    # Detect floor and skybox
    floor_geom_id = next(
        (i for i in range(model.ngeom)
         if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE),
        None
    )
    floor_mat_id = (
        int(model.geom_matid[floor_geom_id])
        if floor_geom_id is not None
           and int(model.geom_matid[floor_geom_id]) >= 0
        else None
    )
    skybox_tex_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_TEXTURE, 'skybox'
    )

    # Apply colors
    if apply_colors:
        colors = settings.get('colors', {})
        alpha = settings.get('alpha', 1.0)
        geom_overrides = {int(k): v for k, v in settings.get('geom_colors', {}).items()}
        for cat, idxs in geom_categories.items():
            cat_rgb = _hex_to_rgb(colors.get(cat, '#888888'))
            for i in idxs:
                if orig_geom_rgba[i, 3] < 0.01:
                    continue
                rgb = _hex_to_rgb(geom_overrides[i]) if i in geom_overrides else cat_rgb
                model.geom_rgba[i, :3] = rgb
                model.geom_rgba[i, 3] = orig_geom_rgba[i, 3] * alpha

    # Apply lighting
    if apply_lighting and 'lighting' in settings:
        lighting = settings['lighting']
        for i, ld in enumerate(lighting.get('lights', [])):
            if i >= model.nlight:
                break
            model.light_active[i] = int(ld['active'])
            model.light_ambient[i] = ld['ambient']
            model.light_diffuse[i] = ld['diffuse']
            model.light_specular[i] = ld['specular']
            model.light_dir[i] = _az_el_to_dir(ld['dir_az'], ld['dir_el'])
        hl = lighting.get('headlight')
        if hl:
            model.vis.headlight.active = int(hl['active'])
            model.vis.headlight.ambient[:] = hl['ambient']
            model.vis.headlight.diffuse[:] = hl['diffuse']
            model.vis.headlight.specular[:] = hl['specular']

    # Apply floor
    if apply_floor and floor_geom_id is not None and 'floor' in settings:
        fld = settings['floor']
        rgb = _hex_to_rgb(fld['color'])
        model.geom_rgba[floor_geom_id] = [*rgb, fld['alpha']]
        if floor_mat_id is not None:
            model.mat_rgba[floor_mat_id] = [*rgb, fld['alpha']]
            model.mat_texrepeat[floor_mat_id] = [fld['texrepeat_x'], fld['texrepeat_y']]
            model.mat_reflectance[floor_mat_id] = fld['reflectance']
            model.mat_shininess[floor_mat_id] = fld['shininess']
            model.mat_emission[floor_mat_id] = fld['emission']

    # Apply skybox
    if apply_skybox and skybox_tex_id >= 0 and 'skybox' in settings:
        sky = settings['skybox']
        pixels = _make_sky_pixels(
            model, skybox_tex_id,
            _hex_to_rgb(sky['sky_top']), _hex_to_rgb(sky['sky_bot'])
        )
        if pixels is not None:
            h = int(model.tex_height[skybox_tex_id])
            w = int(model.tex_width[skybox_tex_id])
            adr = int(model.tex_adr[skybox_tex_id])
            nchan = int(model.tex_nchannel[skybox_tex_id]) if hasattr(
                model, 'tex_nchannel') else 3
            if nchan == 4:
                rgba = np.ones((len(pixels), 4), dtype=np.uint8) * 255
                rgba[:, :3] = pixels
                flat = rgba.flatten()
            else:
                flat = pixels.flatten()
            tex_buf = getattr(model, 'tex_data', None)
            if tex_buf is None:
                tex_buf = getattr(model, 'tex_rgb', None)
            if tex_buf is not None:
                tex_buf[adr:adr + len(flat)] = flat

    return {
        'geom_categories': geom_categories,
        'orig_geom_rgba': orig_geom_rgba,
        'floor_geom_id': floor_geom_id,
        'floor_mat_id': floor_mat_id,
        'skybox_tex_id': skybox_tex_id,
    }


# ---------------------------------------------------------------------------
# Build MjvOption from settings
# ---------------------------------------------------------------------------

def build_scene_option(settings: dict) -> mujoco.MjvOption:
    """Build an MjvOption from a settings dict.

    Applies vis_flags, geom_groups, and site_groups. Use the returned option
    in renderer.update_scene(..., scene_option=opt).

    Args:
        settings: Settings dict (from load_settings).

    Returns:
        Configured MjvOption.
    """
    opt = mujoco.MjvOption()
    f = settings.get('vis_flags', {})
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = f.get('contact_points', False)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = f.get('contact_forces', False)
    opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = f.get('actuators', False)
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = f.get('joints', False)
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = f.get('transparent', False)
    # Default True to match mjVIS_TENDON's own MjvOption() default, and to stay consistent
    # with Visualizer._build_scene_option -- the two read the same vis_flags dicts (settings
    # JSON files, vis_state) and a flag honoured by one and ignored by the other is how a
    # settings preset ends up meaning different things on different render paths.
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = f.get('tendon', True)

    gg = settings.get('geom_groups', [True] * 6)
    sg = settings.get('site_groups', [False] * 6)
    for k in range(min(6, len(gg))):
        opt.geomgroup[k] = int(gg[k])
    for k in range(min(6, len(sg))):
        opt.sitegroup[k] = int(sg[k])

    return opt


def apply_scene_flags(scene: mujoco.MjvScene, settings: dict) -> None:
    """Apply post-render scene flags (shadows, wireframe, skybox).

    Call this AFTER renderer.update_scene() but BEFORE renderer.render().

    Args:
        scene: The MjvScene from renderer.scene.
        settings: Settings dict (from load_settings).
    """
    f = settings.get('vis_flags', {})
    if not f.get('shadows', True):
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    if f.get('wireframe', False):
        scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = True
    if not settings.get('skybox', {}).get('show', True):
        scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False


def get_scene_modifiers(settings: dict) -> List[Tuple]:
    """Get scene modifier functions from settings (dual lighting, etc.).

    Returns list of (function, kwargs) tuples to call after update_scene::

        for fn, kw in get_scene_modifiers(settings):
            fn(renderer.scene, geom_xpos=data.geom_xpos, **kw)
    """
    mods = []
    lighting = settings.get('lighting', {})
    if lighting.get('use_dual_lighting', False):
        mods.append((dual_lighting, {}))
    if lighting.get('use_scale_lights', False):
        mods.append((scale_lights, {'scale': lighting.get('scale_lights_factor', 1.25)}))
    return mods


def build_camera(
    model: mujoco.MjModel,
    settings: dict,
    override: Optional[Union[str, mujoco.MjvCamera]] = None,
) -> Union[str, mujoco.MjvCamera]:
    """Build a camera from settings, or return override if provided.

    Args:
        model: MuJoCo model (needed for trackbody/fixedcam ID lookup).
        settings: Settings dict (from load_settings).
        override: If a string camera name or MjvCamera, returned directly.

    Returns:
        Either a camera name string or a configured MjvCamera.
    """
    if override is not None:
        return override
    c = settings.get('camera', {})
    if c.get('mode', 'named') == 'named':
        return c.get('named', 'track1')

    free_type = c.get('free_type', 'free')
    mj_type, needs_body, needs_fixedcam = _FREE_TYPE_MAP.get(
        free_type, _FREE_TYPE_MAP['free']
    )
    cam = mujoco.MjvCamera()
    cam.type = mj_type
    cam.azimuth = c.get('azimuth', 180.0)
    cam.elevation = c.get('elevation', -20.0)
    cam.distance = c.get('distance', 0.5)
    cam.lookat[:] = c.get('lookat', [0.0, 0.0, 0.0])

    if needs_fixedcam:
        cam_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, c.get('fixedcamid', '')
        )
        cam.fixedcamid = max(cam_id, 0)
    if needs_body:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, c.get('trackbody', '')
        )
        cam.trackbodyid = max(body_id, 0)

    return cam


# ---------------------------------------------------------------------------
# Camera pan
# ---------------------------------------------------------------------------

def make_pan_cameras(
    model: mujoco.MjModel,
    settings: dict,
    preset_names: Sequence[str],
    total_frames: int = 120,
    segment_weights: Optional[Sequence[float]] = None,
    loop: bool = False,
) -> List[mujoco.MjvCamera]:
    """Build a list of MjvCamera objects for a smooth camera pan.

    Standalone equivalent of ``Visualizer.make_pan_cameras`` — works directly
    from a settings dict without instantiating a Visualizer.

    Args:
        model: MuJoCo model (needed for trackbody/fixedcam ID lookup).
        settings: Settings dict containing a ``camera_presets`` mapping.
        preset_names: Ordered list of preset names (>= 2).
        total_frames: Total number of cameras to generate.
        segment_weights: Optional relative time per segment.
        loop: If True, append a segment back to the first preset.

    Returns:
        List of ``mujoco.MjvCamera`` of length ``total_frames``.
    """
    presets = settings.get('camera_presets', {})
    if len(preset_names) < 2:
        raise ValueError("Need at least two preset names to interpolate between.")

    keyframes = []
    for name in preset_names:
        if name not in presets:
            raise KeyError(
                f"Preset '{name}' not found. Available: {list(presets.keys())}"
            )
        keyframes.append(_resolve_preset(presets[name]))

    if loop:
        keyframes.append(keyframes[0])

    n_segs = len(keyframes) - 1
    if segment_weights is None:
        weights = [1.0] * n_segs
    else:
        if len(segment_weights) != n_segs:
            raise ValueError(
                f"segment_weights has {len(segment_weights)} entries but there are "
                f"{n_segs} segments."
            )
        weights = [float(w) for w in segment_weights]

    total_w = sum(weights)
    seg_frames = [max(1, round(w / total_w * total_frames)) for w in weights]
    seg_frames[-1] = max(1, total_frames - sum(seg_frames[:-1]))

    cameras: List[mujoco.MjvCamera] = []
    for seg in range(n_segs):
        A = keyframes[seg]
        B = keyframes[seg + 1]
        n = seg_frames[seg]
        for fi in range(n):
            t = _cosine_ease(fi / n)
            cameras.append(_build_pan_camera(model, A, B, t))
    return cameras


# ---------------------------------------------------------------------------
# Convenience: one-call render setup
# ---------------------------------------------------------------------------

def setup_render(
    model: mujoco.MjModel,
    settings_name: str = DEFAULT_SETTINGS,
    camera: Optional[str] = None,
    anatomy: Optional[AnatomyConfig] = None,
) -> Tuple[mujoco.MjvOption, Union[str, mujoco.MjvCamera], dict]:
    """Load settings, apply to model, and return everything needed to render.

    Args:
        model: MuJoCo model to configure.
        settings_name: Preset name or path to settings JSON.
        camera: Optional camera override (name string).

    Returns:
        (scene_option, camera, settings) tuple. The model is mutated in place.

    Example::

        scene_option, camera, settings = setup_render(mj_model, 'Earthy_V1')
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        apply_scene_flags(renderer.scene, settings)
        frame = renderer.render()
    """
    settings = load_settings(settings_name)
    apply_settings(model, settings, anatomy=anatomy)
    scene_option = build_scene_option(settings)
    cam = build_camera(model, settings, override=camera)
    return scene_option, cam, settings
