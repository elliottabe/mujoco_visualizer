"""visualizer.py — Generic offscreen Visualizer for any MuJoCo model.

Wraps ``mujoco.Renderer`` with visual state (per-category geom colors,
lighting, floor, skybox, named cameras, presets) so callers can render
single frames, videos, and smooth multi-keyframe camera pans without any
notebook environment.

Quick-start::

    from mujoco_visualizer import Visualizer, load_config

    anatomy = load_config('humanoid.yaml')          # or None for auto
    viz = Visualizer('humanoid.xml', anatomy=anatomy)
    frame = viz.render_frame(viz.model.qpos0, camera='side')
    viz.render_video(qposes, camera='side', output_path='out.mp4')
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Union

import mujoco
import numpy as np

from mujoco_visualizer.config import AnatomyConfig, load_config
from mujoco_visualizer.categories import build_geom_categories, _auto_anatomy

# Camera type mapping: free_type str → (mjtCamera, needs_trackbody, needs_fixedcam)
_FREE_TYPE_MAP = {
    'free':     (mujoco.mjtCamera.mjCAMERA_FREE,     False, False),
    'fixed':    (mujoco.mjtCamera.mjCAMERA_FIXED,    False, True),
    'track':    (mujoco.mjtCamera.mjCAMERA_TRACKING, True,  False),
    'trackcom': (mujoco.mjtCamera.mjCAMERA_TRACKING, True,  False),
}

# ---------------------------------------------------------------------------
# Module-level pure helper functions
# ---------------------------------------------------------------------------

def _dir_to_az_el(d: Sequence[float]) -> Tuple[float, float]:
    d = np.asarray(d, dtype=float)
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        return 0.0, -45.0
    d = d / norm
    el = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(d[0], d[1])) % 360)
    return az, el


def _az_el_to_dir(az_deg: float, el_deg: float) -> np.ndarray:
    """Unit direction for a LIGHT at *az_deg*/*el_deg*. The inverse of :func:`_dir_to_az_el`.

    DO NOT DEDUPLICATE THIS WITH ``serve.session.camera_basis``. The two look like the same
    function and are not: this one is ``[cos(el)sin(az), cos(el)cos(az), sin(el)]``, azimuth
    measured from world +y, which pairs with ``_dir_to_az_el``'s ``arctan2(d[0], d[1])`` and is
    the convention ``vis_state['lighting']``'s stored az/el round-trip through. It is
    LIGHT-ONLY: nothing camera-shaped reads it.

    ``camera_basis`` is ``[cos(el)cos(az), cos(el)sin(az), sin(el)]`` -- MuJoCo's CAMERA
    convention, azimuth from world +x, which is what ``MjvCamera.azimuth`` means and therefore
    what a screen-space pan must resolve in. It is 90 degrees rotated from this with the
    opposite handedness (sin/cos swapped). Collapsing them into one helper would silently rotate
    either every light or every pan basis by 90 degrees, and nothing would fail loudly.
    """
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    return np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])


def _hex_to_rgb(hex_str: str) -> List[float]:
    h = hex_str.lstrip('#')
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return '#{:02x}{:02x}{:02x}'.format(
        *[int(np.clip(v, 0, 1) * 255) for v in rgb]
    )


def _apply_forces_vis(forces: dict, model: mujoco.MjModel) -> None:
    """Write whichever of the five force/torque arrow-scaling fields *forces* mentions onto
    *model.vis*, leaving any field it does not mention at whatever the model already holds.

    These fields (``map.force``/``map.torque``/``scale.forcewidth``/``scale.contactwidth``/
    ``scale.contactheight``) control ``mjVIS_CONTACTFORCE`` arrow length and width, but unlike
    every other ``vis_state`` group they live on ``MjModel.vis``, not on the per-render
    ``MjvOption`` -- so there is no scene-option flag to flip, and no reason to expect a fresh
    render to pick these up on its own after the model object itself changes.

    Partial-dict tolerance is deliberate, not an oversight: :func:`render_settings.apply_settings`
    hands this function a raw settings dict from a caller who may reasonably type
    ``{'forces': {'map_force': 0.07}}`` -- the one field they are tuning, not all five. Requiring
    every key crashed that call with a bare ``KeyError``. The fix is NOT to fill the other four
    from MuJoCo's library defaults before calling this -- that would silently overwrite whatever
    the MJCF set for every field the caller did not mention, which is exactly the hardcode
    hazard requirement 1 (initialise ``vis_state`` from the model, never a constant) exists to
    prevent, one call later. ``Visualizer._apply_forces`` and
    ``session._carry_vis_state_across_swap`` both always pass a complete dict (``vis_state``
    holds all five keys from ``__init__`` onward), so this is unobservable from either of them --
    only :func:`apply_settings`'s raw, caller-supplied dict can be partial.

    That is exactly why this is a free function rather than only a ``Visualizer`` method:
    :meth:`Visualizer._apply_forces` calls it against ``self.model`` on every render,
    ``session._carry_vis_state_across_swap`` calls it again directly against the freshly
    swapped-in model, and ``render_settings.apply_settings`` calls it a third time against a
    caller-supplied settings dict that need not be complete.
    """
    if 'map_force' in forces:
        model.vis.map.force = forces['map_force']
    if 'map_torque' in forces:
        model.vis.map.torque = forces['map_torque']
    if 'scale_forcewidth' in forces:
        model.vis.scale.forcewidth = forces['scale_forcewidth']
    if 'scale_contactwidth' in forces:
        model.vis.scale.contactwidth = forces['scale_contactwidth']
    if 'scale_contactheight' in forces:
        model.vis.scale.contactheight = forces['scale_contactheight']


def _make_sky_pixels(
    model: mujoco.MjModel,
    skybox_tex_id: int,
    top_rgb: Sequence[float],
    bot_rgb: Sequence[float],
) -> Optional[np.ndarray]:
    """Build cube-map gradient pixel data for the skybox texture."""
    if skybox_tex_id < 0:
        return None
    total_h = int(model.tex_height[skybox_tex_id])
    w = int(model.tex_width[skybox_tex_id])
    face_h = max(1, total_h // 6)
    top = np.array(top_rgb, dtype=np.float64)
    bot = np.array(bot_rgb, dtype=np.float64)
    face_axes = [
        (np.array([1., 0., 0.]),  np.array([0., 0., -1.]), np.array([0., -1., 0.])),
        (np.array([-1., 0., 0.]), np.array([0., 0., 1.]),  np.array([0., -1., 0.])),
        (np.array([0., 1., 0.]),  np.array([1., 0., 0.]),  np.array([0., 0., 1.])),
        (np.array([0., -1., 0.]), np.array([1., 0., 0.]),  np.array([0., 0., -1.])),
        (np.array([0., 0., 1.]),  np.array([1., 0., 0.]),  np.array([0., -1., 0.])),
        (np.array([0., 0., -1.]), np.array([-1., 0., 0.]), np.array([0., -1., 0.])),
    ]
    pixels = np.zeros((total_h * w, 3), dtype=np.uint8)
    rows = np.arange(face_h)
    cols = np.arange(w)
    v_arr = 1.0 - 2.0 * (rows + 0.5) / face_h
    u_arr = -1.0 + 2.0 * (cols + 0.5) / w
    V, U = np.meshgrid(v_arr, u_arr, indexing='ij')
    for fi, (norm, right, up) in enumerate(face_axes):
        d = norm + U[:, :, None] * right + V[:, :, None] * up
        d /= np.linalg.norm(d, axis=2, keepdims=True)
        t = np.clip(0.5 + 0.5 * d[:, :, 1], 0.0, 1.0)
        color = (1.0 - t[:, :, None]) * bot + t[:, :, None] * top
        pixels[fi * face_h * w:(fi + 1) * face_h * w] = (
            np.clip(color * 255, 0, 255).astype(np.uint8).reshape(-1, 3)
        )
    return pixels


# Scene modifier functions (applied after update_scene)
def dual_lighting(scene: mujoco.MjvScene, geom_xpos: Optional[np.ndarray] = None,
                  **kwargs) -> None:
    if geom_xpos is None:
        return
    body_pos = geom_xpos[1]
    if scene.nlight > 2:
        scene.lights[0].pos[:] = body_pos + np.array([0.6, 0.6, 0.0])
        scene.lights[0].dir[:] = body_pos - scene.lights[2].pos
    scene.lights[0].diffuse[:] = [1.4, 1.3, 1.0]
    scene.lights[0].specular[:] = [1.4, 1.4, 1.4]
    scene.lights[0].cutoff = 20.0
    scene.lights[0].exponent = 3.0
    scene.lights[0].attenuation[:] = [1, 0.0, 0.0]


def add_arrow_to_scene(
    scene: mujoco.MjvScene,
    from_: Sequence[float],
    to: Sequence[float],
    radius: float = 0.003,
    rgba: Sequence[float] = (0.2, 0.2, 0.6, 1.0),
) -> None:
    """Append an arrow geom to ``scene``."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    g.category = mujoco.mjtCatBit.mjCAT_STATIC
    mujoco.mjv_initGeom(
        geom=g,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom=g,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=radius,
        from_=np.asarray(from_, dtype=float),
        to=np.asarray(to, dtype=float),
    )
    scene.ngeom += 1


def build_actuator_tendon_map(
    model: mujoco.MjModel,
    actuator_color_fn: Optional[Callable] = None,
    driven_ids: Optional[Set[int]] = None,
) -> Tuple[Dict[int, int], np.ndarray]:
    """``{actuator id: tendon id}`` for every actuator whose transmission is a tendon, plus a
    ``(model.nu, 4)`` base RGBA array carrying each such actuator's colour.

    Pure and read-only -- never mutates *model*. Built from ``actuator_trntype ==
    mjTRN_TENDON`` and ``actuator_trnid[i, 0]``, i.e. from whichever model is passed in, never
    by position against some other model -- so a caller that keeps this map cached across a
    reference-ghost swap (``ntendon``/``nu`` roughly doubling) MUST call this again on the new
    model rather than reusing the old map's ids against it.

    *actuator_color_fn*, when given, is called as ``(name: str) -> color`` where *color* is a
    hex string (e.g. ``'#d84a2e'``) or an RGBA 4-tuple; an actuator it does not cover, or no
    function at all, falls back to solid red -- the same fallback
    :meth:`Visualizer.render_video_pan` already used before this map-building loop was
    extracted out of it.

    *driven_ids*, when given, is the set of actuator ids on *model* that some primary ctrl
    column actually drives (i.e. the non-negative entries of a :func:`build_ctrl_name_map`
    result). An actuator absent from it is OMITTED from the returned map, because nothing ever
    writes its activation: it is structurally zero for the life of the caller, so its tendon
    carries no signal at all and drawing it palette-coloured states something the data does
    not. Since :func:`apply_tendon_activation` already hides every tendon absent from
    ``act_to_ten.values()``, omitting here IS hiding -- no new code path. On the fly
    reference-ghost pair that is ~260 dim duplicate tendons drawn directly over the muscles they
    mimic; on a single-model session every primary name matches and nothing is dropped.

    That rule lives HERE, in the shared builder, rather than as a post-filter at a call site,
    because it has more than one caller and the two must not diverge:
    ``Session._rebuild_tendon_colors`` (serve/session.py) drives the live preview and
    ``ExportJob`` (serve/export.py) renders video from its own ``Visualizer`` on its own thread.
    While the filter lived on the Session only, exporting with the reference ghost active wrote
    coloured duplicate tendons into the video that the preview never showed. Same reasoning as
    :func:`build_ctrl_name_map`, which was extracted for those same two callers so the
    name-matching rule could not drift between them. Expressed in terms of ctrl columns rather
    than a name suffix so this package needs no knowledge of what a ghost is.

    When *driven_ids* is ``None`` nothing is filtered -- exactly the pre-existing behaviour,
    which ``render_video_pan`` and other single-model callers holding no ctrl map still want.
    """
    act_to_ten: Dict[int, int] = {}
    base_rgba = np.zeros((model.nu, 4), dtype=np.float32)
    _mjTRN_TENDON = int(mujoco.mjtTrn.mjTRN_TENDON)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name is None:
            continue
        if driven_ids is not None and i not in driven_ids:
            continue
        trntype = int(model.actuator_trntype[i])
        trnid = model.actuator_trnid[i, 0]
        if trntype == _mjTRN_TENDON and 0 <= trnid < model.ntendon:
            act_to_ten[i] = trnid
            if actuator_color_fn is not None:
                clr = actuator_color_fn(name)
                if isinstance(clr, str):
                    clr = _hex_to_rgb(clr) + [1.0]
                base_rgba[i] = clr
            else:
                base_rgba[i] = [0.85, 0.15, 0.15, 1.0]
    return act_to_ten, base_rgba


def actuator_names(model: mujoco.MjModel) -> List[str]:
    """Every actuator name on *model*, in id order.

    A single shared implementation of an idiom that had drifted into three separate copies
    (this one, ``Session._actuator_names`` in serve/session.py, and a private one in
    serve/export.py) -- both serve-layer modules already import from this module, so this is
    where it belongs. An unnamed actuator gets a placeholder rather than ``None``, so it can
    still occupy a slot in :func:`build_ctrl_name_map`'s name lists without ever matching a
    real name.
    """
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
        for i in range(model.nu)
    ]


def build_ctrl_name_map(
    primary_names: Sequence[str],
    active_model: mujoco.MjModel,
) -> np.ndarray:
    """``{index into a primary-ordered ctrl vector -> index into data.ctrl on *active_model*}``,
    built by matching actuator NAMES -- never by position.

    Extracted from ``Session._build_ctrl_map`` (serve/session.py) so a caller that has no
    ``Session`` at all -- ``ExportJob`` (serve/export.py), which builds its own ``Visualizer``
    on its own thread -- can reuse exactly the same matching rule rather than re-deriving (or
    silently drifting from) it. ``Session._build_ctrl_map`` itself now calls this function; it
    keeps no separate implementation.

    The reference-ghost pair doubles the actuator count (``nu`` 272 -> 544 on the real
    models), so a 272-wide replay/recorded ctrl vector cannot be written into a 544-wide
    ``data.ctrl`` positionally: assuming the policy's actuators occupy a fixed prefix of the
    doubled model is exactly the attachment-order assumption that produced a real
    data-corruption bug on this branch (a prefix heuristic silently mis-assigning ghost
    counterparts). ``locks.pair_with_suffix`` already solves the equivalent problem for joint
    names for the same reason; this does it for actuator ids.

    A primary name with no match on *active_model* (there is never one when the active model
    IS the primary model) maps to ``-1`` -- callers must filter ``map >= 0`` before scattering,
    never rely on numpy's negative-index wraparound to skip it (see
    ``test_ctrl_map_skips_an_unmatched_primary_name_rather_than_wrapping_onto_the_last_actuator``).
    """
    active_id_of = {name: i for i, name in enumerate(actuator_names(active_model))}
    return np.array(
        [active_id_of.get(name, -1) for name in primary_names],
        dtype=np.int64,
    )


def apply_tendon_activation(
    model: mujoco.MjModel,
    ctrl: Sequence[float],
    act_to_ten: Dict[int, int],
    base_rgba: np.ndarray,
    *,
    tendon_width: float = 0.003,
    tendon_min_width: float = 0.0005,
    tendon_alpha_min: float = 0.05,
    tendon_baseline: float = 0.0,
    ctrl_max: float = 1.0,
) -> None:
    """Colour and thicken muscle tendons by ``|ctrl|`` activation, for one frame.

    Writes ``model.tendon_rgba`` (alpha) and ``model.tendon_width`` in place -- nothing is
    returned. Every tendon NOT in ``act_to_ten.values()`` is hidden (``tendon_rgba[:, 3] =
    0.0``) on every call, so this function is self-contained and idempotent per frame: a
    caller need not separately hide non-muscle tendons once up front before the first call.

    For each muscle actuator/tendon pair, ``|ctrl[act_id]| / ctrl_max`` is clipped to
    ``[0, 1]``, blended with ``tendon_baseline`` (so e.g. 0.3 keeps low activations visibly
    above ``tendon_min_width``/``tendon_alpha_min`` rather than fading to nothing), and the
    result drives BOTH the tendon's alpha (floored at ``tendon_alpha_min``) and its width
    (interpolated between ``tendon_min_width`` and ``tendon_width``).

    ``base_rgba`` supplies RGB only -- colour says which muscle, alpha and width say how hard.
    Its own alpha channel is ignored and OVERWRITTEN by the activation-derived value above, not
    multiplied into it: the rendered alpha is always exactly ``max(norm, tendon_alpha_min)``,
    regardless of what a colour function returned. This is why a colour function passed to
    :func:`build_actuator_tendon_map` may return either a hex string (which has no alpha at
    all) or an RGBA 4-tuple interchangeably -- a 4-tuple's alpha component is accepted but never
    has any effect on what gets drawn, so a palette cannot silently scale activation brightness
    by choosing a dim or bright alpha.

    ``ctrl_max`` is a single caller-supplied scalar, not computed here: this function has no
    lookahead across frames (see :meth:`Visualizer.render_video_pan`, which computes it once
    from the whole ``ctrls`` clip, and the live-serving path in ``serve/session.py``, which
    computes it once from the model's own actuator ``ctrlrange`` -- see that module's
    docstring for why a per-frame recomputed max is the wrong choice).

    Does NOT snapshot or restore the model's original tendon_rgba/tendon_width -- that is the
    caller's responsibility (both existing callers need different "off" semantics: render_video_
    pan restores once after its whole clip, the live viewer restores every frame tendons are
    disabled).
    """
    ctrl = np.asarray(ctrl, dtype=float)
    muscle_ten_ids = set(act_to_ten.values())
    for t in range(model.ntendon):
        if t not in muscle_ten_ids:
            model.tendon_rgba[t, 3] = 0.0
    width_range = tendon_width - tendon_min_width
    denom = max(float(ctrl_max), 1e-8)
    for act_id, ten_id in act_to_ten.items():
        raw = float(np.clip(abs(ctrl[act_id]) / denom, 0.0, 1.0))
        norm = tendon_baseline + (1.0 - tendon_baseline) * raw
        alpha = max(norm, tendon_alpha_min)
        rgba = base_rgba[act_id].copy()
        rgba[3] = alpha
        model.tendon_rgba[ten_id] = rgba
        model.tendon_width[ten_id] = tendon_min_width + width_range * norm


def default_tendon_ctrl_full_scale(
    model: mujoco.MjModel,
    act_to_ten: Optional[Dict[int, int]] = None,
) -> float:
    """The model-only default for ``apply_tendon_activation``'s ``ctrl_max``: the largest
    ``|ctrlrange|`` bound declared by any tendon-driving, ctrl-limited actuator on *model*, or
    ``1.0`` if none declare one. *act_to_ten*, when already available (e.g. a caller that has
    already called :func:`build_actuator_tendon_map`), is reused instead of rebuilding it.

    THIS IS A THEORETICAL CEILING, NOT A MEASURED ONE, and on the real fly model it is a
    misleading one on its own: ``actuator_ctrlrange`` over the 258 tendon-driving actuators
    (all ctrl-limited) spans -1.05 to 1, so this returns 1.05 -- but a trained policy's actual
    ``|ctrl|`` occupies only a small fraction of that ceiling. Measured on a real rollout
    (clip 65, frames 200-320)::

        |ctrl| p50  = 0.0246   -> normalised against 1.05: 0.023
               p90  = 0.0555   ->                          0.053
               p99  = 0.1442   ->                          0.137
               max  = 0.6287   ->                          0.599

    Normalising against 1.05 therefore renders essentially every tendon near minimum width and
    alpha almost all the time -- a uniformly dim, inert-looking picture that shows almost none
    of the variation it exists to show, the same trap as MuJoCo's native force arrows being
    invisible at their principled-but-wrong-scale default.

    This function/value is only ever meant to be vis_state['tendons']['ctrl_full_scale']'s
    FALLBACK default -- what a viewer with no rollout loaded has nothing better to show. A
    caller that HAS the actual data (the serve-layer launcher, which loads the rollout) should
    override ``vis_state['tendons']['ctrl_full_scale']`` with a measured percentile of
    ``|ctrl|`` across it instead of trusting this ceiling. That override is intentionally not
    done here, or anywhere per-frame: recomputing it from ``data.ctrl`` inside the apply path
    (per-frame, or a running max) would reintroduce exactly the problem a fixed reference value
    exists to avoid -- every frame's brightest muscle would render equally bright regardless of
    how active the animal actually is, making a quiet frame indistinguishable from a loud one.
    """
    if act_to_ten is None:
        act_to_ten, _ = build_actuator_tendon_map(model)
    bounds = [
        max(
            abs(float(model.actuator_ctrlrange[a, 0])),
            abs(float(model.actuator_ctrlrange[a, 1])),
        )
        for a in act_to_ten
        if bool(model.actuator_ctrllimited[a])
    ]
    return max(bounds) if bounds else 1.0


def default_force_arrow_scale(model: mujoco.MjModel) -> float:
    """The model-derived default for ``vis_state['force_arrows']['scale']``:
    ``0.1 * model.stat.extent / (total_mass * |gravity_z|)``.

    THIS IS DELIBERATELY NOT A HARDCODED CONSTANT -- the same hardcode hazard
    ``default_tendon_ctrl_full_scale`` and ``_apply_forces_vis`` both exist to avoid. A fixed
    scale is invisible on any model whose size/mass/gravity combination it was not tuned for:
    MuJoCo's own native force-arrow scaling (``model.vis.map.force``, see the ``forces`` group
    above) renders a weight-sized force at roughly ``3e-5`` of this model's own extent -- a
    fixed default carried over from a different unit system is exactly that kind of
    trap. The formula instead asks "how long should the arrow for ONE BODY-WEIGHT of force be,
    relative to how big this model already is": a force of ``total_mass * |gravity_z|``
    (one weight) times this scale renders at ``0.1 * model.stat.extent`` -- a visible,
    consistent fraction of the model's own size, regardless of what units/scale the MJCF uses.

    Verified against the real fly model (CGS units: ``gravity_z = -981``,
    ``total_mass ≈ 9.8e-4``, ``extent = 1.0``): this returns ``0.10397``, matching the
    measurement in ``scripts/rollout_viewer/force_arrows.default_force_arrow_scale`` (task
    15a's sibling module, which computes the identical formula independently -- see this
    module's own docstring note on that duplication).

    Falls back to ``1.0`` when ``total_mass * |gravity_z|`` is zero (a model with no mass or
    zero-gravity ``opt.gravity``, e.g. a bare test fixture) -- there is no principled "one
    weight" reference distance to calibrate against on such a model, so this is a documented
    placeholder a caller on that model must override, not a claim that 1.0 is somehow correct.
    Mirrors ``default_tendon_ctrl_full_scale``'s own ``else: 1.0`` for the equivalent
    no-signal case.
    """
    total_mass = float(model.body_mass.sum())
    gravity_z = abs(float(model.opt.gravity[2]))
    denom = total_mass * gravity_z
    if denom <= 0.0:
        return 1.0
    return 0.1 * float(model.stat.extent) / denom


def default_marker_radius(model) -> float:
    """The model-derived default for ``vis_state['markers']['radius']``:
    ``0.25 * model.stat.meansize``.

    NOT a hardcoded constant, for the same reason ``default_force_arrow_scale`` and
    ``default_tendon_ctrl_full_scale`` are not: a fixed radius is either invisible or
    occludes the whole model depending on what units the MJCF uses. ``stat.meansize`` is
    MuJoCo's own mean-body-size statistic -- the quantity its default visual scaling already
    keys off -- so a quarter of it is "a quarter of a typical body", which means the same
    thing at any scale.

    Verified on the v1 fly (CGS units, ``meansize = 0.02``): returns ``0.005`` cm, a 0.01 cm
    sphere against a 0.25 cm body -- about 4% of body length, readable against a limb without
    swallowing the joint it marks. That also coincides with
    ``add_trajectory_points_to_scene``'s own ``0.005`` default on this model.

    Falls back to ``0.005`` when ``meansize`` is zero (a degenerate or bare test model),
    matching that helper's default -- a documented placeholder, not a claim it is right.
    """
    meansize = float(getattr(model.stat, "meansize", 0.0) or 0.0)
    return 0.25 * meansize if meansize > 0.0 else 0.005


def get_wing_fluid_idxs(model: mujoco.MjModel, suffix='') -> List[int]:
    """Return geom ids of the wing fluid geoms (left, right) in *model*."""
    out = []
    for name in (f'wing_left_fluid{suffix}', f'wing_right_fluid{suffix}'):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            out.append(gid)
    return out


def add_aero_force_arrows_to_scene(
    scene: mujoco.MjvScene,
    aero_forces: np.ndarray,
    wing_fluid_idxs: Sequence[int],
    geom_xpos: np.ndarray,
    radius: float = 0.003,
    rgba: Sequence[Sequence[float]] = (
        (139 / 255, 107 / 255, 127 / 255, 1.0),
        (149 / 255, 184 / 255, 114 / 255, 1.0),
        (200 / 255,  98 / 255,  77 / 255, 1.0),
    ),
    scale_vectors: float = 0.1,
) -> None:
    """Draw per-wing aerodynamic force arrows for one frame.

    Args:
        aero_forces:     (n_forces, n_wings, 3) array for the current frame.
        wing_fluid_idxs: Geom ids of the wing fluid geoms (one per wing).
        geom_xpos:       ``data.geom_xpos`` for the current frame.
        rgba:            One color per force component (length >= n_forces-1).
                         Mirrors fly_logging.add_arrows which skips the last entry.
    """
    aero_forces = np.asarray(aero_forces)
    for wing_idx, gid in enumerate(wing_fluid_idxs):
        wing_xpos = np.asarray(geom_xpos[gid])
        for m in range(len(aero_forces) - 1):
            f = np.asarray(aero_forces[m, wing_idx])
            add_arrow_to_scene(
                scene,
                from_=wing_xpos,
                to=wing_xpos + scale_vectors * f,
                radius=radius,
                rgba=rgba[m],
            )


def add_trajectory_points_to_scene(
    scene: mujoco.MjvScene,
    points: np.ndarray,
    radius: float = 0.005,
    rgba: Sequence[float] = (0.0, 1.0, 1.0, 0.5),
    skip: int = 1,
) -> None:
    """Append a sphere geom for each trajectory point in *points* (N,3)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    mat = np.eye(3).flatten()
    for i in range(0, len(pts), max(1, skip)):
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        g.category = mujoco.mjtCatBit.mjCAT_STATIC
        mujoco.mjv_initGeom(
            geom=g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, radius, radius], dtype=float),
            pos=pts[i],
            mat=mat,
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scene.ngeom += 1


def scale_lights(scene: mujoco.MjvScene, scale: float = 1.25, **kwargs) -> None:
    for i in range(scene.nlight):
        scene.lights[i].diffuse[:] *= scale
        scene.lights[i].ambient[:] *= scale
        scene.lights[i].specular[:] = 0


# Camera pan helpers
def _lerp_angle(a: float, b: float, t: float) -> float:
    diff = ((b - a + 180.0) % 360.0) - 180.0
    return a + diff * t


def _cosine_ease(t: float) -> float:
    return 0.5 * (1.0 - np.cos(np.pi * t))


def allocate_segment_frames(weights: Sequence[float], total_frames: int) -> List[int]:
    """Split *total_frames* across segments in proportion to *weights*, summing to EXACTLY
    *total_frames* for any weights and any ``total_frames >= 1``.

    This is the whole reason ``make_pan_cameras`` can promise a list of length
    ``total_frames``. The rule it replaces was
    ``[max(1, round(w / total_w * n)) for w in weights]`` with the last segment set to the
    remainder, and the ``max(1, ...)`` floor broke the promise: once the non-last segments
    rounded UP to n or more frames between them, the last segment's remainder went <= 0, was
    floored back to 1, and the list came out LONGER than n. Measured: weights [3, 6, 6, 1] over
    n = 20 produced [4, 8, 8, 1] = 21 cameras, which a preview happily clamped its index into
    while ``ExportJob.__init__`` refused the same path outright -- a shot the user could preview
    and then not export. Reachable for weights in 1-6 over 2-4 segments whenever n is small
    (a short trim, or a large stride over a long clip).

    Largest-remainder (Hamilton) apportionment, which is exact by construction: floor each
    segment's ideal share, then hand the frames that floor discarded to the segments with the
    largest fractional parts. Ties break toward the LOWER index, stated explicitly rather than
    left to sort stability, because `RVSettings.pathSegmentCounts` in the fly viewer's
    ``rollout_settings.js`` mirrors this function to print the split in the path editor and the
    two must agree. Floor-plus-fraction is part of what makes that mirroring sound across
    runtimes: the old rule used ``round``, and Python (banker's, to even) and JS
    (``Math.round``, half away from zero) disagree on exact ``.5`` shares, whereas ``floor`` and
    an IEEE double comparison agree.

    **The weight total is summed by an explicit left-fold, and JAVASCRIPT'S ARITHMETIC IS
    NORMATIVE here.** Floor-plus-fraction is not sufficient on its own: CPython >= 3.12's
    built-in ``sum()`` applies Neumaier compensated summation to floats, while JS ``reduce``
    is a naive left-fold, so the two runtimes disagree on the TOTAL before either divides by
    it. ``[0.1, 1.3, 0.1]`` totals ``1.5`` under ``sum()`` and ``1.5000000000000002`` under
    ``reduce``; the ideal shares then differ in the last bits and the largest-remainder ordering
    of two near-tied fractions flips. Measured on the tenths grid the editor's
    ``min="0.1" step="0.1"`` weight input actually produces: weights ``[0.1, 1.1, 0.3]`` over
    351 frames printed ``24 / 257 / 70`` in the readout while the server rendered
    ``23 / 258 / 70`` -- always a one-frame move between two segments, and never caught by any
    total check because both sides still sum to ``n``. The loop below reproduces JS's naive fold
    exactly rather than the reverse, because a naive accumulator is one line that can be stated
    and kept true in both languages, whereas hand-writing Neumaier in JavaScript would be a
    second delicate implementation free to drift. ``math.fsum`` would be a THIRD answer and is
    deliberately not used.

    **A segment may receive 0 frames, and that is deliberate** -- but only when there are fewer
    frames than segments. With ``total_frames < len(weights)`` some keyframes simply cannot be
    visited: one frame renders one camera, so a 3-segment path over 1 frame shows one shot and
    skips the rest. Skipping a keyframe is strictly better than the alternative the old floor
    chose, which was to return MORE cameras than there are frames -- a list that cannot be
    rendered at all. When ``total_frames >= len(weights)`` every segment is guaranteed at least
    one frame (the repair loop below), so each keyframe is still visited and
    ``test_each_segments_first_frame_is_its_start_keyframe`` holds; a pathological weight ratio
    like [1, 100] over 3 frames yields [1, 2], not [0, 3].
    """
    n_segs = len(weights)
    total_frames = int(total_frames)
    if n_segs == 0:
        return []
    # Naive left-fold, matching `pathSegmentCounts`' `weights.reduce((a, b) => a + b, 0)` bit for
    # bit. NOT `sum(weights)`: see the docstring -- CPython >= 3.12 compensates, JS does not.
    total_w = 0.0
    for w in weights:
        total_w += float(w)
    if not math.isfinite(total_w) or total_w <= 0.0:
        # Otherwise this is a bare ZeroDivisionError from `w / total_w`, which is not a
        # ValueError -- so `SimLoop._publish`'s `except ValueError` misses it and the loop PAUSES
        # with a kind:"render" error. `Session.set_camera_path` rejects each offending weight by
        # index before a path can be armed; this is the same failure made survivable for a direct
        # `make_pan_cameras` caller, which has no such gate in front of it.
        raise ValueError(
            f"segment weights must sum to a finite positive number; got {list(weights)!r}"
        )
    ideal = [w / total_w * total_frames for w in weights]
    counts = [int(math.floor(x)) for x in ideal]
    short = total_frames - sum(counts)
    # `short` is in [0, n_segs): each floor discards less than one frame.
    order = sorted(range(n_segs), key=lambda i: (-(ideal[i] - counts[i]), i))
    for i in order[:short]:
        counts[i] += 1

    # Restore the old rule's "every segment gets at least one frame" intent wherever it is
    # actually affordable, by moving a frame from the longest segment to each starved one.
    # Safe: with total_frames >= n_segs and at least one zero, the non-zero segments share
    # total_frames > (number of non-zero segments) frames, so the longest holds >= 2 and cannot
    # be emptied by giving one away.
    if total_frames >= n_segs:
        for i in range(n_segs):
            if counts[i] == 0:
                donor = max(range(n_segs), key=lambda j: (counts[j], -j))
                counts[donor] -= 1
                counts[i] += 1
    return counts


def _resolve_preset(cam_cfg: dict) -> dict:
    return {
        'azimuth':    float(cam_cfg.get('azimuth',    180.0)),
        'elevation':  float(cam_cfg.get('elevation',  -20.0)),
        'distance':   float(cam_cfg.get('distance',    0.5)),
        'lookat':     [float(v) for v in cam_cfg.get('lookat', [0.0, 0.0, 0.0])],
        'free_type':  cam_cfg.get('free_type',  'free'),
        'trackbody':  cam_cfg.get('trackbody',  ''),
        'fixedcamid': cam_cfg.get('fixedcamid', ''),
    }


def _build_pan_camera(
    model: mujoco.MjModel,
    A: dict,
    B: dict,
    t: float,
) -> mujoco.MjvCamera:
    kf = A if t < 0.5 else B
    free_type = kf['free_type']
    mj_type, needs_body, needs_fixedcam = _FREE_TYPE_MAP.get(free_type, _FREE_TYPE_MAP['free'])

    cam = mujoco.MjvCamera()
    cam.type = mj_type
    cam.azimuth   = _lerp_angle(A['azimuth'],   B['azimuth'],   t)
    cam.elevation = A['elevation'] + (B['elevation'] - A['elevation']) * t
    cam.distance  = A['distance']  + (B['distance']  - A['distance'])  * t
    cam.lookat[:] = [A['lookat'][i] + (B['lookat'][i] - A['lookat'][i]) * t for i in range(3)]

    if needs_fixedcam:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, kf.get('fixedcamid', ''))
        cam.fixedcamid = max(cam_id, 0)
    if needs_body:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kf.get('trackbody', ''))
        cam.trackbodyid = max(body_id, 0)

    return cam


def _save_video(frames: np.ndarray, output_path: str, fps: int) -> None:
    """Write video frames to a file using mediapy or imageio."""
    try:
        import mediapy
        mediapy.write_video(output_path, frames, fps=fps)
    except ImportError:
        try:
            import imageio
            imageio.mimwrite(output_path, frames, fps=fps)
        except ImportError:
            raise ImportError(
                "Install 'mediapy' or 'imageio' to save videos: pip install mediapy"
            )


def _save_image(frame: np.ndarray, output_path: str) -> None:
    """Save a single frame as an image file."""
    try:
        from PIL import Image
        Image.fromarray(frame).save(output_path)
    except ImportError:
        try:
            import imageio
            imageio.imwrite(output_path, frame)
        except ImportError:
            raise ImportError(
                "Install 'Pillow' or 'imageio' to save images: pip install Pillow"
            )


class _ModelInitVisuals(NamedTuple):
    """The floor/light initial-state snapshot ``__init__`` needs to seed ``vis_state``.

    Returned by :meth:`Visualizer._rebuild_model_derived_state` rather than stashed on
    ``self``: ``__init__``'s ``vis_state`` literal is the only reader, so its correctness
    should not depend on attributes a prior call happened to leave behind. ``rebind_model``
    calls the same method and simply discards this -- ``vis_state`` already exists by the
    time a rebind happens and is deliberately left alone.
    """
    lights: list
    floor_rgb: list
    floor_alpha: float
    floor_mat_props: dict


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Visualizer:
    """Self-contained generic visualizer for any MuJoCo model.

    Wraps a MjModel with visual state (colors, lighting, floor, skybox, camera)
    and provides methods to render frames, videos, and camera pans.  All
    rendering is done with the standard ``mujoco.Renderer`` (CPU offscreen);
    no GPU, JAX, or notebook environment required.

    Args:
        xml_path:       Path to a MuJoCo XML model file.  Mutually exclusive
                        with *model*.
        model:          Pre-loaded ``mujoco.MjModel``.  Mutually exclusive with
                        *xml_path*.
        settings_json:  Optional path to a pose_tuner JSON settings file to
                        load immediately.
        joint_names:    Optional list of joint names to keep (calls
                        ``filter_model_to_config_joints`` before compiling).
        amputate:       Passed to ``filter_model_to_config_joints`` when
                        *joint_names* is provided.  See that function's docs.
    """

    def __init__(
        self,
        xml_path: Optional[str] = None,
        model: Optional[mujoco.MjModel] = None,
        *,
        spec: Optional[mujoco.MjSpec] = None,
        anatomy: Optional[Union[AnatomyConfig, str, dict]] = None,
        settings_json: Optional[str] = None,
        joint_names: Optional[List[str]] = None,
        amputate: Union[bool, str, List[str]] = False,
        floor_xml: Optional[str] = None,
        attach_body: Optional[str] = None,
        suffix: str = '',
        model_transform: Optional[Callable] = None,
        excluded_suffix: Optional[str] = None,
    ):
        """
        Args:
            xml_path:        Path to a MuJoCo XML file. Mutually exclusive with *model*.
            model:           Pre-loaded MjModel.
            anatomy:         AnatomyConfig, or path/dict for ``load_config``.
                             ``None`` => one category per top-level body.
            settings_json:   Optional settings JSON to apply immediately.
            joint_names:     If given, ``filter_model_to_config_joints`` is run
                             on the spec before compile.
            amputate:        Forwarded to ``filter_model_to_config_joints``.
            floor_xml:       Optional XML file containing a floor scene; the
                             loaded model is attached to this scene at
                             ``attach_body`` (or anatomy.root_body).
            attach_body:     Body name on the loaded model to attach to floor.
            suffix:          Name suffix passed through to attach.
            model_transform: Optional callable ``MjSpec -> MjSpec`` applied
                             before compile. Use for domain-specific tweaks
                             (e.g. fly flight setup).
            excluded_suffix: Geom-name suffix excluded from category colouring
                             (see ``self.excluded_suffix`` below).
        """
        if sum(x is not None for x in (xml_path, model, spec)) > 1:
            raise ValueError("Provide only one of xml_path, model, or spec.")

        if not isinstance(anatomy, AnatomyConfig):
            anatomy = load_config(anatomy)
        self.anatomy: AnatomyConfig = anatomy

        # Geoms whose name ends with this are excluded from category colouring and get the
        # `ghost` tint/alpha instead. Passed in rather than hardcoded: this package must not
        # know that the caller's second body is a "-ghost" fly. When None, nothing is excluded
        # and behaviour is exactly as before.
        self.excluded_suffix = excluded_suffix

        # --- Build the spec / model ---
        if joint_names is not None:
            if xml_path is None:
                raise ValueError("joint_names filtering requires xml_path.")
            from mujoco_visualizer.model_utils import filter_model_to_config_joints
            spec = mujoco.MjSpec.from_file(xml_path)
            spec = filter_model_to_config_joints(joint_names, spec=spec, amputate=amputate)
        elif xml_path is not None and (floor_xml is not None or model_transform is not None):
            spec = mujoco.MjSpec.from_file(xml_path)

        if spec is not None and floor_xml is not None:
            child_spec = spec
            target_spec = mujoco.MjSpec.from_file(floor_xml)
            spawn_frame = target_spec.worldbody.add_frame(
                pos=[0, 0, -0.005], quat=[1, 0, 0, 0])
            root = attach_body or self.anatomy.root_body
            if root is None:
                raise ValueError(
                    "floor_xml requires attach_body= or anatomy.root_body."
                )
            spawn_frame.attach_body(child_spec.body(root), "", suffix)
            spec = target_spec

        if spec is not None and model_transform is not None:
            spec = model_transform(spec)

        if spec is not None:
            self.model = spec.compile()
        elif xml_path is not None:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self.model = model

        self.data = mujoco.MjData(self.model)

        # If anatomy was empty, auto-derive one from the loaded model.
        if not self.anatomy.categories:
            self.anatomy = _auto_anatomy(self.model)

        # Save originals for reset / color baking
        self._orig_geom_rgba = self.model.geom_rgba.copy()
        _init_visuals = self._rebuild_model_derived_state()

        # Initialize vis_state (mirrors notebook vis_state)
        self.vis_state: dict = {
            'colors':      {cat: self._cat_default_hex.get(cat, '#888888')
                            for cat in self.anatomy.category_names},
            'geom_colors': {},   # {geom_id (int): hex str} per-geom overrides
            'alpha': 1.0,
            # Applied only to geoms matching `excluded_suffix`. alpha here REPLACES the global
            # alpha for those geoms rather than multiplying with it, so the number a UI shows is
            # the alpha that renders.
            'ghost': {'tint': '#cccccc', 'alpha': 0.3},
            'vis_flags': {
                'contact_points': False, 'contact_forces': False,
                'actuators': False, 'joints': False, 'transparent': False,
                'shadows': True, 'wireframe': False,
                # True, not False, because mjVIS_TENDON defaults ON in a bare MjvOption():
                # the fly renders 1449 scene geoms with tendons drawn and 802 without, so
                # defaulting this to False would silently change every existing render path.
                'tendon': True,
            },
            'geom_groups': [True, True, True, True, False, False],
            'site_groups':  [True, True, True, True, True,  False],
            'camera': {
                'mode': 'free', 'named': '',
                'azimuth': 180.0, 'elevation': -30.0, 'distance': 0.3,
                'lookat': [0.0, 0.0, 0.0],
                'free_type': 'free', 'trackbody': '', 'fixedcamid': '',
            },
            'lighting': {
                'lights': _init_visuals.lights,
                'use_dual_lighting':   False,
                'use_scale_lights':    False,
                'scale_lights_factor': 1.25,
                'headlight': {
                    'active':   bool(self.model.vis.headlight.active),
                    'ambient':  list(map(float, self.model.vis.headlight.ambient)),
                    'diffuse':  list(map(float, self.model.vis.headlight.diffuse)),
                    'specular': list(map(float, self.model.vis.headlight.specular)),
                },
            },
            'floor': {
                'color':       _rgb_to_hex(_init_visuals.floor_rgb),
                'alpha':       _init_visuals.floor_alpha,
                'texrepeat_x': _init_visuals.floor_mat_props['texrepeat'][0],
                'texrepeat_y': _init_visuals.floor_mat_props['texrepeat'][1],
                'reflectance': _init_visuals.floor_mat_props['reflectance'],
                'shininess':   _init_visuals.floor_mat_props['shininess'],
                'emission':    _init_visuals.floor_mat_props['emission'],
            },
            'skybox': {
                'show':    True,
                'sky_top': _rgb_to_hex([0.4, 0.6, 0.8]),
                'sky_bot': _rgb_to_hex([0.0, 0.0, 0.0]),
            },
            # Force/torque arrow scaling (mjVIS_CONTACTFORCE draws with these). Read from
            # ``self.model.vis`` rather than hardcoded -- unlike vis_flags these live on
            # MjModel.vis, not MjvOption, and MuJoCo's own library default (map.force=0.005)
            # is not what the fly model's MJCF sets (2e-05); hardcoding here would silently
            # overwrite the MJCF's value the moment a Visualizer is constructed.
            'forces': {
                'map_force':          float(self.model.vis.map.force),
                'map_torque':         float(self.model.vis.map.torque),
                'scale_forcewidth':   float(self.model.vis.scale.forcewidth),
                'scale_contactwidth': float(self.model.vis.scale.contactwidth),
                'scale_contactheight': float(self.model.vis.scale.contactheight),
            },
            # Muscle-tendon activation colouring/thickening (see ``apply_tendon_activation``).
            # ``max_width``/``min_width`` are read from ``self.model.tendon_width`` -- NOT
            # hardcoded -- for the same reason ``forces`` above reads ``self.model.vis``: the
            # fly's MJCF ships tendon widths (min ~0.00015, max ~0.003) that a hardcoded
            # default would silently overwrite the moment a Visualizer is constructed. A model
            # with no tendons at all has no widths to read, so it falls back to
            # render_video_pan's own long-standing defaults (0.003/0.0005) -- there is nothing
            # else to read, and this group is inert on such a model anyway (``enabled`` stays
            # off and there is nothing for it to colour).
            # ``ctrl_full_scale`` is a REFERENCE/full-scale |ctrl| value for normalisation, not
            # a measured one -- see ``default_tendon_ctrl_full_scale``'s docstring for the
            # measured gap on the real fly (ctrlrange ceiling 1.05 vs. a trained policy's
            # actual |ctrl| sitting at p50 0.0246 / p90 0.0555 / p99 0.1442 / max 0.6287 on a
            # real rollout) that makes this model-only default look uniformly dim on real
            # data. It is deliberately overridable: a caller holding the actual rollout (the
            # serve-layer launcher) should replace it with a measured percentile of |ctrl|
            # across that rollout -- this key exists so a viewer with nothing else to go on
            # still has a principled model-only fallback.
            'tendons': {
                'enabled':   False,
                # Which caller-supplied colour scheme (see Session's ``actuator_color_schemes``)
                # recolours tendons -- 'function' is not itself a registered name here (this
                # package ships no palette), it is just the stable default that makes the key
                # exist so ``scene_message``'s settings payload always advertises 'color_by',
                # which is what lets a UI control bind to it before any scheme is registered.
                'color_by':  'function',
                'max_width': (
                    float(self.model.tendon_width.max()) if self.model.ntendon else 0.003
                ),
                'min_width': (
                    float(self.model.tendon_width.min()) if self.model.ntendon else 0.0005
                ),
                'min_alpha': 0.05,
                'baseline':  0.0,
                'ctrl_full_scale': default_tendon_ctrl_full_scale(self.model),
            },
            # Recorded per-frame force-sensor arrows (drawn via ``add_arrow_to_scene`` by a
            # ``modify_scene_fns`` callable a caller registers on ``Session.scene_modifiers`` --
            # nothing in THIS package reads this group or draws anything from it; see
            # ``add_arrow_to_scene``/``Session.render`` for the seam that would). Off by
            # default: an arrow group with nothing feeding it real per-frame force data would
            # otherwise draw at a stale/zero pose the moment a caller flips it on by habit.
            #
            # 'scale' is seeded from ``self._default_force_arrow_scale`` (see
            # ``default_force_arrow_scale``'s docstring) rather than a hardcoded number,
            # because a fixed default is invisible on any model whose mass/gravity/extent it
            # was not tuned for -- the same reasoning ``ctrl_full_scale`` above and
            # ``_apply_forces_vis``'s five fields already follow. Deliberately left UNTOUCHED
            # by rebind_model/swap_model, exactly like 'tendons' ctrl_full_scale above (see
            # that key's own comment) and UNLIKE 'forces' (which _carry_vis_state_across_swap
            # DOES reapply, because it lives on model.vis and gets wiped by a fresh model
            # object -- this key lives only in vis_state, so nothing wipes it, and a caller's
            # explicit override deserves to survive a swap the same way ctrl_full_scale's does).
            # self._default_force_arrow_scale IS still kept fresh for the CURRENTLY active
            # model on every rebind (see _rebuild_model_derived_state) -- so a consumer that
            # wants "the right default for THIS model" can read that attribute directly,
            # even though this vis_state key itself is not automatically reset to it.
            #
            # 'radius' (arrow shaft width) reuses 0.003, the same default every existing
            # arrow-drawing helper in this module already uses (``add_arrow_to_scene``,
            # ``add_aero_force_arrows_to_scene``) -- consistent with them rather than a new
            # number, and not model-derived: unlike 'scale' there is no principled formula for
            # it in the task this group exists for, so consistency with the rest of the module
            # is the least-surprising choice.
            'force_arrows': {
                'enabled': False,
                'scale':   self._default_force_arrow_scale,
                'radius':  0.003,
            },
            # Measured mocap markers, drawn by a caller-registered scene modifier (see
            # scripts/rollout_viewer/launch.py in the fly_neuromech repo). This package
            # ships no marker DATA -- like 'force_arrows', the group is the render-side half
            # of a feature whose data half lives in the consumer.
            #
            # 'enabled' defaults TRUE, unlike force_arrows: markers exist to show how well a
            # solved pose matches what was measured, which is the first thing a viewer of IK
            # output wants to see. Deliberate asymmetry, not an oversight.
            'markers': {
                'enabled': True,
                'radius':  default_marker_radius(self.model),
            },
            'camera_presets': {},
        }

        if settings_json is not None:
            self.load_settings(settings_json)

    def _rebuild_model_derived_state(self) -> _ModelInitVisuals:
        """(Re)compute everything derived purely from ``self.model``: baked colors, geom
        categories, floor/skybox/light detection, and per-category default hex.

        Called from ``__init__`` and from :meth:`rebind_model`. Deliberately does NOT touch
        ``vis_state`` -- a rebind must be able to either carry the caller's current look
        across to the new model (:meth:`Session.swap_model` does exactly this) or leave it to
        be replaced wholesale, and clobbering it here would make both impossible.

        Callers must set ``self._orig_geom_rgba = self.model.geom_rgba.copy()`` before calling
        this, since baking below reads it as the pre-bake original.

        Returns the floor/light initial-state snapshot (see :class:`_ModelInitVisuals`) so
        ``__init__`` can fold it straight into its ``vis_state`` literal without depending on
        instance attributes this method happens to leave behind; ``rebind_model`` calls this
        for its side effects alone and discards the return value.
        """
        self._orig_mat_rgba  = self.model.mat_rgba.copy()

        # Build body-segment → geom_id categorization
        self._geom_categories = build_geom_categories(self.model, self.anatomy)

        # Bake material rgba into geom_rgba for all categorized geoms so they
        # can be independently recolored via geom_rgba alone.
        _cat_geom_ids = {i for idxs in self._geom_categories.values() for i in idxs}
        _DEFAULT_GEOM_RGBA = np.array([0.5, 0.5, 0.5, 1.0])
        for gi in _cat_geom_ids:
            mid = int(self.model.geom_matid[gi])
            if mid >= 0:
                if np.allclose(self._orig_geom_rgba[gi], _DEFAULT_GEOM_RGBA):
                    # Geom has no explicit rgba override; use material rgba directly.
                    self.model.geom_rgba[gi] = self._orig_mat_rgba[mid]
                else:
                    # Geom has an explicit rgba; modulate material rgba by it.
                    self.model.geom_rgba[gi] = np.clip(
                        self._orig_mat_rgba[mid] * self._orig_geom_rgba[gi], 0.0, 1.0
                    )
                self.model.geom_matid[gi] = -1
        # Hide *_inertial helper geoms (e.g. wing_left_inertial bounding box).
        # Substring check catches MjSpec.attach_body(..., suffix=...) renames
        # such as wing_left_inertial_fly1.
        # Also resolve which geoms are excluded from category colouring (see
        # ``self.excluded_suffix`` in __init__), by geom name rather than category, since an
        # excluded geom otherwise falls into the very same category as its non-excluded twin.
        self._excluded_geom_ids: set = set()
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
            if '_inertial' in gname:
                self.model.geom_rgba[gid, 3] = 0.0
            # The truthiness guard also makes excluded_suffix="" a no-op, even though every
            # name technically ends with "" under str.endswith. Only None is required to be a
            # no-op; "" piggybacking on that is deliberate, not an oversight.
            if self.excluded_suffix and gname.endswith(self.excluded_suffix):
                self._excluded_geom_ids.add(gid)

        # Refresh originals after baking
        self._orig_geom_rgba = self.model.geom_rgba.copy()

        # The FALLBACK default for vis_state['force_arrows']['scale'] -- recomputed here (i.e.
        # on every __init__ AND every rebind_model/swap) so it never goes stale for whichever
        # model is CURRENTLY self.model, exactly like _cat_default_hex and the floor/light
        # snapshot below. Unlike those, nothing in THIS package re-applies it onto vis_state on
        # a swap: vis_state['force_arrows']['scale'] itself is intentionally left untouched by
        # a swap (see Session._carry_vis_state_across_swap's docstring, and this class's own
        # __init__ comment on 'force_arrows', for why that mirrors 'tendons' ctrl_full_scale
        # rather than 'forces') -- a caller who explicitly set it is assumed to mean it on
        # whichever model is active. This attribute exists so that claim is falsifiable: a
        # future consumer (or a test) that wants "the right default for the CURRENT model" has
        # somewhere to read it from without recomputing the formula itself.
        self._default_force_arrow_scale = default_force_arrow_scale(self.model)

        # Cached render context. Building one re-uploads every mesh to the GPU (399 ms on
        # the fly model vs 8.6 ms reused), so exactly one is kept and reused; a resolution
        # change closes it and builds another. Released by close().
        self._renderer_cache = None
        self._renderer_key = None
        # Fingerprint of the skybox settings the current tex_data was generated from.
        # None means "never generated", so the first apply always runs.
        self._sky_fingerprint = None
        # Whether the regenerated skybox texture still needs pushing to a live render context.
        # SEPARATE from `_sky_fingerprint` on purpose: the fingerprint answers "is `model.tex_data`
        # already correct", this answers "has a context seen it". Collapsing the two meant the
        # single True `_apply_sky_props` returns was consumed by whichever caller applied first --
        # `load_settings` does its own `_apply_all()` -- leaving `render_with`'s call to return
        # False and the upload to never happen, so the canvas kept the previous sky.
        #
        # CAVEAT (pre-existing, not fixed here): neither `_sky_fingerprint` nor
        # `_skybox_tex_id` is invalidated by `rebind_model`/`swap_model`. After a clip swap, a
        # restored `skybox` setting that happens to match the fingerprint LATCHED FROM THE OLD
        # MODEL makes `_apply_sky_props` return early and never write the new model's
        # `tex_data` -- same class of bug as the render-flags write-once defect this task's
        # sibling fix addressed, and it also affects `load_settings`/`apply_render` after a
        # swap, not just Reset. Left unfixed here: it needs its own testing and is a separate
        # change from this task's scope.
        self._sky_needs_upload = False

        # Detect floor geom and material
        self._floor_geom_id: Optional[int] = next(
            (i for i in range(self.model.ngeom)
             if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE),
            None
        )
        self._floor_mat_id: Optional[int] = (
            int(self.model.geom_matid[self._floor_geom_id])
            if self._floor_geom_id is not None
               and int(self.model.geom_matid[self._floor_geom_id]) >= 0
            else None
        )

        # Detect skybox texture
        self._skybox_tex_id: int = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_TEXTURE, 'skybox'
        )

        # Default hex colors per category (from baked geom_rgba)
        self._cat_default_hex: dict = {}
        for cat in self.anatomy.category_names:
            for gid in self._geom_categories.get(cat, []):
                r = self._orig_geom_rgba[gid]
                if r[3] > 0.01:
                    self._cat_default_hex[cat] = _rgb_to_hex(r[:3])
                    break
            if cat not in self._cat_default_hex:
                self._cat_default_hex[cat] = '#888888'

        # Floor initial state
        _floor_rgb = (
            list(self._orig_geom_rgba[self._floor_geom_id, :3])
            if self._floor_geom_id is not None else [0.5, 0.5, 0.5]
        )
        _floor_alpha = (
            float(self._orig_geom_rgba[self._floor_geom_id, 3])
            if self._floor_geom_id is not None else 1.0
        )
        _floor_mat_props = {'texrepeat': [1.0, 1.0], 'reflectance': 0.2,
                            'shininess': 0.5, 'emission': 0.0}
        if self._floor_mat_id is not None:
            _floor_mat_props['texrepeat'] = list(
                map(float, self.model.mat_texrepeat[self._floor_mat_id])
            )
            _floor_mat_props['reflectance'] = float(
                self.model.mat_reflectance[self._floor_mat_id]
            )
            _floor_mat_props['shininess'] = float(
                self.model.mat_shininess[self._floor_mat_id]
            )
            _floor_mat_props['emission'] = float(
                self.model.mat_emission[self._floor_mat_id]
            )

        # Light initial state
        _init_lights = []
        for li in range(min(self.model.nlight, 3)):
            az, el = _dir_to_az_el(self.model.light_dir[li])
            _init_lights.append({
                'active':   bool(self.model.light_active[li]),
                'ambient':  list(map(float, self.model.light_ambient[li])),
                'diffuse':  list(map(float, self.model.light_diffuse[li])),
                'specular': list(map(float, self.model.light_specular[li])),
                'dir_az': az, 'dir_el': el,
            })
        return _ModelInitVisuals(
            lights=_init_lights,
            floor_rgb=_floor_rgb,
            floor_alpha=_floor_alpha,
            floor_mat_props=_floor_mat_props,
        )

    # ── Settings I/O ─────────────────────────────────────────────────────────

    def load_settings(self, json_path_or_dict: Union[str, dict]) -> None:
        """Load visual settings from a pose_tuner JSON file or dict.

        Updates vis_state and immediately applies all settings to the model.
        """
        if isinstance(json_path_or_dict, str):
            from mujoco_visualizer.render_settings import _resolve_settings_path
            try:
                path = _resolve_settings_path(json_path_or_dict)
            except FileNotFoundError:
                path = Path(json_path_or_dict)
            with open(path) as f:
                settings = json.load(f)
        else:
            settings = json_path_or_dict

        # NOTE: geom_render_state intentionally not applied — it's a raw
        # gid->rgba cache baked against a specific model topology and will
        # clobber unrelated geoms (e.g. floor) when applied to a composed
        # model. The name-based colors/geom_colors path below is correct.

        # Merge settings into vis_state
        # 'ghost' (tint/alpha for the translucent reference overlay) is included here so a
        # preset written after it was added round-trips it too. A preset written BEFORE it
        # existed simply has no 'ghost' key -- the `if key in settings` guard below means
        # that case takes neither branch and self.vis_state['ghost'] is left at whatever
        # default __init__ set, never an error and never a synthesized value.
        for key in ('colors', 'geom_colors', 'alpha', 'vis_flags',
                    'geom_groups', 'site_groups', 'camera', 'lighting',
                    'floor', 'skybox', 'ghost', 'forces', 'tendons', 'force_arrows',
                    'markers'):
            if key in settings:
                if isinstance(settings[key], dict) and isinstance(self.vis_state.get(key), dict):
                    self.vis_state[key] = {**self.vis_state[key], **settings[key]}
                else:
                    self.vis_state[key] = copy.deepcopy(settings[key])

        # Convert geom_colors string keys to int
        if 'geom_colors' in settings:
            self.vis_state['geom_colors'] = {
                int(k): v for k, v in settings['geom_colors'].items()
            }

        # Camera presets
        if 'camera_presets' in settings:
            self.vis_state['camera_presets'].update(settings['camera_presets'])

        self._apply_all()

    def save_settings(self, json_path: str) -> None:
        """Save current vis_state to a JSON file.

        Bare names (e.g. ``'MyPreset'`` or ``'MyPreset.json'``) are written
        into the package's ``settings/`` directory so they show up under
        ``list_available_settings()``. Pass an absolute path or a name
        containing a path separator to save elsewhere.
        """
        from mujoco_visualizer.render_settings import _SETTINGS_DIR
        p = Path(json_path)
        if not p.is_absolute() and p.parent == Path('.'):
            if p.suffix != '.json':
                p = p.with_suffix('.json')
            p = _SETTINGS_DIR / p.name
            p.parent.mkdir(parents=True, exist_ok=True)
        json_path = str(p)
        self._apply_geom_colors()
        _all_cat_ids = sorted({i for idxs in self._geom_categories.values() for i in idxs})
        geom_render_state = {
            str(i): list(map(float, self.model.geom_rgba[i])) for i in _all_cat_ids
        }
        geom_colors_str = {str(k): v for k, v in self.vis_state['geom_colors'].items()}
        data = {
            'colors':            self.vis_state['colors'],
            'geom_colors':       geom_colors_str,
            'alpha':             self.vis_state['alpha'],
            'vis_flags':         dict(self.vis_state['vis_flags']),
            'geom_groups':       self.vis_state['geom_groups'][:],
            'site_groups':       self.vis_state['site_groups'][:],
            'camera':            copy.deepcopy(self.vis_state['camera']),
            'lighting':          copy.deepcopy(self.vis_state['lighting']),
            'floor':             copy.deepcopy(self.vis_state['floor']),
            'skybox':            copy.deepcopy(self.vis_state['skybox']),
            'forces':            copy.deepcopy(self.vis_state['forces']),
            'tendons':           copy.deepcopy(self.vis_state['tendons']),
            'force_arrows':      copy.deepcopy(self.vis_state.get('force_arrows', {})),
            # .get(..., {}) for the same reason 'ghost' uses it: this key postdates every
            # bundled preset, so a Visualizer rebuilt from an old one has no entry to copy.
            'markers':           copy.deepcopy(self.vis_state.get('markers', {})),
            'geom_render_state': geom_render_state,
            'camera_presets':    self.vis_state.get('camera_presets', {}),
            # .get(..., {}), not ['ghost'], because this key was added after every existing
            # vis_state literal and after all 17 bundled presets -- on a Visualizer built
            # before it exists (or rebuilt from an old preset that never sets it), there is
            # no 'ghost' entry to copy. Saving {} in that case is harmless: load_settings's
            # `if 'ghost' in settings` guard treats an empty dict as "nothing to merge",
            # same as the key being absent outright.
            'ghost':             copy.deepcopy(self.vis_state.get('ghost', {})),
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    # ── Apply helpers (mirror notebook apply_* functions) ─────────────────────

    def _apply_geom_colors(self) -> None:
        alpha = self.vis_state['alpha']
        geom_overrides = self.vis_state['geom_colors']
        excluded_ids = self._excluded_geom_ids
        ghost = self.vis_state['ghost']
        ghost_rgb = _hex_to_rgb(ghost['tint'])
        for cat, idxs in self._geom_categories.items():
            cat_rgb = _hex_to_rgb(self.vis_state['colors'].get(cat, '#888888'))
            for i in idxs:
                if self._orig_geom_rgba[i, 3] < 0.01:
                    continue
                if i in excluded_ids:
                    # Excluded geoms (see `excluded_suffix`) never take the category colour or
                    # per-geom override -- they get the ghost tint, and ghost.alpha REPLACES
                    # rather than multiplies the global alpha here.
                    self.model.geom_rgba[i, :3] = ghost_rgb
                    self.model.geom_rgba[i,  3] = ghost['alpha']
                    continue
                rgb = _hex_to_rgb(geom_overrides[i]) if i in geom_overrides else cat_rgb
                self.model.geom_rgba[i, :3] = rgb
                self.model.geom_rgba[i,  3] = self._orig_geom_rgba[i, 3] * alpha

    def _apply_lighting(self) -> None:
        for i, ld in enumerate(self.vis_state['lighting']['lights']):
            if i >= self.model.nlight:
                break
            self.model.light_active[i]   = int(ld['active'])
            self.model.light_ambient[i]  = ld['ambient']
            self.model.light_diffuse[i]  = ld['diffuse']
            self.model.light_specular[i] = ld['specular']
            self.model.light_dir[i]      = _az_el_to_dir(ld['dir_az'], ld['dir_el'])
        hl = self.vis_state['lighting']['headlight']
        self.model.vis.headlight.active      = int(hl['active'])
        self.model.vis.headlight.ambient[:]  = hl['ambient']
        self.model.vis.headlight.diffuse[:]  = hl['diffuse']
        self.model.vis.headlight.specular[:] = hl['specular']

    def _apply_floor_props(self) -> None:
        if self._floor_geom_id is None:
            return
        fld = self.vis_state['floor']
        rgb = _hex_to_rgb(fld['color'])
        self.model.geom_rgba[self._floor_geom_id] = [*rgb, fld['alpha']]
        if self._floor_mat_id is not None:
            self.model.mat_rgba[self._floor_mat_id]        = [*rgb, fld['alpha']]
            self.model.mat_texrepeat[self._floor_mat_id]   = [fld['texrepeat_x'],
                                                               fld['texrepeat_y']]
            self.model.mat_reflectance[self._floor_mat_id] = fld['reflectance']
            self.model.mat_shininess[self._floor_mat_id]   = fld['shininess']
            self.model.mat_emission[self._floor_mat_id]    = fld['emission']

    def _apply_sky_props(self) -> bool:
        """Regenerate the skybox texture, but only if its settings changed.

        Returns True if ``tex_data`` was rewritten, so a caller holding a live render
        context knows to re-upload it.

        Guarded because regenerating a 100x600 skybox costs ~4 ms -- about 45% of a 640x480
        render -- and it previously ran on every single frame. The guard is a fingerprint
        rather than a dirty flag because ``vis_state`` is a plain dict that gui.py,
        widget_gui.py and Session.apply_render all mutate directly; a ``mark_dirty()`` API
        would be silently bypassed by every one of them.
        """
        if self._skybox_tex_id < 0:
            return False
        sky = self.vis_state['skybox']
        fingerprint = (sky.get('show', True), sky['sky_top'], sky['sky_bot'])
        if fingerprint == self._sky_fingerprint:
            return False

        pixels = _make_sky_pixels(
            self.model, self._skybox_tex_id,
            _hex_to_rgb(sky['sky_top']), _hex_to_rgb(sky['sky_bot'])
        )
        if pixels is None:
            return False
        adr = int(self.model.tex_adr[self._skybox_tex_id])
        nchan = int(self.model.tex_nchannel[self._skybox_tex_id]) if hasattr(
            self.model, 'tex_nchannel') else 3
        if nchan == 4:
            rgba = np.ones((len(pixels), 4), dtype=np.uint8) * 255
            rgba[:, :3] = pixels
            flat = rgba.flatten()
        else:
            flat = pixels.flatten()
        tex_buf = getattr(self.model, 'tex_data', None)
        if tex_buf is None:
            tex_buf = getattr(self.model, 'tex_rgb', None)
        if tex_buf is None:
            return False
        tex_buf[adr:adr + len(flat)] = flat
        self._sky_fingerprint = fingerprint
        self._sky_needs_upload = True
        return True

    def _apply_forces(self) -> None:
        """Write ``vis_state['forces']`` onto ``self.model.vis`` (see :func:`_apply_forces_vis`
        for why this cannot be folded into a scene-option flag like the other vis_flags)."""
        _apply_forces_vis(self.vis_state['forces'], self.model)

    def _apply_all(self) -> bool:
        """Apply all vis_state properties to the model.

        Returns True if the skybox texture was regenerated, meaning any live render context
        needs it re-uploaded. The other appliers are cheap (measured 0.025 ms combined) and
        run unconditionally.
        """
        self._apply_geom_colors()
        self._apply_lighting()
        self._apply_floor_props()
        self._apply_forces()
        return self._apply_sky_props()

    def _build_scene_option(self) -> mujoco.MjvOption:
        opt = mujoco.MjvOption()
        f = self.vis_state['vis_flags']
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = f.get('contact_points', False)
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = f.get('contact_forces',  False)
        opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR]     = f.get('actuators',       False)
        opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]        = f.get('joints',          False)
        opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]  = f.get('transparent',     False)
        # Default True: mjVIS_TENDON is ON in a bare MjvOption(), so anything else here would
        # change what every caller already renders. Wired at all because the serve layer's
        # `export.tendons` field and the render-cost benchmark both set this key, and until
        # this line existed both were writing to a flag nothing read.
        opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON]       = f.get('tendon',          True)
        gg = self.vis_state['geom_groups']
        sg = self.vis_state['site_groups']
        for k in range(min(6, len(opt.geomgroup))):
            opt.geomgroup[k] = int(gg[k])
        for k in range(min(6, len(opt.sitegroup))):
            opt.sitegroup[k] = int(sg[k])
        return opt

    def _build_scene_modifiers(self) -> list:
        mods = []
        if self.vis_state['lighting'].get('use_dual_lighting', False):
            mods.append((dual_lighting, {}))
        if self.vis_state['lighting'].get('use_scale_lights', False):
            mods.append((scale_lights,
                         {'scale': self.vis_state['lighting'].get('scale_lights_factor', 1.25)}))
        return mods

    def get_camera(
        self, override: Optional[Union[str, mujoco.MjvCamera]] = None
    ) -> Union[str, mujoco.MjvCamera]:
        """Return camera from vis_state, or *override* if provided.

        *override* can be a named XML camera string, an MjvCamera, or the
        name of a camera preset from the loaded settings.
        """
        if override is not None:
            if isinstance(override, str):
                # Check XML cameras first
                if mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, override
                ) != -1:
                    return override
                # Fall back to settings presets
                presets = self.vis_state.get('camera_presets', {})
                if override in presets:
                    cfg = _resolve_preset(presets[override])
                    return self._cfg_to_mjvcamera(cfg)
            return override
        c = self.vis_state['camera']
        if c.get('mode', 'free') == 'named':
            named = c.get('named', '')
            if named and mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, named
            ) != -1:
                return named
            # fall through to free camera if named camera missing
        return self._cfg_to_mjvcamera(c)

    def _cfg_to_mjvcamera(self, c: dict) -> mujoco.MjvCamera:
        """Build an MjvCamera from a camera config dict."""
        free_type = c.get('free_type', 'free')
        mj_type, needs_body, needs_fixedcam = _FREE_TYPE_MAP.get(
            free_type, _FREE_TYPE_MAP['free']
        )
        cam = mujoco.MjvCamera()
        cam.type      = mj_type
        cam.azimuth   = c['azimuth']
        cam.elevation = c['elevation']
        cam.distance  = c['distance']
        cam.lookat[:] = c['lookat']
        if needs_fixedcam:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, c.get('fixedcamid', '')
            )
            cam.fixedcamid = max(cam_id, 0)
        if needs_body:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, c.get('trackbody', '')
            )
            cam.trackbodyid = max(body_id, 0)
        return cam

    def list_cameras(self) -> List[str]:
        """Return list of named cameras (from anatomy config, else from model)."""
        if self.anatomy.cameras:
            return list(self.anatomy.cameras)
        return [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}"
            for i in range(self.model.ncam)
        ]

    def list_presets(self) -> List[str]:
        """Return list of saved camera preset names."""
        return list(self.vis_state.get('camera_presets', {}).keys())

    # ── Rendering ─────────────────────────────────────────────────────────────

    def make_renderer(self, height: int = 480, width: int = 640) -> mujoco.Renderer:
        """A fresh Renderer for this model. The caller owns it and must ``close()`` it.

        Prefer :meth:`render_frame` / :meth:`render_with`, which reuse a cached one.
        """
        return mujoco.Renderer(self.model, height=height, width=width)

    def rebind_model(self, model: mujoco.MjModel) -> None:
        """Point this Visualizer at a different compiled model, keeping its settings.

        Rebuilds only what is derived from the model: the pristine colour/lighting baselines
        used for reset and colour baking, the geom categories, and ``data``. ``vis_state`` is
        left alone deliberately -- the caller may want to carry the current look across
        (Session.swap_model does) or replace it wholesale.

        Any cached renderer belongs to the OLD model's context and would draw the old
        geometry, so it is dropped here rather than left to be reused.
        """
        if self._renderer_cache is not None:
            self._renderer_cache.close()
            self._renderer_cache = None
            self._renderer_key = None
        self.model = model
        self.data = mujoco.MjData(model)
        self._orig_geom_rgba = self.model.geom_rgba.copy()
        self._rebuild_model_derived_state()

    def _cached_renderer(self, height: int, width: int) -> mujoco.Renderer:
        """The reused Renderer for (height, width), building it on first use.

        Exactly one is kept. A resolution change closes the old one rather than keeping a
        cache keyed by size: each Renderer holds GPU framebuffers plus every uploaded mesh
        (139 MB of them on the fly model), so a multi-entry cache would leak VRAM across a
        session that renders several resolutions.
        """
        key = (int(height), int(width))
        if self._renderer_cache is not None and self._renderer_key == key:
            return self._renderer_cache
        if self._renderer_cache is not None:
            self._renderer_cache.close()
            self._renderer_cache = None
        self._renderer_cache = self.make_renderer(height=key[0], width=key[1])
        self._renderer_key = key
        return self._renderer_cache

    def render_with(
        self,
        renderer: mujoco.Renderer,
        camera: Optional[Union[str, mujoco.MjvCamera]] = None,
        apply_settings: bool = True,
        modify_scene_fns: Optional[Sequence[Callable]] = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Render ``self.data`` as it currently stands into a caller-supplied *renderer*.

        Does NOT write ``qpos`` and does NOT call ``mj_forward`` -- the caller owns the
        state, which is what lets a stepping simulation render its own live data. Use
        :meth:`render_frame` for the set-a-pose-and-render-it case.

        Returns (renderer.height, renderer.width, 3) uint8.
        """
        if apply_settings:
            self._apply_all()
        if self._sky_needs_upload:
            # MuJoCo uploads textures when the render context is built, so a regenerated
            # skybox never reaches a context that already exists. Without this, reusing a
            # renderer silently pins the sky at whatever it was when the context was made
            # (verified: max pixel diff 0 across a red->green change, vs 255 with a fresh
            # context). ``_mjr_context`` is private to mujoco.Renderer; there is no public
            # accessor for the MjrContext.
            #
            # Gated on the sticky flag rather than on this call's `_apply_all()` return, so an
            # upload is not lost when some other caller applied the settings first.
            mujoco.mjr_uploadTexture(
                self.model, renderer._mjr_context, self._skybox_tex_id
            )
            self._sky_needs_upload = False

        cam = self.get_camera(camera)
        opt = self._build_scene_option()
        scene_mods = self._build_scene_modifiers()
        vf = self.vis_state['vis_flags']
        show_shadows = vf.get('shadows', True)
        show_wireframe = vf.get('wireframe', False)
        show_skybox = self.vis_state.get('skybox', {}).get('show', True)

        renderer.update_scene(self.data, camera=cam, scene_option=opt)
        # Assigned unconditionally, never `if not show_x: ... = False`. `update_scene()` does not
        # reset `scene.flags`, and these flags live on the renderer's scene object, which the live
        # viewer reuses for every frame -- so a one-directional write moved a flag once and never
        # back. Ticking wireframe in the Settings tab could not be unticked, unticking shadows
        # could not be undone, and `reset_render_settings` restored `vis_state` while the canvas
        # went on rendering the old flags. Found by the reset round-trip figure, which reported
        # 13/13 roots restored and a max per-pixel difference of 137: no test in this repo could
        # see it, because `vis_state` was correct the whole time.
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = show_shadows
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = show_wireframe
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = show_skybox
        for fn, kw in scene_mods:
            fn(renderer.scene, geom_xpos=self.data.geom_xpos, **kw)
        if modify_scene_fns:
            for fn in modify_scene_fns:
                fn(renderer.scene, data=self.data, frame_idx=frame_idx)
        return renderer.render().copy()

    def render_frame(
        self,
        qpos: np.ndarray,
        camera: Optional[Union[str, mujoco.MjvCamera]] = None,
        height: int = 480,
        width: int = 640,
        apply_settings: bool = True,
        modify_scene_fns: Optional[Sequence[Callable]] = None,
    ) -> np.ndarray:
        """Render a single frame at *qpos*.

        Reuses a cached render context, so repeated calls at one resolution pay the mesh
        upload once (399 ms -> 8.6 ms per frame on the fly model). Call :meth:`close` when
        done with the Visualizer to release it.

        Args:
            qpos:            Joint positions array (nq,).
            camera:          Named camera str, MjvCamera, or None (use vis_state).
            height, width:   Output resolution in pixels.
            apply_settings:  If True, apply vis_state to model before rendering.

        Returns:
            np.ndarray of shape (height, width, 3) uint8.
        """
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        renderer = self._cached_renderer(height, width)
        return self.render_with(
            renderer,
            camera=camera,
            apply_settings=apply_settings,
            modify_scene_fns=modify_scene_fns,
        )

    def close(self) -> None:
        """Release the cached render context.

        EGL teardown raises from ``Renderer.__del__`` if left to the garbage collector, so
        release it deliberately. Safe to call more than once.
        """
        if getattr(self, '_renderer_cache', None) is not None:
            self._renderer_cache.close()
            self._renderer_cache = None
            self._renderer_key = None

    def save_frame(
        self,
        qpos: np.ndarray,
        output_path: str,
        camera: Optional[Union[str, mujoco.MjvCamera]] = None,
        height: int = 2160,
        width: int = 3840,
    ) -> np.ndarray:
        """Render and save a high-quality single frame.

        Args:
            qpos:         Joint positions array (nq,).
            output_path:  Output file path (.png, .jpg, etc.).
            camera:       Camera override; defaults to vis_state camera.
            height, width: Output resolution (default 4K: 2160×3840).

        Returns:
            np.ndarray of the rendered frame.
        """
        frame = self.render_frame(qpos, camera=camera, height=height, width=width)
        _save_image(frame, output_path)
        return frame

    def render_video(
        self,
        qposes: np.ndarray,
        camera: Optional[Union[str, mujoco.MjvCamera]] = None,
        height: int = 480,
        width: int = 640,
        fps: int = 30,
        output_path: Optional[str] = None,
        show: bool = False,
        modify_scene_fns: Optional[Sequence[Callable]] = None,
    ) -> np.ndarray:
        """Render a video from a qpos trajectory.

        Args:
            qposes:      (T, nq) array of joint positions.
            camera:      Camera override; defaults to vis_state camera.
            height, width: Frame resolution.
            fps:         Playback frame rate (used when saving/displaying).
            output_path: If provided, save the video to this path (.mp4).
            show:        If True, display inline (requires mediapy).

        Returns:
            np.ndarray of shape (T, height, width, 3) uint8.
        """
        renderer = self._cached_renderer(height, width)
        frames = []
        for i, qpos in enumerate(qposes):
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            frames.append(
                self.render_with(
                    renderer,
                    camera=camera,
                    modify_scene_fns=modify_scene_fns,
                    frame_idx=i,
                )
            )

        video = np.stack(frames)
        if output_path is not None:
            _save_video(video, output_path, fps)
        if show:
            try:
                import mediapy
                mediapy.show_video(video, fps=fps)
            except ImportError:
                pass
        return video

    def render_video_pan(
        self,
        qposes: np.ndarray,
        cameras: List[mujoco.MjvCamera],
        height: int = 480,
        width: int = 640,
        fps: int = 30,
        output_path: Optional[str] = None,
        show: bool = False,
        ctrls: Optional[np.ndarray] = None,
        tendon_width: float = 0.003,
        tendon_min_width: float = 0.0005,
        tendon_alpha_min: float = 0.05,
        tendon_baseline: float = 0.0,
        actuator_color_fn: Optional[Callable] = None,
        modify_scene_fns: Optional[Sequence[Callable]] = None,
    ) -> np.ndarray:
        """Render a video with per-frame camera positions (for panning shots).

        Args:
            qposes:         (T, nq) array — must have the same length as *cameras*.
            cameras:        List of MjvCamera from :meth:`make_pan_cameras`.
            height, width:  Frame resolution.
            fps:            Playback frame rate.
            output_path:    If provided, save the video to this path.
            show:           If True, display inline (requires mediapy).
            ctrls:          Optional (T, nu) control signals for muscle visualization.
            tendon_width:   Max tendon rendering width at full activation.
            tendon_min_width: Min tendon width at zero activation.
            tendon_alpha_min: Minimum alpha for muscle tendons (default 0.05).
            tendon_baseline: Baseline added to normalized activation before
                scaling width and alpha (e.g. 0.3 makes low activations visible).
            actuator_color_fn: Optional callable ``(name: str) -> color`` where
                *color* is a hex string (e.g. ``'#d84a2e'``) or an RGBA
                4-tuple.  Falls back to ``self.actuator_color_fn`` then solid red.

        Returns:
            np.ndarray of shape (T, height, width, 3) uint8.
        """
        # Optional muscle visualization
        show_muscles = ctrls is not None
        orig_tendon_rgba = orig_tendon_width = act_to_ten = base_rgba = None
        ctrl_max = 1.0
        if show_muscles:
            ctrls = np.asarray(ctrls)
            # mjVIS_TENDON defaults to on in a freshly-built MjvOption() (verified), which
            # is what render_with constructs per frame, so no explicit override is needed.
            # Resolve color function: parameter > self attribute > solid red.
            _color_fn = actuator_color_fn or getattr(self, 'actuator_color_fn', None)
            act_to_ten, base_rgba = build_actuator_tendon_map(self.model, _color_fn)
            orig_tendon_rgba = self.model.tendon_rgba.copy()
            orig_tendon_width = self.model.tendon_width.copy()
            # Normalize to global max across all timesteps
            ctrl_max = max(float(np.abs(ctrls).max()), 1e-8)

        renderer = self._cached_renderer(height, width)
        frames = []
        for i, qpos in enumerate(qposes):
            self.data.qpos[:] = qpos
            if show_muscles and ctrls is not None:
                self.data.ctrl[:] = ctrls[i]
            mujoco.mj_forward(self.model, self.data)

            if show_muscles and act_to_ten is not None:
                apply_tendon_activation(
                    self.model,
                    ctrls[i],
                    act_to_ten,
                    base_rgba,
                    tendon_width=tendon_width,
                    tendon_min_width=tendon_min_width,
                    tendon_alpha_min=tendon_alpha_min,
                    tendon_baseline=tendon_baseline,
                    ctrl_max=ctrl_max,
                )

            frames.append(
                self.render_with(
                    renderer,
                    camera=cameras[i],
                    modify_scene_fns=modify_scene_fns,
                    frame_idx=i,
                )
            )

        if show_muscles and orig_tendon_rgba is not None:
            self.model.tendon_rgba[:] = orig_tendon_rgba
            self.model.tendon_width[:] = orig_tendon_width

        video = np.stack(frames)
        if output_path is not None:
            _save_video(video, output_path, fps)
        if show:
            try:
                import mediapy
                mediapy.show_video(video, fps=fps)
            except ImportError:
                pass
        return video

    # ── Camera utilities ──────────────────────────────────────────────────────

    def make_pan_cameras(
        self,
        preset_names: List[str],
        total_frames: int = 120,
        segment_weights: Optional[List[float]] = None,
        loop: bool = False,
        settings: Optional[Union[str, dict]] = None,
    ) -> List[mujoco.MjvCamera]:
        """Build a list of MjvCamera objects for a smooth camera pan.

        Args:
            preset_names:     Ordered list of preset names (at least 2).
                              Presets must exist in vis_state['camera_presets']
                              or in an externally supplied *settings* dict.
            total_frames:     Total number of frames to generate.
            segment_weights:  Relative time budget per segment (normalised).
            loop:             If True, append a segment back to the first preset.
            settings:         Optional alternative settings dict/path to look up
                              presets from (instead of self.vis_state).

        Returns:
            List of ``mujoco.MjvCamera`` of length EXACTLY *total_frames*, for any weights and
            any ``total_frames >= 1`` -- see :func:`allocate_segment_frames`, which owns that
            guarantee and explains why a segment is allowed 0 frames when there are fewer
            frames than segments.
        """
        if settings is not None:
            if isinstance(settings, str):
                with open(settings) as f:
                    settings = json.load(f)
            presets = settings.get('camera_presets', {})
        else:
            presets = self.vis_state.get('camera_presets', {})

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

        seg_frames = allocate_segment_frames(weights, total_frames)

        cameras = []
        for seg in range(n_segs):
            A = keyframes[seg]
            B = keyframes[seg + 1]
            n = seg_frames[seg]
            for fi in range(n):
                t = _cosine_ease(fi / n)
                cameras.append(_build_pan_camera(self.model, A, B, t))

        return cameras

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def scan_frames(
        qposes: np.ndarray,
        output_dir: str,
        viz: 'Visualizer',
        frame_indices: Optional[Union[List[int], np.ndarray]] = None,
        camera: Optional[Union[str, mujoco.MjvCamera]] = None,
        height: int = 2160,
        width: int = 3840,
        prefix: str = 'frame',
    ) -> List[str]:
        """Render and save multiple high-quality frames from a trajectory.

        Useful for scanning a video to find good frames before committing to a
        full high-quality render.

        Args:
            qposes:        (T, nq) trajectory array.
            output_dir:    Directory to save frames.
            viz:           Visualizer instance.
            frame_indices: Which frame indices to render.  None = evenly spaced 10.
            camera:        Camera override.
            height, width: Output resolution (default 4K).
            prefix:        File name prefix.

        Returns:
            List of output file paths.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        if frame_indices is None:
            T = len(qposes)
            frame_indices = list(np.linspace(0, T - 1, min(10, T), dtype=int))

        viz._apply_all()
        paths = []
        for idx in frame_indices:
            out = str(Path(output_dir) / f'{prefix}_{idx:06d}.png')
            viz.save_frame(qposes[idx], out, camera=camera, height=height, width=width)
            paths.append(out)
            print(f'  Saved {out}')
        return paths
