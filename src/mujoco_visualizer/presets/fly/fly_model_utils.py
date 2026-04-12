"""Fly-specific MjSpec helpers (flight aerodynamics, wing frame setup).

Extracted from the original ``visualizer/model_utils.py`` so that the generic
``mujoco_visualizer`` package has no fly knowledge.  Only ``apply_flight_setup``
and its dependencies live here.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants copied verbatim from fly_mimic/envs/fruitfly/constants.py so this
# module has no runtime dependency on fly_mimic.
# ---------------------------------------------------------------------------
_AIR_DENSITY = 0.00128            # mg/cm^3
_AIR_VISCOSITY = 0.000185         # mg/(cm*s)
_FLY_PHYSICS_TIMESTEP = 5e-5      # s
_BODY_PITCH_ANGLE = 55            # deg
_WING_PARAMS = {
    'stiffness': 0.01,
    'damping': 0.007769230,
    'gainprm': 18,
    'ellipsoid_fluidcoef': [1.0, 0.5, 1.5, 1.7, 1.0],
    'quasi_fluidcoef':     [0.00128, 3.3207, 0.4123, 3.1670, 1],
    'default_qpos': [1.5, 0.7, -0.85, 1.5, 0.7, -0.85],
}


def _quat_mul(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def _neg_quat(q):
    q = np.array(q, dtype=float).copy()
    q[0] = -q[0]
    return q


def _quat_rotate(v, q):
    v = np.array(v, dtype=float)
    q = np.array(q, dtype=float)
    w, x, y, z = q
    u = np.array([x, y, z])
    return (2.0 * np.dot(u, v) * u
            + (w * w - np.dot(u, u)) * v
            + 2.0 * w * np.cross(u, v))


def change_body_frame(body, frame_pos, frame_quat):
    frame_pos = np.zeros(3) if frame_pos is None else np.array(frame_pos, dtype=float)
    frame_quat = np.array([1., 0, 0, 0]) if frame_quat is None else np.array(frame_quat, dtype=float)
    body_pos = np.zeros(3) if body.pos is None else np.array(body.pos, dtype=float)
    dpos = body_pos - frame_pos
    body_quat = np.array([1., 0, 0, 0]) if body.quat is None else np.array(body.quat, dtype=float)
    dquat = _quat_mul(_neg_quat(frame_quat), body_quat)
    body.pos = frame_pos
    body.quat = frame_quat
    for child in body.find_all('body'):
        if not hasattr(child, 'pos'):
            continue
        if hasattr(child, 'quat'):
            child_quat = np.array([1., 0, 0, 0]) if child.quat is None else np.array(child.quat, dtype=float)
            child.quat = _quat_mul(dquat, child_quat)
        child_pos = np.zeros(3) if child.pos is None else np.array(child.pos, dtype=float)
        pos_in_parent = _quat_rotate(child_pos, body_quat) + dpos
        child.pos = _quat_rotate(pos_in_parent, _neg_quat(frame_quat))


def change_wing_frame(body, new_quat):
    old_quat = np.array(body.quat, dtype=float) if body.quat is not None else np.array([1., 0, 0, 0])
    new_quat = np.array(new_quat, dtype=float)
    dquat = _quat_mul(_neg_quat(new_quat), old_quat)
    body.quat = new_quat
    for geom in body.find_all('geom'):
        gq = np.array(geom.quat, dtype=float) if geom.quat is not None else np.array([1., 0, 0, 0])
        geom.quat = _quat_mul(dquat, gq)
        gp = np.array(geom.pos, dtype=float) if geom.pos is not None else np.zeros(3)
        world_pos = _quat_rotate(gp, old_quat)
        geom.pos = _quat_rotate(world_pos, _neg_quat(new_quat))
    for site in body.find_all('site'):
        sq = np.array(site.quat, dtype=float) if site.quat is not None else np.array([1., 0, 0, 0])
        site.quat = _quat_mul(dquat, sq)
        sp = np.array(site.pos, dtype=float) if site.pos is not None else np.zeros(3)
        world_pos = _quat_rotate(sp, old_quat)
        site.pos = _quat_rotate(world_pos, _neg_quat(new_quat))


def _reset_wing_orientation(spec, suffix='', body_pitch_angle=_BODY_PITCH_ANGLE):
    site_name = 'hover_up_dir' + suffix
    up_dir = np.array(spec.site(site_name).quat, dtype=float).copy()
    up_dir_angle = 2 * np.arccos(np.clip(up_dir[0], -1.0, 1.0))
    delta = np.deg2rad(body_pitch_angle) - up_dir_angle
    dquat = np.array([np.cos(delta / 2), 0., np.sin(delta / 2), 0.])
    up_dir = _quat_mul(dquat, up_dir)

    stroke_plane_angle = np.deg2rad(body_pitch_angle)
    stroke_plane_quat = np.array([np.cos(stroke_plane_angle / 2), 0.,
                                  np.sin(stroke_plane_angle / 2), 0.])
    wing_body_names = ['wing_left' + suffix, 'wing_right' + suffix]
    for quat, wing in [(np.array([0., 0, 0, 1.]), wing_body_names[0]),
                       (np.array([0., -1, 0, 0.]), wing_body_names[1])]:
        dq = _quat_mul(_neg_quat(stroke_plane_quat), quat)
        new_wing_quat = _quat_mul(dq, _neg_quat(up_dir))
        body = spec.body(wing)
        change_body_frame(body, body.pos, new_wing_quat)
    return spec


def apply_flight_setup(spec, *, quasi_aero=False, wbpg=False, suffix=''):
    """Apply flight aerodynamics / physics / wing-frame setup to an MjSpec.

    Numpy port of ``Fruitfly._set_up_flight`` from
    ``fly_mimic/envs/fruitfly/base.py`` (lines 807-840).
    """
    coefs = _WING_PARAMS['quasi_fluidcoef' if quasi_aero else 'ellipsoid_fluidcoef']
    for geom in spec.geoms:
        if geom.name and 'fluid' in geom.name:
            geom.fluid_coefs = coefs

    spec.option.density = _AIR_DENSITY
    spec.option.viscosity = _AIR_VISCOSITY
    spec.option.timestep = _FLY_PHYSICS_TIMESTEP

    wing_joint_names = [j.name for j in spec.joints
                        if j.name and ('wing_left' in j.name or 'wing_right' in j.name)]
    for jname in wing_joint_names:
        spec.joint(jname).stiffness = _WING_PARAMS['stiffness']
        spec.joint(jname).damping = _WING_PARAMS['damping']
        try:
            spec.actuator(jname).gainprm[0] = _WING_PARAMS['gainprm']
        except (KeyError, ValueError):
            pass

    if not wbpg:
        spec = _reset_wing_orientation(spec, suffix=suffix, body_pitch_angle=47.5)
        beta = np.deg2rad(_BODY_PITCH_ANGLE)
        q_srf = np.array([np.cos(beta / 2), 0., -np.sin(beta / 2), 0.])
        for w in ('wing_left' + suffix, 'wing_right' + suffix):
            change_wing_frame(spec.body(w), q_srf)
    else:
        spec = _reset_wing_orientation(spec, suffix=suffix, body_pitch_angle=_BODY_PITCH_ANGLE)
    return spec
