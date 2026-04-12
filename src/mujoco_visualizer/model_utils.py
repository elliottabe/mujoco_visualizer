"""model_utils.py — Generic MuJoCo model filtering / amputation utilities.

``filter_model_to_config_joints`` removes joints, actuators, tendons, sensors,
geoms, and sites from an ``MjSpec`` so that only elements tied to a chosen set
of joint names are retained. Uses structural references (joint/tendon targets,
sensor object names, site body ownership) rather than name-substring heuristics,
so muscle actuators and site-based sensors are handled correctly.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import List, Optional, Union

import mujoco
import numpy as np

# qpos count per joint type: FREE=0 (7), BALL=1 (4), SLIDE=2 (1), HINGE=3 (1)
_JOINT_NQPOS = {0: 7, 1: 4, 2: 1, 3: 1}


def filter_model_to_config_joints(
    joint_names: List[str],
    spec: Optional[mujoco.MjSpec] = None,
    xml_path: Optional[str] = None,
    amputate: Union[bool, str, List[str]] = False,
    verbose: bool = True,
) -> mujoco.MjSpec:
    """Remove joints, actuators, tendons, and sensors from *spec* so that only
    elements tied to ``joint_names`` are retained.

    Uses structural references (joint/tendon targets, sensor object names, site
    body ownership) rather than name-substring heuristics, so muscle actuators
    and site-based sensors are handled correctly.  Keyframe qpos arrays are
    trimmed to match the reduced joint set.

    Args:
        joint_names:        Names of joints to keep.
        spec:               MjSpec to filter in-place.  Loaded from ``xml_path``
                            if None.
        xml_path:           Path to a MuJoCo XML file; used when ``spec`` is None.
        amputate:           Controls geom/site removal for bodies whose joints are
                            being removed.

                            * ``False`` (default): no geometry removed.
                            * ``True``: remove geoms and sites from *all* bodies in
                              the removed set (includes non-locomotion bodies such as
                              head/antenna if their joints are not in joint_names).
                            * ``str`` or ``List[str]``: body-name substrings — only
                              bodies whose name contains at least one pattern have
                              their geoms and sites removed.
                              E.g. ``amputate='T1_right'`` to amputate the right
                              front leg only.
        verbose:            Print a summary of what was removed.

    Returns:
        Filtered MjSpec.
    """
    if spec is None:
        if xml_path is None:
            raise ValueError("Either spec or xml_path must be provided.")
        spec = mujoco.MjSpec.from_file(xml_path)

    allowed_joints = set(joint_names)

    # --- Build removed_bodies: bodies where ALL non-free joints are being removed ---
    removed_bodies: set[str] = set()
    for body in spec.bodies:
        non_free = [j.name for j in body.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
        if non_free and not any(j in allowed_joints for j in non_free):
            removed_bodies.add(body.name)

    # --- Resolve which removed bodies to amputate (geom + site removal) ---
    if amputate is False:
        amputate_body_set: set[str] = set()
    elif amputate is True:
        amputate_body_set = set(removed_bodies)
    else:
        patterns = [amputate] if isinstance(amputate, str) else list(amputate)
        amputate_body_set = {b for b in removed_bodies if any(p in b for p in patterns)}

    # --- Build site -> body map for site-based sensor resolution ---
    site_to_body: dict[str, str] = {}
    for body in spec.bodies:
        for site in body.sites:
            site_to_body[site.name] = body.name

    # --- Parse tendon joint/body references from exported XML ---
    # tendon.wraps is not iterable in the Python MjSpec API, so we parse XML.
    tendon_joint_refs: dict[str, set] = {}   # fixed tendon name -> set of joint names
    tendon_ins_bodies: dict[str, set] = {}   # spatial tendon -> insertion body names
    tendon_all_bodies: dict[str, set] = {}   # spatial tendon -> ALL referenced body names
    xml_root = ET.fromstring(spec.to_xml())
    tendon_el = xml_root.find('tendon')
    if tendon_el is not None:
        for child in tendon_el:
            name = child.get('name', '')
            if child.tag == 'fixed':
                tendon_joint_refs[name] = {
                    sub.get('joint') for sub in child.findall('joint') if sub.get('joint')
                }
            elif child.tag == 'spatial':
                # Site naming convention: "{muscle}^{type}^{body_name}"
                sites = [sub.get('site', '') for sub in child.findall('site')]
                tendon_all_bodies[name] = {s.split('^')[-1] for s in sites if '^' in s}
                ins_bodies = {s.split('^')[-1] for s in sites if '^ins^' in s}
                if not ins_bodies:
                    ins_bodies = tendon_all_bodies[name]
                tendon_ins_bodies[name] = ins_bodies

    # --- Collect elements to delete ---
    joints_to_delete = [
        j for body in spec.bodies for j in body.joints
        if j.name not in allowed_joints and j.type != mujoco.mjtJoint.mjJNT_FREE
    ]

    tendons_to_delete = []
    tendons_to_delete_names: set[str] = set()
    for tendon in spec.tendons:
        name = tendon.name
        if name in tendon_joint_refs:
            keep = bool(tendon_joint_refs[name] & allowed_joints)
        elif name in tendon_ins_bodies:
            no_active_insertion = not bool(tendon_ins_bodies[name] - removed_bodies)
            touches_amputated = bool(tendon_all_bodies.get(name, set()) & amputate_body_set)
            keep = not (no_active_insertion or touches_amputated)
        else:
            keep = any(jname in name or name in jname for jname in allowed_joints)
        if not keep:
            tendons_to_delete.append(tendon)
            tendons_to_delete_names.add(name)

    actuators_to_delete = []
    for actuator in spec.actuators:
        trntype = actuator.trntype
        target = actuator.target
        if trntype == mujoco.mjtTrn.mjTRN_JOINT:
            keep = target in allowed_joints
        elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
            keep = target not in tendons_to_delete_names
        else:
            keep = True
        if not keep:
            actuators_to_delete.append(actuator)

    sensors_to_delete = []
    for sensor in spec.sensors:
        objtype = sensor.objtype
        objname = sensor.objname
        if objtype == mujoco.mjtObj.mjOBJ_JOINT:
            keep = objname in allowed_joints
        elif objtype == mujoco.mjtObj.mjOBJ_SITE:
            body = site_to_body.get(objname)
            keep = body is None or body not in removed_bodies
        elif objtype == mujoco.mjtObj.mjOBJ_BODY:
            keep = objname not in removed_bodies
        else:
            keep = True
        if not keep:
            sensors_to_delete.append(sensor)

    if verbose:
        n_joints_total = sum(len(list(b.joints)) for b in spec.bodies)
        print(f"filter_model_to_config_joints: keeping {len(allowed_joints)} joint names")
        print(f"  removed_bodies ({len(removed_bodies)}): "
              f"{sorted(removed_bodies)[:6]}{'...' if len(removed_bodies) > 6 else ''}")
        if amputate_body_set:
            print(f"  amputate_bodies ({len(amputate_body_set)}): "
                  f"{sorted(amputate_body_set)[:6]}{'...' if len(amputate_body_set) > 6 else ''}")
        print(f"  Joints:    removing {len(joints_to_delete)} of {n_joints_total}")
        print(f"  Actuators: removing {len(actuators_to_delete)} of {len(list(spec.actuators))}")
        print(f"  Tendons:   removing {len(tendons_to_delete)} of {len(list(spec.tendons))}")
        print(f"  Sensors:   removing {len(sensors_to_delete)} of {len(list(spec.sensors))}")

    # --- Trim keyframe qpos BEFORE any deletions ---
    joints_to_delete_names = {j.name for j in joints_to_delete}
    try:
        orig_model = spec.compile()
        kept_qpos_idx = []
        for i in range(orig_model.njnt):
            jtype = int(orig_model.jnt_type[i])
            adr = int(orig_model.jnt_qposadr[i])
            nqpos = _JOINT_NQPOS.get(jtype, 1)
            jname = orig_model.joint(i).name
            if jtype == 0 or jname not in joints_to_delete_names:
                kept_qpos_idx.extend(range(adr, adr + nqpos))
        for key in spec.keys:
            if len(key.qpos) > 0:
                key.qpos = [key.qpos[i] for i in kept_qpos_idx]
        if verbose:
            print(f"  Keyframes: trimmed qpos {orig_model.nq} -> {len(kept_qpos_idx)}")
    except Exception as e:
        if verbose:
            print(f"  Warning: could not trim keyframe qpos: {e}")

    # --- Delete in safe order: structural elements before geoms/sites ---
    for joint in joints_to_delete:
        spec.delete(joint)
    for actuator in actuators_to_delete:
        spec.delete(actuator)
    for tendon in tendons_to_delete:
        spec.delete(tendon)
    for sensor in sensors_to_delete:
        spec.delete(sensor)

    # --- Geoms + sites last (after all referencing tendons/sensors are gone) ---
    if amputate_body_set:
        geoms_to_delete = [
            geom
            for body in spec.bodies if body.name in amputate_body_set
            for geom in list(body.geoms)
        ]
        sites_to_delete = [
            site
            for body in spec.bodies if body.name in amputate_body_set
            for site in list(body.sites)
        ]
        if verbose:
            print(f"  Geoms:     removing {len(geoms_to_delete)}")
            print(f"  Sites:     removing {len(sites_to_delete)}")
        for geom in geoms_to_delete:
            spec.delete(geom)
        for site in sites_to_delete:
            spec.delete(site)

    return spec


