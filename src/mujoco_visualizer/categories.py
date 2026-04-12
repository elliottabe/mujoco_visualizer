"""categories.py — Build geom→category buckets from an AnatomyConfig.

Generic replacement for the fly-specific ``_body_to_cat`` /
``_build_geom_categories`` functions in the original visualizer.
"""
from __future__ import annotations

from typing import Dict, List

import mujoco

from mujoco_visualizer.config import AnatomyConfig, CategoryRule


def _auto_anatomy(model: mujoco.MjModel) -> AnatomyConfig:
    """Derive a trivial AnatomyConfig from the model: one category per top-level
    body (i.e. each direct child of the worldbody and all its descendants).
    """
    cats: List[CategoryRule] = []
    seen: set = set()
    for bid in range(1, model.nbody):
        if int(model.body_parentid[bid]) != 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if not name or name in seen:
            continue
        seen.add(name)
        cats.append(CategoryRule(name=name, body_substring=[name]))
    cameras = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"cam{i}"
        for i in range(model.ncam)
    ]
    return AnatomyConfig(cameras=cameras, categories=cats)


def build_geom_categories(
    model: mujoco.MjModel,
    anatomy: AnatomyConfig,
) -> Dict[str, List[int]]:
    """Bucket geom ids by category name. Each geom is assigned to the FIRST
    matching rule in ``anatomy.categories`` (order matters)."""
    out: Dict[str, List[int]] = {c.name: [] for c in anatomy.categories}
    if not anatomy.categories:
        return out
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        for rule in anatomy.categories:
            if rule.matches(body_name, geom_name):
                out[rule.name].append(gid)
                break
    return out
