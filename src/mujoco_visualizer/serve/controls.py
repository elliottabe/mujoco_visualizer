"""Group a model's actuators into a tree the browser UI can render generically.

The fly model has 272 actuators (258 muscle tendons, 6 joint, 8 adhesion), which is far too
many for a flat slider list. Names carry the structure -- a T1/T2/T3 segment token and a
left/right token -- so the grouping is pure name introspection and stays model-agnostic:
a model without those tokens simply lands everything in ``other.*``.
"""

from typing import Dict, List, Optional

import mujoco

_SEGMENTS = ("T1", "T2", "T3")
_SIDES = ("left", "right")

# Display order. Group ids absent from the model are dropped, so this is a superset.
_ORDER = (
    [f"leg.{seg}.{side}" for seg in _SEGMENTS for side in _SIDES]
    + [f"wing.{side}" for side in _SIDES]
    + ["wing.none", "abdomen"]
    + [f"other.{side}" for side in _SIDES]
    + ["other.none"]
)

_LABELS = {"none": "", "left": "left", "right": "right"}


def _side_of(name: str) -> str:
    for side in _SIDES:
        if side in name:
            return side
    return "none"


def _group_of(name: str) -> str:
    """Group id for one actuator name. Order matters: the wing check precedes the segment
    check because wing actuators carry no T-token, and abdomen precedes ``other`` because
    ``abdomen`` actuators are unsided."""
    lowered = name.lower()
    side = _side_of(name)
    if "wing" in lowered:
        return f"wing.{side}"
    for seg in _SEGMENTS:
        if seg in name:
            return f"leg.{seg}.{side}"
    if "abdomen" in lowered:
        return "abdomen"
    return f"other.{side}"


def _label_of(group_id: str) -> str:
    parts = group_id.split(".")
    if parts[0] == "leg":
        return f"{parts[1]} {_LABELS[parts[2]]}".strip()
    if parts[0] == "abdomen":
        return "Abdomen"
    head = parts[0].capitalize()
    tail = _LABELS.get(parts[1], "") if len(parts) > 1 else ""
    return f"{head} {tail}".strip()


def build_control_tree(model: mujoco.MjModel) -> Dict:
    """Grouped actuator tree for the client UI.

    Returns ``{"groups": [{"id", "label", "actuators": [{"id", "name", "lo", "hi",
    "limited"}]}]}``. Every actuator appears in exactly one group; empty groups are omitted.
    Unnamed actuators get a synthetic ``act<i>`` name so the client always has a stable key.
    """
    buckets: Dict[str, List[Dict]] = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
        limited = bool(model.actuator_ctrllimited[i])
        lo, hi = (float(x) for x in model.actuator_ctrlrange[i])
        if not limited:
            lo, hi = -1.0, 1.0
        buckets.setdefault(_group_of(name), []).append(
            {"id": i, "name": name, "lo": lo, "hi": hi, "limited": limited}
        )

    ordered = [g for g in _ORDER if g in buckets]
    ordered += sorted(g for g in buckets if g not in _ORDER)
    return {
        "groups": [
            {"id": g, "label": _label_of(g), "actuators": buckets[g]} for g in ordered
        ]
    }


def actuator_group_map(tree: Dict) -> Dict[int, str]:
    """Actuator id -> group id, inverting :func:`build_control_tree`."""
    return {a["id"]: g["id"] for g in tree["groups"] for a in g["actuators"]}
