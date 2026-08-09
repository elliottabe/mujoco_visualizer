"""Freeze chosen degrees of freedom in a qpos frame.

Model-agnostic: the joint layout comes from ``jnt_type``/``jnt_qposadr``, so this works for
any MuJoCo model and knows nothing about the caller's anatomy.

``apply_locks`` ALWAYS returns a copy, even with no locks. Its callers hand it frames that
came from a frozen, process-wide store (the replay source freezes its array and returns a
copy per call); writing in place would corrupt that store for every later frame and for the
render and export threads both. An unconditional copy also keeps the contract uniform, so no
caller has to reason about when the return aliases its input.

A free joint is exposed as TWO locks, position and orientation, rather than one: pinning
translation while letting the body rotate is a real thing to want, and a quaternion has no
meaningful component-wise lock.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

__all__ = ["ROOT_POS", "ROOT_QUAT", "build_joint_qpos_map", "apply_locks",
           "resolve_lock_values", "pair_with_suffix"]

ROOT_POS = "<root>.pos"
ROOT_QUAT = "<root>.quat"

_WIDTH = {
    int(mujoco.mjtJoint.mjJNT_FREE): 7,
    int(mujoco.mjtJoint.mjJNT_BALL): 4,
    int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
    int(mujoco.mjtJoint.mjJNT_HINGE): 1,
}


def build_joint_qpos_map(model) -> Dict[str, Tuple[int, int]]:
    """``{joint name: (qpos address, width)}`` for every joint in *model*."""
    out: Dict[str, Tuple[int, int]] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint{i}"
        jtype = int(model.jnt_type[i])
        adr = int(model.jnt_qposadr[i])
        if jtype == int(mujoco.mjtJoint.mjJNT_FREE):
            out[f"{name}{ROOT_POS}"] = (adr, 3)
            out[f"{name}{ROOT_QUAT}"] = (adr + 3, 4)
        else:
            out[name] = (adr, _WIDTH[jtype])
    return out


def apply_locks(qpos, locks: Dict[str, Sequence[float]], jmap) -> np.ndarray:
    """Return a copy of *qpos* with each locked joint overwritten by its held value."""
    out = np.array(qpos, dtype=np.float64, copy=True)
    for name, value in locks.items():
        if name not in jmap:
            raise KeyError(
                f"no joint {name!r} in this model; known joints include "
                f"{', '.join(sorted(jmap)[:4])}…"
            )
        adr, width = jmap[name]
        vals = np.asarray(value, dtype=np.float64).ravel()
        if vals.size != width:
            raise ValueError(f"joint {name!r} expects {width} value(s), got {vals.size}")
        if not np.isfinite(vals).all():
            raise ValueError(f"joint {name!r}: lock values must be finite, got {value!r}")
        out[adr:adr + width] = vals
    return out


def resolve_lock_values(qpos, names: Iterable[str], jmap) -> Dict[str, List[float]]:
    """Sample *names* out of *qpos* — the freeze-at-engage values."""
    arr = np.asarray(qpos, dtype=np.float64)
    resolved = {}
    for name in names:
        if name not in jmap:
            raise KeyError(f"no joint {name!r} in this model")
        adr, width = jmap[name]
        resolved[name] = [float(x) for x in arr[adr:adr + width]]
    return resolved


def pair_with_suffix(names: Iterable[str], jmap, suffix: Optional[str]) -> List[str]:
    """Each name, plus its ``name + suffix`` counterpart when the model has one.

    A doubled model (policy fly + suffixed reference copy) must lock both halves together:
    freezing one half while the other keeps moving renders a tracking error that does not
    exist.
    """
    out: List[str] = []
    for name in names:
        out.append(name)
        if suffix:
            paired = f"{name}{suffix}"
            if paired in jmap and paired not in out:
                out.append(paired)
    return out
