"""Joint locking: the map, and applying it to a COPY.

The copy discipline is the load-bearing part. Callers hand this function a frame that came
from a frozen, shared store; if apply_locks wrote in place it would corrupt the store for
every later frame and for both other threads.
"""
import mujoco
import numpy as np
import pytest

from mujoco_visualizer.serve.locks import (
    ROOT_POS, ROOT_QUAT, apply_locks, build_joint_qpos_map, pair_with_suffix,
    resolve_lock_values,
)

_XML = """
<mujoco><worldbody>
  <body name="root"><freejoint name="free"/><geom type="box" size=".1 .1 .1"/>
    <body name="a"><joint name="hinge_a" type="hinge" axis="0 0 1"/>
      <geom type="box" size=".05 .05 .05"/>
      <body name="b"><joint name="slide_b" type="slide" axis="1 0 0"/>
        <geom type="box" size=".05 .05 .05"/>
        <body name="c"><joint name="ball_c" type="ball"/>
          <geom type="box" size=".05 .05 .05"/></body></body></body></body>
</worldbody></mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


def test_map_covers_all_four_joint_widths(model):
    jmap = build_joint_qpos_map(model)
    assert jmap[f"free{ROOT_POS}"] == (0, 3)
    assert jmap[f"free{ROOT_QUAT}"] == (3, 4)
    assert jmap["hinge_a"] == (7, 1)
    assert jmap["slide_b"] == (8, 1)
    assert jmap["ball_c"] == (9, 4)
    assert "free" not in jmap, "a free joint is exposed as its two halves, never as one lock"


def test_map_widths_sum_to_nq(model):
    jmap = build_joint_qpos_map(model)
    assert sum(w for _, w in jmap.values()) == model.nq


def test_apply_locks_returns_a_copy_and_leaves_the_input_untouched(model):
    jmap = build_joint_qpos_map(model)
    q = np.arange(model.nq, dtype=np.float64)
    original = q.copy()
    out = apply_locks(q, {"hinge_a": [9.0]}, jmap)
    assert out is not q
    np.testing.assert_array_equal(q, original), "the caller's array must be untouched"
    assert out[7] == 9.0
    assert out[8] == original[8], "neighbours must not move"


def test_apply_locks_writes_every_component_of_a_wide_lock(model):
    jmap = build_joint_qpos_map(model)
    q = np.zeros(model.nq)
    out = apply_locks(q, {f"free{ROOT_QUAT}": [1.0, 0.0, 0.0, 0.0]}, jmap)
    np.testing.assert_array_equal(out[3:7], [1.0, 0.0, 0.0, 0.0])


def test_apply_locks_rejects_a_width_mismatch(model):
    jmap = build_joint_qpos_map(model)
    with pytest.raises(ValueError, match="expects 4"):
        apply_locks(np.zeros(model.nq), {f"free{ROOT_QUAT}": [1.0]}, jmap)


def test_apply_locks_rejects_an_unknown_joint(model):
    jmap = build_joint_qpos_map(model)
    with pytest.raises(KeyError, match="nope"):
        apply_locks(np.zeros(model.nq), {"nope": [0.0]}, jmap)


def test_apply_locks_rejects_non_finite_values(model):
    jmap = build_joint_qpos_map(model)
    with pytest.raises(ValueError, match="finite"):
        apply_locks(np.zeros(model.nq), {"hinge_a": [float("nan")]}, jmap)


def test_no_locks_is_still_a_copy(model):
    q = np.zeros(model.nq)
    out = apply_locks(q, {}, build_joint_qpos_map(model))
    assert out is not q, "an unconditional copy keeps the caller's contract uniform"


def test_resolve_lock_values_samples_the_frame_given(model):
    jmap = build_joint_qpos_map(model)
    q = np.arange(model.nq, dtype=np.float64)
    assert resolve_lock_values(q, ["hinge_a"], jmap) == {"hinge_a": [7.0]}
    assert resolve_lock_values(q, [f"free{ROOT_POS}"], jmap) == {f"free{ROOT_POS}": [0.0, 1.0, 2.0]}


def test_pair_with_suffix_adds_the_counterpart_only_when_it_exists(model):
    jmap = dict(build_joint_qpos_map(model))
    jmap["hinge_a-ghost"] = (99, 1)
    assert pair_with_suffix(["hinge_a"], jmap, "-ghost") == ["hinge_a", "hinge_a-ghost"]
    assert pair_with_suffix(["slide_b"], jmap, "-ghost") == ["slide_b"]
    assert pair_with_suffix(["hinge_a"], jmap, None) == ["hinge_a"]
