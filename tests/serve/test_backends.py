"""PhysicsBackend seam: CpuBackend over plain mj_step, tested with no GL and no JAX.

The model below carries two keyframes on purpose -- one unnamed at index 0, one named
``'default_pose'`` at index 1 -- to pin that ``reset_to_keyframe`` resolves BY NAME, never by
index. This mirrors the composed fly+floor model's real keyframe list
``[None, 'default_pose']``, where index-based reset silently picks the wrong pose.
"""

import mujoco
import numpy as np
import pytest

from mujoco_visualizer.serve.backends import CpuBackend, PhysicsBackend, UnknownKeyframe

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="b" pos="0 0 0.5">
      <joint name="j" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m" joint="j" ctrlrange="-1 1"/>
  </actuator>
  <keyframe>
    <key qpos="0.0"/>
    <key name="default_pose" qpos="0.7"/>
  </keyframe>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


@pytest.fixture
def data(model):
    return mujoco.MjData(model)


@pytest.fixture
def backend(model, data):
    return CpuBackend(model, data)


def test_cpu_backend_satisfies_the_protocol(backend):
    assert isinstance(backend, PhysicsBackend)


def test_label_is_cpu(backend):
    assert backend.label == "cpu"


def test_warning_is_a_non_empty_string(backend):
    """The CPU path drives tendons linearly, not with the trained muscle mechanics -- a
    viewer that hides that fact would be the worst possible outcome."""
    assert isinstance(backend.warning, str)
    assert backend.warning != ""


def test_set_ctrl_writes_data_ctrl(backend, data):
    backend.set_ctrl(np.array([0.5]))
    assert data.ctrl[0] == pytest.approx(0.5)


def test_step_advances_model_time(backend, data, model):
    backend.step(10)
    assert backend.time == pytest.approx(10 * model.opt.timestep)
    assert data.time == pytest.approx(10 * model.opt.timestep)


def test_sync_to_is_a_noop_for_a_shared_mjdata(backend, data, model):
    """CpuBackend already steps the caller's own MjData in place, so sync_to must not touch
    an unrelated MjData passed to it -- there is nothing to copy."""
    other = mujoco.MjData(model)
    other.qpos[:] = 123.0
    backend.sync_to(other)
    assert other.qpos[0] == pytest.approx(123.0)


def test_reset_to_keyframe_resolves_by_name_not_index(backend, data):
    """Index 0 is the unnamed keyframe (qpos 0.0); 'default_pose' is index 1 (qpos 0.7).
    A by-index reset would silently pick the wrong one."""
    backend.reset_to_keyframe("default_pose")
    assert data.qpos[0] == pytest.approx(0.7)


def test_reset_to_keyframe_missing_name_raises_clearly(backend):
    with pytest.raises(UnknownKeyframe):
        backend.reset_to_keyframe("no_such_keyframe")


def test_close_does_not_raise(backend):
    backend.close()
