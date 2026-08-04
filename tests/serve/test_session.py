"""Session owns model+data+renderer, stepped through a swappable physics backend. Everything
here runs headless; only the render and encode tests need GL.

Session no longer calls ``mj_step`` directly -- it delegates to a ``backend`` (default
``CpuBackend``, sharing Session's own ``MjData``) and syncs from it each step. A ``FakeBackend``
below proves that seam is swappable without a real physics engine (or JAX) in the loop.
"""

import mujoco
import numpy as np
import pytest

from mujoco_visualizer.serve.backends import CpuBackend
from mujoco_visualizer.serve.session import Diverged, Session

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j_coxa_T1_left" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      <body name="b1" pos="0.1 0 0">
        <joint name="j_coxa_T1_right" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="coxa_T1_left"  joint="j_coxa_T1_left"  ctrlrange="-1 1"/>
    <motor name="coxa_T1_right" joint="j_coxa_T1_right" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def xml(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_XML)
    return str(p)


@pytest.fixture
def sess(xml):
    s = Session(xml_path=xml, width=64, height=64)
    yield s
    s.close()


class FakeController:
    """Returns a constant ctrl so additive composition is checkable by hand."""

    rate_hz = 100.0

    def __init__(self, nu, value=0.25):
        self._out = np.full(nu, value, dtype=np.float64)
        self.calls = 0

    def step(self, model, data):
        self.calls += 1
        return self._out

    def readout(self):
        return {"calls": self.calls}


class FakeBackend:
    """Minimal stand-in for :class:`PhysicsBackend` -- no mj_step, no MjData of its own.

    Proves Session's stepping surface is swappable: ``step`` records what it was asked to do
    and ``sync_to`` writes a deterministic qpos into whatever MjData Session hands it, so a
    test can check Session actually delegated instead of quietly stepping on its own.
    """

    label = "fake"
    warning = None

    def __init__(self, nu, nq):
        self._t = 0.0
        self._qpos = np.zeros(nq)
        self.last_ctrl = np.zeros(nu)
        self.step_calls = []
        self.sync_calls = 0
        self.reset_calls = []

    def set_ctrl(self, ctrl):
        self.last_ctrl = np.asarray(ctrl, dtype=np.float64).copy()

    def step(self, n):
        self.step_calls.append(int(n))
        self._t += int(n) * 0.5
        self._qpos[:] = self._t  # deterministic, checkable marker

    def sync_to(self, data):
        self.sync_calls += 1
        data.qpos[: len(self._qpos)] = self._qpos

    def reset_to_keyframe(self, name):
        self.reset_calls.append(name)
        self._t = 0.0
        self._qpos[:] = 0.0

    @property
    def time(self):
        return self._t

    def close(self):
        self.close_calls = getattr(self, "close_calls", 0) + 1


def test_step_advances_sim_time(sess):
    t0 = sess.data.time
    sess.step(10)
    assert sess.data.time == pytest.approx(t0 + 10 * sess.model.opt.timestep)


def test_absolute_mode_writes_ctrl_directly(sess):
    sess.set_ctrl({"coxa_T1_left": 0.4})
    sess.step(1)
    assert sess.data.ctrl[0] == pytest.approx(0.4)


def test_additive_mode_adds_to_controller_output(xml):
    s = Session(xml_path=xml, width=64, height=64)
    try:
        s.attach_controller(FakeController(s.model.nu, 0.25))
        s.set_ctrl_mode("additive")
        s.set_ctrl({"coxa_T1_left": 0.1})
        s.advance_controller()  # SimLoop calls this at rate_hz; step() itself never does
        s.step(1)
        assert s.data.ctrl[0] == pytest.approx(0.35)
        assert s.data.ctrl[1] == pytest.approx(0.25)
    finally:
        s.close()


def test_ctrl_is_clamped_to_ctrlrange(sess):
    sess.set_ctrl({"coxa_T1_left": 5.0, "coxa_T1_right": -9.0})
    sess.step(1)
    assert sess.data.ctrl[0] == pytest.approx(1.0)
    assert sess.data.ctrl[1] == pytest.approx(-2.0)


def test_group_gain_scales_its_members(sess):
    sess.set_ctrl({"coxa_T1_left": 0.8})
    sess.set_group_gain("leg.T1.left", 0.5)
    sess.step(1)
    assert sess.data.ctrl[0] == pytest.approx(0.4)


def test_unknown_actuator_name_is_rejected(sess):
    with pytest.raises(KeyError):
        sess.set_ctrl({"no_such_actuator": 0.1})


def test_divergence_is_detected_and_state_restored(sess):
    sess.step(1)
    good = sess.data.qpos.copy()
    sess.data.qvel[0] = np.nan  # inject the blow-up
    with pytest.raises(Diverged):
        sess.step(1)
    assert np.isfinite(sess.data.qpos).all()
    np.testing.assert_allclose(sess.data.qpos, good)


def test_divergence_is_detected_via_fatal_warning_counter_not_isfinite_alone(sess):
    """MuJoCo's own 'Nan, Inf or huge value' check repairs the bad DOF in place before
    mj_step returns, so qpos/qvel end up finite again even though a real blow-up happened --
    an isfinite-only check would miss this. This pins that detection actually goes through
    the mjWARN_BADQVEL/mjWARN_BADQACC counter, by checking the counter itself moved."""
    sess.step(1)
    before = (
        sess.data.warning[mujoco.mjtWarning.mjWARN_BADQVEL].number
        + sess.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    )
    sess.data.qvel[0] = np.nan  # inject the blow-up
    with pytest.raises(Diverged):
        sess.step(1)
    after = (
        sess.data.warning[mujoco.mjtWarning.mjWARN_BADQVEL].number
        + sess.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number
    )
    assert after > before, "expected a fatal warning counter to increase during the step"
    # And confirm the isfinite state alone would NOT have caught this: MuJoCo already
    # repaired it back to finite by the time step() looks, which is exactly why the counter
    # (not isfinite) has to be the primary signal.
    assert np.isfinite(sess.data.qpos).all()


def test_capacity_warning_is_not_treated_as_divergence(sess):
    """A full contact buffer (mjWARN_CONTACTFULL) is a capacity/quality warning, not state
    corruption -- step() must keep running (and keep surfacing it via warnings()), not pause
    the viewer the way a fatal BADQVEL/BADQPOS/BADQACC/BADCTRL warning does."""
    sess.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number += 1
    sess.step(1)  # must not raise Diverged
    assert "CONTACTFULL" in sess.warnings()


def test_reset_restores_initial_state(sess):
    """The fixture model has no 'default_pose' keyframe, so reset() must fall back to
    mj_resetData rather than raising."""
    sess.set_ctrl({"coxa_T1_left": 1.0})
    sess.step(50)
    assert sess.data.time > 0
    sess.reset()
    assert sess.data.time == pytest.approx(0.0)
    np.testing.assert_allclose(sess.data.qpos, sess.model.qpos0)


def test_set_qpos_writes_state_without_stepping(sess):
    target = sess.model.qpos0.copy()
    target[0] = 0.3
    sess.set_qpos(target)
    assert sess.data.qpos[0] == pytest.approx(0.3)
    assert sess.data.time == pytest.approx(0.0)


def test_set_qpos_rejects_non_finite_input_and_does_not_poison_the_snapshot(sess):
    """A non-finite qpos (a malformed replay-scrub, a bad client message) must be rejected
    outright rather than written and snapshotted -- accepting it would make _good the bad
    state, and every later step() would restore that poisoned snapshot and raise Diverged
    forever until reset()."""
    sess.step(1)
    good_qpos = sess.data.qpos.copy()
    bad = good_qpos.copy()
    bad[0] = np.nan

    with pytest.raises(ValueError):
        sess.set_qpos(bad)

    # Rejected before it touched data or the rollback snapshot.
    np.testing.assert_allclose(sess.data.qpos, good_qpos)
    # And stepping still works normally afterwards -- no poisoned snapshot to restore into.
    sess.step(1)
    assert np.isfinite(sess.data.qpos).all()


def test_scene_message_describes_the_model(sess):
    msg = sess.scene_message()
    assert msg["nu"] == sess.model.nu
    assert msg["nq"] == sess.model.nq
    assert msg["has_controller"] is False
    ids = [a["id"] for g in msg["controls"]["groups"] for a in g["actuators"]]
    assert sorted(ids) == list(range(sess.model.nu))


def test_scene_message_lists_available_settings(sess):
    msg = sess.scene_message()
    assert isinstance(msg["settings_available"], list)
    assert len(msg["settings_available"]) > 0
    assert all(isinstance(name, str) for name in msg["settings_available"])
    # The existing "settings" field (current vis_state) is unchanged by this addition.
    assert isinstance(msg["settings"], dict)


@pytest.mark.gl
def test_render_and_encode_round_trip(sess):
    frame = sess.render()
    assert frame.shape == (64, 64, 3)
    assert frame.dtype == np.uint8
    blob = sess.encode(frame)
    assert blob[:2] == b"\xff\xd8"  # JPEG SOI


@pytest.mark.gl
def test_resize_changes_frame_shape(sess):
    sess.resize(80, 48)
    assert sess.render().shape == (48, 80, 3)


# -- backend seam ------------------------------------------------------------


def test_default_backend_is_cpu(sess):
    assert isinstance(sess.backend, CpuBackend)


def test_scene_message_carries_backend_and_warning(sess):
    msg = sess.scene_message()
    assert msg["backend"] == "cpu"
    assert isinstance(msg["backend_warning"], str)
    assert msg["backend_warning"] != ""


def test_step_delegates_to_backend_and_syncs_afterwards(xml):
    """Session must not step physics itself: FakeBackend never calls mj_step, so if data.qpos
    changes at all it can only be through backend.step() + backend.sync_to(self.data)."""
    s = Session(xml_path=xml, width=64, height=64, backend=FakeBackend(nu=2, nq=2))
    try:
        s.step(4)
        assert s.backend.step_calls == [4]
        assert s.backend.sync_calls == 1
        np.testing.assert_allclose(s.data.qpos, [2.0, 2.0])  # 4 * 0.5, written via sync_to
    finally:
        s.close()


def test_fake_backend_scene_message_reports_its_own_label_and_warning(xml):
    s = Session(xml_path=xml, width=64, height=64, backend=FakeBackend(nu=2, nq=2))
    try:
        msg = s.scene_message()
        assert msg["backend"] == "fake"
        assert msg["backend_warning"] is None
    finally:
        s.close()


def test_reset_calls_backend_reset_to_keyframe_with_default_pose(xml):
    fake = FakeBackend(nu=2, nq=2)
    s = Session(xml_path=xml, width=64, height=64, backend=fake)
    try:
        s.step(4)
        s.reset()
        assert fake.reset_calls == ["default_pose"]
    finally:
        s.close()


def test_close_does_not_close_the_backend_twice(xml):
    """A backend holding a real device context (a future WarpBackend) must not be closed
    twice by a double Session.close() -- harmless for CpuBackend/FakeBackend here, but a trap
    for anything that isn't."""
    fake = FakeBackend(nu=2, nq=2)
    s = Session(xml_path=xml, width=64, height=64, backend=fake)
    s.close()
    s.close()  # idempotent
    assert fake.close_calls == 1
