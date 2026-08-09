"""Session owns model+data+renderer, stepped through a swappable physics backend. Everything
here runs headless; only the render and encode tests need GL.

Session no longer calls ``mj_step`` directly -- it delegates to a ``backend`` (default
``CpuBackend``, sharing Session's own ``MjData``) and syncs from it each step. A ``FakeBackend``
below proves that seam is swappable without a real physics engine (or JAX) in the loop.
"""

import mujoco
import numpy as np
import pytest

from mujoco_visualizer.serve.backends import CpuBackend, UnknownKeyframe
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


# A deliberately BIG, off-centre arm plus a named camera: the pixel-level tests below need a
# pose change (and a camera change) that moves a large fraction of the frame, so that
# "the render followed the state" is not a handful of anti-aliased edge pixels.
_POSE_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="topcam" pos="0 0 3" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j_T1_left" type="hinge" axis="0 1 0"/>
      <geom name="arm" type="box" size="0.4 0.06 0.06" pos="0.4 0 0" rgba="0.9 0.2 0.2 1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m_T1_left" joint="j_T1_left" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def xml(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_XML)
    return str(p)


@pytest.fixture
def pose_xml(tmp_path):
    p = tmp_path / "pose.xml"
    p.write_text(_POSE_XML)
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
        self.set_state_calls = []

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

    def set_state(self, qpos, qvel, time):
        self.set_state_calls.append(
            (np.asarray(qpos).copy(), np.asarray(qvel).copy(), float(time))
        )
        self._qpos[:] = np.asarray(qpos)[: len(self._qpos)]
        self._t = float(time)

    @property
    def time(self):
        return self._t

    def close(self):
        self.close_calls = getattr(self, "close_calls", 0) + 1


class DeviceBackend:
    """A backend whose authoritative state lives somewhere Session cannot see, exactly like
    :class:`scripts.vnc_explorer.live.WarpBackend`.

    The whole point: ``sync_to`` writes ONLY ``qpos``/``qvel``/``time`` into the host
    ``MjData`` and deliberately never calls ``mj_forward``. Everything ``mjv_updateScene``
    actually draws from -- ``xpos``, ``xquat``, ``geom_xpos`` -- is *derived* state that only
    ``mj_forward``/``mj_step`` computes, so a Session that does not forward after syncing
    renders a frozen (or origin-collapsed) pose no matter how the device state moves.
    ``CpuBackend`` cannot expose that bug: its ``mj_step`` populates those arrays as a side
    effect.
    """

    label = "device"
    warning = None

    def __init__(self, nq, nu, keyframes=("default_pose",)):
        self._qpos = np.zeros(nq)
        self._qvel = np.zeros(nq)
        self._t = 0.0
        self.nu = nu
        self.set_state_calls = []
        # Named keyframes this "device" knows, so reset() exercises the backend path rather
        # than Session's mj_resetData fallback (which forwards on its own and would mask the
        # missing forward this test exists to catch).
        self._keyframes = set(keyframes)

    # Test-side handle on the "device" state, standing in for real physics.
    def poke_qpos(self, qpos):
        self._qpos[:] = np.asarray(qpos, dtype=np.float64)

    def set_ctrl(self, ctrl):
        pass

    def step(self, n):
        self._t += int(n) * 1e-3

    def sync_to(self, data):
        data.qpos[:] = self._qpos
        data.qvel[:] = self._qvel
        data.time = self._t
        # NO mj_forward here, on purpose. See the class docstring.

    def reset_to_keyframe(self, name):
        if name not in self._keyframes:
            raise UnknownKeyframe(name)
        self._qpos[:] = 0.0
        self._qvel[:] = 0.0
        self._t = 0.0

    def set_state(self, qpos, qvel, time):
        self.set_state_calls.append(
            (np.asarray(qpos).copy(), np.asarray(qvel).copy(), float(time))
        )
        self._qpos[:] = np.asarray(qpos)
        self._qvel[:] = np.asarray(qvel)
        self._t = float(time)

    @property
    def time(self):
        return self._t

    def close(self):
        pass


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


def test_scene_message_settings_catalog_carries_origin_alongside_the_flat_list(xml, tmp_path):
    """Design spec S6: the scene message's settings_available entries gain an origin tag, so
    a later preset dropdown can mark each entry bundled or user. That must not change
    "settings_available" itself -- viewer.js iterates it expecting plain strings -- so the
    tagged shape is carried on a second, additive field, "settings_catalog", instead.
    """
    user_dir = tmp_path / "user_settings"
    user_dir.mkdir()
    (user_dir / "my_look.json").write_text("{}")

    s = Session(xml_path=xml, width=64, height=64, user_settings_dir=user_dir)
    try:
        msg = s.scene_message()

        # The pre-existing flat field is untouched: still a list of plain strings.
        assert isinstance(msg["settings_available"], list)
        assert all(isinstance(name, str) for name in msg["settings_available"])

        # The new field carries list_available_settings()'s real, untransformed shape.
        assert "settings_catalog" in msg
        catalog = msg["settings_catalog"]
        assert isinstance(catalog, list)
        assert all(set(d) == {"name", "origin"} for d in catalog)
        assert {d["origin"] for d in catalog} == {"bundled", "user"}
        assert {"name": "my_look", "origin": "user"} in catalog
        assert any(d["origin"] == "bundled" for d in catalog)
    finally:
        s.close()


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


# -- rendered state follows the backend (the "frozen fly" bug) ---------------
#
# mjv_updateScene draws from PRECOMPUTED xpos/xquat/geom_xpos, which only mj_forward/mj_step
# populate. A backend that owns its state off-host writes qpos/qvel/time in sync_to and
# nothing else, so unless Session forwards after syncing, the render is pinned to whatever
# derived state the MjData last happened to hold -- while sim_time and rtf keep advancing
# convincingly. These tests are deliberately PIXEL-level: a qpos-only assertion passes
# against the broken code.


def _frame_camera(sess):
    """Frame the arm from the side, writing the keys ``_cfg_to_mjvcamera`` reads directly.

    Deliberately not via ``set_camera`` -- these tests must fail for the C1 reason only, not
    depend on the separate camera-key fix.
    """
    sess.viz.vis_state["camera"].update(
        mode="free", azimuth=90.0, elevation=0.0, distance=2.0, lookat=[0.0, 0.0, 0.5]
    )


def test_construction_leaves_data_forwarded(pose_xml):
    """A fresh MjData has geom_xpos all-zero and xquat all-zero, so the frames published
    before the first Play (SimLoop starts with _playing=False but publishes every tick) draw
    the model collapsed at the origin on EVERY backend unless Session forwards at build."""
    s = Session(xml_path=pose_xml, width=64, height=64)
    try:
        ref = mujoco.MjData(s.model)
        mujoco.mj_forward(s.model, ref)
        np.testing.assert_allclose(s.data.geom_xpos, ref.geom_xpos)
        np.testing.assert_allclose(s.data.xquat, ref.xquat)
    finally:
        s.close()


def test_step_forwards_derived_state_for_a_device_resident_backend(pose_xml):
    """Cheap non-pixel half of the same claim: after step(), the arm's geom_xpos must match
    what mj_forward would compute for the synced qpos."""
    backend = DeviceBackend(nq=1, nu=1)
    s = Session(xml_path=pose_xml, width=64, height=64, backend=backend)
    try:
        backend.poke_qpos([1.2])
        s.step(1)
        ref = mujoco.MjData(s.model)
        ref.qpos[:] = [1.2]
        mujoco.mj_forward(s.model, ref)
        np.testing.assert_allclose(s.data.geom_xpos, ref.geom_xpos, atol=1e-9)
    finally:
        s.close()


@pytest.mark.gl
def test_rendered_pixels_follow_a_device_resident_backend(pose_xml):
    """THE regression guard for the frozen-fly bug.

    Steps a device-style backend (``sync_to`` writes qpos without forwarding, mimicking
    WarpBackend) through two very different poses and asserts the RENDERED PIXELS changed. A
    test that only checked ``data.qpos`` would pass against the broken code, because the sync
    itself was never the problem.
    """
    backend = DeviceBackend(nq=1, nu=1)
    s = Session(xml_path=pose_xml, width=128, height=128, backend=backend)
    try:
        _frame_camera(s)
        backend.poke_qpos([0.0])
        s.step(1)
        flat = s.render()
        backend.poke_qpos([1.2])
        s.step(1)
        tilted = s.render()
        diff = np.abs(flat.astype(int) - tilted.astype(int))
        assert diff.max() > 40, "rendered pose did not follow the backend's state"
        assert (diff.max(axis=2) > 20).sum() > 100, "only a handful of pixels moved"
    finally:
        s.close()


@pytest.mark.gl
def test_reset_forwards_derived_state_for_a_device_resident_backend(pose_xml):
    """reset() syncs from the backend too, so it needs the same forward -- otherwise the
    post-reset frame keeps drawing the pre-reset pose."""
    backend = DeviceBackend(nq=1, nu=1)
    s = Session(xml_path=pose_xml, width=128, height=128, backend=backend)
    try:
        _frame_camera(s)
        backend.poke_qpos([1.2])
        s.step(1)
        tilted = s.render()
        s.reset()  # the backend knows 'default_pose' and returns its own qpos to 0
        after = s.render()
        assert np.abs(tilted.astype(int) - after.astype(int)).max() > 40
    finally:
        s.close()


def test_divergence_rollback_pushes_the_good_state_back_into_the_backend(pose_xml):
    """``_restore`` rewriting only the host MjData leaves a device-resident backend holding
    the diverged state, so every later step() re-diverges until reset -- and the "rolled back
    to the last good step" message is false. The rollback must go through the backend."""
    backend = DeviceBackend(nq=1, nu=1)
    s = Session(xml_path=pose_xml, width=64, height=64, backend=backend)
    try:
        backend.poke_qpos([0.3])
        s.step(1)
        good = s.data.qpos.copy()

        backend.poke_qpos([np.nan])
        with pytest.raises(Diverged):
            s.step(1)

        assert backend.set_state_calls, "the rollback never reached the backend"
        qpos, _qvel, _t = backend.set_state_calls[-1]
        np.testing.assert_allclose(qpos, good)
        # The claim that matters: the backend is no longer diverged, so stepping resumes.
        s.step(1)
        assert np.isfinite(s.data.qpos).all()
    finally:
        s.close()


# -- per-frame warnings ------------------------------------------------------


def test_new_warnings_reports_a_per_frame_delta_not_the_cumulative_total(sess):
    """frame_meta.warn must come and go with the condition. warnings() stays cumulative for
    anything that wants that view, but a banner fed from it is pinned forever after the first
    warning -- and a permanently pinned banner permanently masks whatever shares its slot."""
    assert sess.new_warnings() is None

    sess.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number += 1
    assert "CONTACTFULL" in sess.new_warnings()
    # Nothing new happened since, so this frame is clean...
    assert sess.new_warnings() is None
    # ...while the cumulative view still remembers it.
    assert "CONTACTFULL" in sess.warnings()

    # A fresh occurrence is reported again.
    sess.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number += 1
    assert "CONTACTFULL" in sess.new_warnings()


def test_reset_clears_both_the_cumulative_and_the_per_frame_warning_views(sess):
    sess.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number += 3
    sess.new_warnings()
    sess.reset()
    assert sess.warnings() is None
    assert sess.new_warnings() is None
    # And the baseline came down with the counters, so the next real warning still registers.
    sess.data.warning[mujoco.mjtWarning.mjWARN_CONTACTFULL].number += 1
    assert "CONTACTFULL" in sess.new_warnings()


# -- scene message is a snapshot, not a live view ----------------------------


def test_scene_message_settings_is_a_deep_copy_of_vis_state(sess):
    """Flask request threads serialise this while the loop thread mutates vis_state via
    apply_render/load_settings. Returning it by reference is how json.dumps ends up raising
    "dictionary changed size during iteration" mid-response."""
    msg = sess.scene_message()
    assert msg["settings"] is not sess.viz.vis_state
    assert msg["settings"]["floor"] is not sess.viz.vis_state["floor"]

    sess.viz.vis_state["floor"]["alpha"] = 0.123
    sess.viz.vis_state["a_brand_new_key"] = 1
    assert msg["settings"]["floor"]["alpha"] != 0.123
    assert "a_brand_new_key" not in msg["settings"]


def test_scene_message_is_safe_after_close(sess):
    """A late /api/scene must not blow up just because close() dropped the backend."""
    sess.close()
    msg = sess.scene_message()
    assert msg["t"] == "scene"
    assert msg["backend"] is None


# -- camera ------------------------------------------------------------------
#
# The wire protocol says az/el/dist; Visualizer._cfg_to_mjvcamera reads
# azimuth/elevation/distance. Session.set_camera is the only place that translation can
# happen, and until it did, dragging the canvas wrote three dead keys into vis_state.


def test_free_camera_lands_on_the_keys_the_visualizer_actually_reads(pose_xml):
    s = Session(xml_path=pose_xml, width=64, height=64)
    try:
        s.set_camera(az=45.0, el=-10.0, dist=2.5, lookat=[0.1, 0.2, 0.3])
        cam = s.viz.vis_state["camera"]
        assert cam["azimuth"] == pytest.approx(45.0)
        assert cam["elevation"] == pytest.approx(-10.0)
        assert cam["distance"] == pytest.approx(2.5)
        assert list(cam["lookat"]) == pytest.approx([0.1, 0.2, 0.3])
        # And it reaches the MjvCamera the renderer is handed, which is the claim that
        # matters -- the vis_state keys above are only the mechanism.
        mjcam = s.viz.get_camera(s._camera)
        assert isinstance(mjcam, mujoco.MjvCamera)
        assert mjcam.azimuth == pytest.approx(45.0)
        assert mjcam.elevation == pytest.approx(-10.0)
        assert mjcam.distance == pytest.approx(2.5)
    finally:
        s.close()


def test_free_camera_update_overrides_a_named_camera_settings_file(pose_xml):
    """live.py defaults to --settings Earthy_V1, whose camera.mode is "named" -- and
    Visualizer.get_camera short-circuits to the XML camera whenever mode == 'named'. So a
    correct az/el/dist write is still a no-op unless the mode flips to free."""
    s = Session(xml_path=pose_xml, width=64, height=64)
    try:
        s.viz.vis_state["camera"].update(mode="named", named="topcam")
        assert s.viz.get_camera(s._camera) == "topcam"  # precondition
        s.set_camera(az=45.0)
        assert s.viz.vis_state["camera"]["mode"] == "free"
        mjcam = s.viz.get_camera(s._camera)
        assert isinstance(mjcam, mujoco.MjvCamera)
        assert mjcam.azimuth == pytest.approx(45.0)
    finally:
        s.close()


def test_named_camera_command_still_selects_the_named_camera(pose_xml):
    s = Session(xml_path=pose_xml, width=64, height=64)
    try:
        s.set_camera(az=45.0)  # go free first
        s.set_camera(named="topcam")
        assert s.viz.get_camera(s._camera) == "topcam"
    finally:
        s.close()


def test_camera_property_reports_what_render_will_use(pose_xml):
    """``Session.camera`` is what an export job is handed, so it must track set_camera.

    Exists so a host project (the fly rollout viewer's export factory) does not have to read
    the private ``_camera`` to render its video with the camera the preview was showing.
    """
    s = Session(xml_path=pose_xml, width=64, height=64)
    try:
        assert s.camera is None                # free camera by default
        s.set_camera(named="topcam")
        assert s.camera == "topcam"
        s.set_camera(az=45.0)                  # a drag returns to the free camera
        assert s.camera is None
    finally:
        s.close()


@pytest.mark.gl
def test_rendered_pixels_follow_a_camera_drag(pose_xml):
    """The end-to-end claim: two different azimuths must produce different frames."""
    s = Session(xml_path=pose_xml, width=128, height=128)
    try:
        s.set_camera(az=90.0, el=0.0, dist=2.0, lookat=[0.0, 0.0, 0.5])
        side = s.render()
        s.set_camera(az=0.0)
        front = s.render()
        assert np.abs(side.astype(int) - front.astype(int)).max() > 40
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


def test_load_settings_rejects_a_path_even_though_visualizer_accepts_one(sess, tmp_path):
    """The serve layer's whitelist is enforced here too, so the invariant does not depend on
    which entry point reached the Session. Visualizer.load_settings deliberately still takes
    paths for its own local callers."""
    victim = tmp_path / "secret.json"
    victim.write_text('{"alpha": 0.5}')
    with pytest.raises(ValueError) as exc:
        sess.load_settings(str(victim))
    assert str(victim) in str(exc.value)


def test_load_settings_accepts_a_bundled_preset_name(sess):
    from mujoco_visualizer import list_available_settings

    sess.load_settings(list_available_settings()[0]["name"])  # must not raise


# -- vis_state_snapshot / swap_model ----------------------------------------
#
# NOTE: the task brief's snippets name the single-model fixture `session`, but this file's
# existing fixture for "one small model, width=64, height=64" is `sess` -- there is no
# `session` fixture anywhere in this package. Using `sess` here rather than introducing a
# duplicate fixture under a second name.


def test_vis_state_snapshot_is_a_deep_copy(sess):
    snap = sess.vis_state_snapshot()
    snap["vis_flags"]["shadows"] = "mutated"
    assert sess.viz.vis_state["vis_flags"]["shadows"] != "mutated"


def test_active_model_name_defaults_to_primary(sess):
    assert sess.active_model_name == "primary"


def test_swap_model_without_alt_raises(sess):
    with pytest.raises(ValueError, match="no alt_model"):
        sess.swap_model("alt")


def test_swap_model_rejects_unknown_name(sess):
    with pytest.raises(ValueError, match="unknown"):
        sess.swap_model("ghost")


_ONE_BODY = """
<mujoco><worldbody>
  <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1"/></body>
</worldbody></mujoco>
"""

_TWO_BODY = """
<mujoco><worldbody>
  <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1"/></body>
  <body name="b2" pos="0 .5 0"><joint name="j2" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1" rgba=".8 .8 .8 .3"/></body>
</worldbody></mujoco>
"""


@pytest.fixture
def two_model_session():
    import mujoco
    from mujoco_visualizer.serve.session import Session

    primary = mujoco.MjModel.from_xml_string(_ONE_BODY)
    alt = mujoco.MjModel.from_xml_string(_TWO_BODY)
    s = Session(model=primary, alt_model=alt, width=64, height=48)
    try:
        yield s
    finally:
        s.close()


@pytest.mark.gl
def test_swap_model_switches_nq_and_keeps_rendering(two_model_session):
    """The ghost toggle's whole job: a different model, still rendering, right size."""
    s = two_model_session
    primary_nq = s.model.nq
    s.swap_model("alt")
    assert s.active_model_name == "alt"
    assert s.model.nq != primary_nq
    frame = s.render()
    assert frame.shape == (s.height, s.width, 3)
    s.swap_model("primary")
    assert s.model.nq == primary_nq
    assert s.render().shape == (s.height, s.width, 3)


@pytest.mark.gl
def test_swap_model_preserves_the_edited_look(two_model_session):
    """A swap rebuilds the renderer; it must not silently reset the user's settings."""
    s = two_model_session
    s.apply_render({"vis_flags.shadows": False})
    s.swap_model("alt")
    assert s.viz.vis_state["vis_flags"]["shadows"] is False


# -- Fix round 1: failure safety, hidden self-state, and stale geom ids ------


@pytest.mark.gl
def test_swap_model_rolls_back_on_a_transient_failure(two_model_session, monkeypatch):
    """A failure partway through the rebind must not leave the renderer null nor the
    Visualizer half-migrated to the new model.

    ``Visualizer.rebind_model`` sets ``self.model`` to the new model, THEN recomputes the
    caches that read it (``_rebuild_model_derived_state``) -- so a failure in that second step
    is exactly the scenario where ``self.viz.model`` can already be the new model while
    everything else on the Session is still the old one. Patched to fail only on the first
    call (the swap's own attempt) and succeed on the second (swap_model's own rollback), which
    models a transient failure -- e.g. a one-off allocation error -- rather than a permanently
    broken model, and is exactly the case the rollback exists to recover from.
    """
    import mujoco_visualizer.visualizer as viz_mod

    s = two_model_session
    primary_nq = s.model.nq
    original = viz_mod.Visualizer._rebuild_model_derived_state
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient rebind failure")
        return original(self)

    monkeypatch.setattr(viz_mod.Visualizer, "_rebuild_model_derived_state", flaky)

    with pytest.raises(RuntimeError, match="simulated transient rebind failure"):
        s.swap_model("alt")

    assert calls["n"] == 2  # the swap's own attempt, then swap_model's rollback
    assert s.active_model_name == "primary"
    assert s.model.nq == primary_nq
    assert s.viz.model is s.model
    assert s.viz.data is s.data
    frame = s.render()
    assert frame.shape == (s.height, s.width, 3)
    # The Session must still be fully usable afterwards, not just able to render once.
    s.swap_model("alt")
    assert s.active_model_name == "alt"
    assert s.render().shape == (s.height, s.width, 3)


@pytest.mark.gl
def test_swap_model_keeps_the_old_renderer_live_if_the_new_one_fails_to_build(
    two_model_session, monkeypatch
):
    """The literal bug this all started from: ``self._renderer`` must never be nulled before
    the replacement exists, so a failure in :meth:`Visualizer.make_renderer` itself -- the
    step most analogous to a GL-context hiccup during :meth:`Session.resize` -- must leave
    ``render()`` still serving the last good frame instead of crashing on ``None``.
    """
    s = two_model_session

    def boom(*a, **kw):
        raise RuntimeError("simulated GL failure building the new renderer")

    monkeypatch.setattr(s.viz, "make_renderer", boom)

    with pytest.raises(RuntimeError, match="simulated GL failure"):
        s.swap_model("alt")

    assert s.active_model_name == "primary"
    assert s._renderer is not None
    frame = s.render()
    assert frame.shape == (s.height, s.width, 3)


def test_rebuild_model_derived_state_returns_rather_than_stashes(sess):
    """Regression for the hidden self-state finding: __init__'s vis_state literal must not
    depend on attributes a prior call happened to leave on ``self`` -- the floor/light
    baseline is returned, not stashed, and nothing under that old name lingers on the
    instance."""
    result = sess.viz._rebuild_model_derived_state()
    assert hasattr(result, "lights")
    assert hasattr(result, "floor_rgb")
    assert hasattr(result, "floor_alpha")
    assert hasattr(result, "floor_mat_props")
    for leaked_name in ("_init_lights", "_init_floor_rgb", "_init_floor_alpha",
                        "_init_floor_mat_props"):
        assert not hasattr(sess.viz, leaked_name)


@pytest.mark.gl
def test_swap_model_drops_geom_colors_that_do_not_exist_on_the_new_model(two_model_session):
    """``geom_colors`` is keyed by geom id, and ids are model-specific: swapping from the
    bigger model (2 geoms) to the smaller one (1 geom) must drop the id that no longer exists
    rather than carry a dangling reference across -- while the still-valid id survives."""
    s = two_model_session
    s.swap_model("alt")  # alt (_TWO_BODY) has geom ids 0 and 1
    assert s.model.ngeom == 2
    s.apply_render({"geom_colors": {0: "#ff0000", 1: "#00ff00"}})
    assert s.viz.vis_state["geom_colors"] == {0: "#ff0000", 1: "#00ff00"}

    s.swap_model("primary")  # primary (_ONE_BODY) has only geom id 0

    assert s.model.ngeom == 1
    assert s.viz.vis_state["geom_colors"] == {0: "#ff0000"}


def test_a_dotted_geom_color_key_lands_as_an_int_geom_id(sess):
    """``geom_colors`` is keyed by INT geom id everywhere else in this package.

    ``apply_render`` walks dotted keys verbatim, so a wire command
    ``{"t":"render","set":{"geom_colors.0":"#f00"}}`` inserted the STRING ``"0"``. Two
    consequences: ``Visualizer._apply_geom_colors`` tests ``if i in geom_overrides`` with an
    int, so the colour never applied at all; and the next model swap compared ``"0" <
    model.ngeom`` and raised TypeError, surfacing as the ghost toggle failing.
    """
    sess.apply_render({"geom_colors.0": "#ff0000"})
    assert sess.viz.vis_state["geom_colors"] == {0: "#ff0000"}
    assert all(isinstance(k, int) for k in sess.viz.vis_state["geom_colors"])


def test_a_dotted_geom_color_key_that_is_not_a_geom_id_is_refused(sess):
    """A non-integer key cannot name a geom, so it is a client error -- reported as a bad
    command (which does not pause the shared session) rather than stored to break a later
    swap."""
    with pytest.raises(ValueError, match="keyed by geom id"):
        sess.apply_render({"geom_colors.floor": "#ff0000"})
    assert "floor" not in sess.viz.vis_state["geom_colors"]


def test_other_dotted_keys_are_still_written_verbatim(sess):
    """Only geom_colors is coerced: every other sub-dict really is keyed by name."""
    sess.apply_render({"floor.alpha": 0.25, "vis_flags.shadows": False})
    assert sess.viz.vis_state["floor"]["alpha"] == 0.25
    assert sess.viz.vis_state["vis_flags"]["shadows"] is False


def test_carrying_geom_colors_tolerates_a_non_int_key_directly():
    """``_carry_vis_state_across_swap``'s own coercion, tested at the function.

    ``Session.apply_render`` now normalises wire keys before they are ever stored, so the GL
    swap test below no longer reaches this branch -- it would pass with the coercion removed.
    Testing the function directly is what keeps the defence from becoming a line nothing
    constrains: this is the last barrier for any future producer of string keys.
    """
    from mujoco_visualizer.serve.session import _carry_vis_state_across_swap

    model = type("M", (), {"ngeom": 3})()
    state = {"geom_colors": {"1": "#a", 2: "#b", "9": "#c", 5: "#d", -1: "#e", "x": "#f"}}
    _carry_vis_state_across_swap(state, model)
    # "1" survives as int 1; 2 stays; out-of-range ("9", 5), negative and non-numeric go.
    assert state["geom_colors"] == {1: "#a", 2: "#b"}


@pytest.mark.gl
def test_a_wire_inserted_geom_color_key_does_not_break_a_model_swap(two_model_session):
    """The end-to-end shape of the bug: colour a geom through the wire path, then toggle the
    ghost. Before the coercion this raised ``TypeError: '<' not supported between instances of
    'str' and 'int'`` out of ``_carry_vis_state_across_swap`` and the swap failed."""
    s = two_model_session
    s.swap_model("alt")
    s.apply_render({"geom_colors.1": "#00ff00"})
    s.swap_model("primary")  # geom id 1 does not exist here, so it must be DROPPED, not raise
    assert s.viz.vis_state["geom_colors"] == {}
    assert s.active_model_name == "primary"


def test_apply_render_sets_a_list_element(sess):
    before = list(sess.viz.vis_state["geom_groups"])
    sess.apply_render({"geom_groups.3": not before[3]})
    after = sess.viz.vis_state["geom_groups"]
    assert after[3] is (not before[3])
    assert after[:3] == before[:3] and after[4:] == before[4:], "only index 3 may change"
    assert isinstance(after, list), "the list must stay a list, not become a dict"


def test_apply_render_list_index_out_of_range_names_the_bound(sess):
    n = len(sess.viz.vis_state["geom_groups"])
    with pytest.raises(ValueError, match=rf"9.*{n}"):
        sess.apply_render({"geom_groups.9": True})


def test_apply_render_list_index_must_be_an_integer(sess):
    with pytest.raises(ValueError, match="integer"):
        sess.apply_render({"geom_groups.x": True})


def test_apply_render_still_sets_nested_dict_keys(sess):
    sess.apply_render({"floor.alpha": 0.25, "vis_flags.shadows": False})
    assert sess.viz.vis_state["floor"]["alpha"] == 0.25
    assert sess.viz.vis_state["vis_flags"]["shadows"] is False


def test_apply_render_still_coerces_geom_colors_keys(sess):
    sess.apply_render({"geom_colors.5": "#ff0000"})
    assert 5 in sess.viz.vis_state["geom_colors"], "int coercion must survive this change"


def test_a_rejected_bare_list_key_never_reaches_vis_state(sess):
    """End-to-end shape of the destructive case ``protocol._LIST_VALUED_ROOTS`` guards
    against: a wire command naming ``geom_groups`` with no ``.<index>`` must be rejected by
    ``parse_command`` BEFORE ``apply_render`` ever runs, so ``vis_state['geom_groups']`` is
    still the same list -- not a bool, and not even a same-length copy standing in for the
    original.

    Deliberately NOT written as a bare ``pytest.raises`` around both calls: if the guard were
    ever removed, ``parse_command`` would return normally and ``apply_render`` WOULD run
    inside that block, corrupting ``vis_state`` -- but ``pytest.raises`` would report only
    "DID NOT RAISE", never showing that corruption. Running both calls in a plain ``try`` and
    asserting on the resulting state afterwards means a missing guard shows up as
    ``geom_groups`` having become a bool, which is the actual consequence this guards
    against -- not just a missing exception.
    """
    from mujoco_visualizer.serve.protocol import CommandError, parse_command

    before = sess.viz.vis_state["geom_groups"]
    before_len = len(before)
    raised = False
    try:
        cmd = parse_command({"t": "render", "set": {"geom_groups": True}})
        sess.apply_render(cmd["set"])
    except CommandError:
        raised = True
    after = sess.viz.vis_state["geom_groups"]
    assert isinstance(after, list), f"geom_groups must stay a list; got {after!r}"
    assert after is before and len(after) == before_len, "apply_render must never have run"
    assert raised, "parse_command must reject the bare list key"
