"""SimLoop drives a Session on a thread. Tested against a fake session, so none of this
needs GL or a real model."""

import contextlib
import threading
import time

import mujoco
import numpy as np
import pytest

from mujoco_visualizer.serve.loop import SimLoop
from mujoco_visualizer.serve.replay import ArrayTrajectorySource
from mujoco_visualizer.serve.session import Diverged

# A tiny real model for the lock tests: three independent hinge DOFs (nq == 3, matching
# make_source()'s default), so build_joint_qpos_map is exercised for real instead of a faked
# map. "joint0" is left unlocked in every lock test (dof 0 already encodes the frame index, per
# make_source below) and "joint1"/"joint2" are free to be locked without disturbing it.
_LOCK_XML = """
<mujoco>
  <!-- timestep pinned to 1e-4: _physics_steps_per_control_step reads model.opt.timestep once
       a session HAS a `.model` (which FakeSession now does, for build_joint_qpos_map), and
       every controller-rate test in this file (e.g. RampSession) is written assuming exactly
       the dt=1e-4 MuJoCo's own default (2e-3) would silently replace. -->
  <option timestep="0.0001"/>
  <worldbody>
    <body name="b0"><joint name="joint0" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b1"><joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b2"><joint name="joint2" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
  </worldbody>
</mujoco>
"""
_LOCK_MODEL = mujoco.MjModel.from_xml_string(_LOCK_XML)

# The "ghost" counterpart: same nq (3), but the middle joint is named as if it were a
# suffixed reference copy -- proof that a loop's cached joint map is rebuilt from THIS model,
# not the primary one, once a ghost swap lands (see
# test_the_joint_map_rebuilds_after_a_ghost_model_swap).
_LOCK_ALT_XML = """
<mujoco>
  <option timestep="0.0001"/>
  <worldbody>
    <body name="b0"><joint name="joint0" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b1"><joint name="joint1_ref" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b2"><joint name="joint2" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
  </worldbody>
</mujoco>
"""
_LOCK_ALT_MODEL = mujoco.MjModel.from_xml_string(_LOCK_ALT_XML)

# An nq-CHANGING ghost pair: primary's three joints kept at the SAME names/addresses (0, 1, 2),
# plus a suffixed reference copy of each appended after them (3, 4, 5) -- the actual shape of a
# real doubled ghost model (policy + suffixed reference), and the one shape review round 1
# found the swap fixture above (which keeps nq == 3 throughout) could not exercise: a lock
# resolved/expanded against the OLD (narrower) map or qpos and then addressed against the NEW
# (wider) one is exactly what silently truncates via numpy's past-the-end slicing rather than
# raising.
_LOCK_WIDE_ALT_XML = """
<mujoco>
  <option timestep="0.0001"/>
  <worldbody>
    <body name="b0"><joint name="joint0" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b1"><joint name="joint1" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b2"><joint name="joint2" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b0r"><joint name="joint0_ref" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b1r"><joint name="joint1_ref" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
    <body name="b2r"><joint name="joint2_ref" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/></body>
  </worldbody>
</mujoco>
"""
_LOCK_WIDE_ALT_MODEL = mujoco.MjModel.from_xml_string(_LOCK_WIDE_ALT_XML)


class FakeSession:
    """Duck-typed stand-in for Session: records calls, renders a 1x1 frame."""

    def __init__(self, diverge_after=None):
        self.width, self.height = 1, 1
        self.steps = 0
        self.renders = 0
        self.controller_advances = 0
        self.ctrl_calls = []
        self.group_calls = []
        self.camera_calls = []
        self.render_calls = []
        self.settings_loaded = []
        self.settings_saved = []
        self.resizes = []
        self.resets = 0
        self.mode = None
        self.closed = False
        self.controller_rate_hz = None
        self._diverge_after = diverge_after
        self._time = 0.0
        self.qpos_writes = []
        self.model_swaps = []
        self.pose = None
        # For build_joint_qpos_map to exercise a real model. See _LOCK_MODEL/_LOCK_ALT_MODEL
        # above; swap_model below actually switches this, mirroring the real Session so a
        # ghost toggle is visible to SimLoop's own joint-map cache, not just recorded here.
        self.model = _LOCK_MODEL

    # -- surface SimLoop uses --
    @property
    def data(self):
        return type("D", (), {"time": self._time})()

    def step(self, n):
        if self._diverge_after is not None and self.steps >= self._diverge_after:
            raise Diverged("boom")
        self.steps += n
        self._time += n * 1e-4

    def advance_controller(self):
        self.controller_advances += 1

    def readout(self):
        return {}

    def warnings(self):
        return None

    def new_warnings(self):
        return None

    def render(self):
        self.renders += 1
        return np.zeros((1, 1, 3), np.uint8)

    def encode(self, frame):
        return b"\xff\xd8jpeg"

    def set_ctrl(self, values):
        self.ctrl_calls.append(values)

    def set_group_gain(self, group, gain):
        self.group_calls.append((group, gain))

    def set_ctrl_mode(self, mode):
        self.mode = mode

    def set_camera(self, **kw):
        self.camera_calls.append(kw)

    def apply_render(self, settings):
        self.render_calls.append(settings)

    def load_settings(self, name):
        self.settings_loaded.append(name)

    def save_settings_as(self, name):
        self.settings_saved.append(name)
        return f"/fake/user/settings/{name}.json"

    # -- added for replay mode --
    def set_qpos(self, qpos):
        self.qpos_writes.append(np.asarray(qpos).copy())
        # `pose` models WHAT IS ON SCREEN, as opposed to `qpos_writes`, which is a history.
        # The real Session.reset() snaps qpos to a keyframe, so a reset that does not
        # re-render leaves the canvas showing the rest pose while replay_state() keeps
        # reporting the old frame -- see
        # test_a_reset_in_replay_mode_re_renders_the_frame_it_reports.
        self.pose = np.asarray(qpos).copy()

    def swap_model(self, which):
        self.model_swaps.append(which)
        self.model = _LOCK_ALT_MODEL if which == "alt" else _LOCK_MODEL

    def vis_state_snapshot(self):
        return {}

    def reset(self):
        self.resets += 1
        self.pose = "rest"  # what Session.reset() does: snap qpos to the rest keyframe

    def resize(self, w, h):
        self.resizes.append((w, h))
        self.width, self.height = w, h

    def scene_message(self):
        return {"t": "scene", "nu": 0}

    def close(self):
        self.closed = True


class WideGhostFakeSession(FakeSession):
    """Swaps to ``_LOCK_WIDE_ALT_MODEL`` (nq changes 3 -> 6) instead of the nq-preserving
    ``_LOCK_ALT_MODEL`` -- see review round 1: a fixture that keeps nq constant across every
    swap cannot exercise the stale-width truncation bug (a lock resolved/expanded against one
    map's addresses and then read against a different, WIDER map's ``qpos``)."""

    def swap_model(self, which):
        self.model_swaps.append(which)
        self.model = _LOCK_WIDE_ALT_MODEL if which == "alt" else _LOCK_MODEL


def wait_until(predicate, timeout=5.0):
    """Poll until predicate() is truthy or the timeout expires. Returns predicate()."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@contextlib.contextmanager
def running(loop):
    """Start the loop and guarantee it is stopped and joined afterwards.

    Use this rather than a start-wait-stop helper whenever the test needs to assert
    something WHILE the loop is live -- e.g. submitting a pause and observing that stepping
    actually ceases, or checking the thread survived an error. A helper that stops the loop
    before the assertions run cannot test either of those.
    """
    loop.start()
    try:
        yield loop
    finally:
        loop.stop()
        loop.join(timeout=5.0)


def _run_briefly(loop, predicate, timeout=5.0):
    """Start the loop, wait for predicate(), then stop and join. Returns predicate().

    For tests that only need a post-mortem assertion. If the assertion needs the loop
    alive, use ``running`` instead.
    """
    with running(loop):
        return wait_until(predicate, timeout)


def test_does_not_step_until_played():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=5, idle_pause_s=None)
    assert _run_briefly(loop, lambda: sess.renders > 2)
    assert sess.steps == 0  # renders while paused, never steps


def test_play_starts_stepping():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=5, idle_pause_s=None)
    loop.submit({"t": "sim", "cmd": "play", "n": 1})
    assert _run_briefly(loop, lambda: sess.steps >= 10)


def test_pause_stops_stepping():
    """Pause must actually stop stepping, observed while the loop is still live."""
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=5, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "sim", "cmd": "play", "n": 1})
        assert wait_until(lambda: sess.steps >= 10)
        loop.submit({"t": "sim", "cmd": "pause", "n": 1})
        assert wait_until(lambda: loop.playing is False)
        settled = sess.steps
        time.sleep(0.15)                    # several ticks at fps_cap=60
        assert sess.steps == settled        # nothing advanced while paused
        assert sess.renders > 0             # but it kept rendering (render-only ticks)


def test_single_step_advances_exactly_once_then_pauses():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=7, idle_pause_s=None)
    loop.submit({"t": "sim", "cmd": "step", "n": 1})
    _run_briefly(loop, lambda: sess.steps >= 7)
    assert sess.steps == 7
    assert loop.playing is False


def test_reset_is_forwarded():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=5, idle_pause_s=None)
    loop.submit({"t": "sim", "cmd": "reset", "n": 1})
    assert _run_briefly(loop, lambda: sess.resets >= 1)


def test_settings_load_is_forwarded_to_the_session():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=5, idle_pause_s=None)
    loop.submit({"t": "settings", "load": "Default"})
    assert _run_briefly(loop, lambda: sess.settings_loaded == ["Default"])


def test_settings_save_is_forwarded_to_the_session():
    """protocol.parse_command can hand the loop {"t": "settings", "save": name} -- _apply
    must route that to save_settings_as, not treat it as a load and KeyError on cmd["load"]."""
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=5, idle_pause_s=None)
    loop.submit({"t": "settings", "save": "my_look"})
    assert _run_briefly(loop, lambda: sess.settings_saved == ["my_look"])
    assert sess.settings_loaded == []


def test_a_failed_settings_save_is_reported_not_fatal():
    """save_settings_as raising (bad name reaching _apply somehow, unwritable dir, ...) must
    surface as a command error like any other bad command -- not kill the sim thread."""
    class ExplodingSession(FakeSession):
        def save_settings_as(self, name):
            raise OSError("disk full")

    sess = ExplodingSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=5, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "settings", "save": "whatever"})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "command"
        assert "disk full" in loop.error["msg"]
        # The thread survived -- it can still take and apply further commands.
        loop.submit({"t": "sim", "cmd": "step", "n": 1})
        assert wait_until(lambda: sess.steps >= 1)


def test_commands_are_coalesced_before_applying():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=5, substeps_per_frame=1, idle_pause_s=None)
    for az in (1.0, 2.0, 3.0):
        loop.submit({"t": "camera", "az": az})
    assert _run_briefly(loop, lambda: len(sess.camera_calls) >= 1)
    assert sess.camera_calls == [{"az": 3.0}]


def test_divergence_pauses_and_reports_instead_of_killing_the_thread():
    """The thread must SURVIVE a divergence: still alive, still publishing frames.

    Asserting ``is_alive() is False`` after the helper stopped the loop would prove nothing
    -- the helper always stops it. The real claim is that the loop kept running, so the
    client keeps receiving the last good frame and can hit reset.
    """
    sess = FakeSession(diverge_after=0)
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "sim", "cmd": "play", "n": 1})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "diverged"
        assert loop.playing is False
        assert loop.is_alive() is True
        renders_at_error = sess.renders
        assert wait_until(lambda: sess.renders > renders_at_error)


def test_controller_is_advanced_at_rate_hz_not_per_step():
    sess = FakeSession()
    sess.controller_rate_hz = 1000.0  # 1 kHz, dt=1e-4 -> one advance per 10 steps
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=100, idle_pause_s=None)
    loop.submit({"t": "sim", "cmd": "play", "n": 1})
    assert _run_briefly(loop, lambda: sess.steps >= 100)
    # 100 physics steps at dt=1e-4 is 10 ms of sim time -> ~10 controller advances
    assert 8 <= sess.controller_advances <= 12


class RampSession(FakeSession):
    """A session whose controller output ramps, and whose ``step`` records the controller
    value in force for the physics steps it just ran.

    That record is the only way to see the real bug: advancing the controller 26 times and
    THEN handing Session one ctrl vector for all 260 physics steps discards 25 of the 26
    outputs, so ``ctrl`` updates at ~38 Hz instead of the trained 1 kHz. A test that counted
    only ``advance_controller`` calls passes against that.
    """

    def __init__(self):
        super().__init__()
        self.controller_rate_hz = 1000.0  # dt=1e-4 -> one control step per 10 physics steps
        self._out = 0.0
        self.ctrl_per_chunk = []  # (n_physics_steps, controller value used for them)

    def advance_controller(self):
        self.controller_advances += 1
        self._out = float(self.controller_advances)

    def step(self, n):
        self.ctrl_per_chunk.append((n, self._out))
        super().step(n)


def test_controller_output_is_interleaved_with_physics_within_one_tick():
    """One 100-substep tick at 1 kHz must consume TEN distinct controller outputs, not one.

    Driven by a single ``sim step`` (not play) at fps_cap=1 so exactly one tick's worth of
    stepping happens and the assertions are deterministic.
    """
    sess = RampSession()
    loop = SimLoop(sess, fps_cap=1, substeps_per_frame=100, idle_pause_s=None)
    loop.submit({"t": "sim", "cmd": "step", "n": 1})
    assert _run_briefly(loop, lambda: sess.steps >= 100)

    assert sess.steps == 100, "no physics steps lost or double-counted"
    assert sum(n for n, _ in sess.ctrl_per_chunk) == 100
    assert sess.controller_advances == 10

    values = [v for _, v in sess.ctrl_per_chunk]
    assert len(set(values)) > 1, (
        "ctrl took a single value across the whole tick: every controller output but the "
        "last was discarded"
    )
    assert len(set(values)) == 10
    # Interleaved, not front-loaded: the controller runs before each chunk, so the values
    # ascend in step with the chunks.
    assert values == sorted(values)


def test_interleaving_preserves_the_controller_rate_across_ticks():
    """The controller still runs at its own rate_hz, not once per physics step: the FRACTION
    of steps per control step is what changed hands, never the rate itself."""
    sess = RampSession()
    sess.controller_rate_hz = 250.0  # dt=1e-4 -> one control step per 40 physics steps
    loop = SimLoop(sess, fps_cap=1, substeps_per_frame=100, idle_pause_s=None)
    loop.submit({"t": "sim", "cmd": "step", "n": 1})
    assert _run_briefly(loop, lambda: sess.steps >= 100)
    assert sess.steps == 100
    assert sum(n for n, _ in sess.ctrl_per_chunk) == 100
    # 100 steps of 1e-4 s is 10 ms; at 250 Hz that is 2.5 control steps. The fractional
    # remainder is carried, not dropped or rounded up into an extra advance.
    assert sess.controller_advances == 3
    assert [n for n, _ in sess.ctrl_per_chunk] == [40, 40, 20]


def test_divergence_mid_tick_is_reported_exactly_once():
    """With the tick now split into chunks, a divergence must still surface as ONE error."""

    class DivergingRamp(RampSession):
        def step(self, n):
            self.ctrl_per_chunk.append((n, self._out))
            if self.steps >= 40:
                raise Diverged("boom")
            self.steps += n
            self._time += n * 1e-4

    sess = DivergingRamp()
    loop = SimLoop(sess, fps_cap=1, substeps_per_frame=100, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "sim", "cmd": "play", "n": 1})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "diverged"
        assert loop.playing is False
        # Stopped at the diverging chunk instead of grinding through the rest of the tick.
        assert sess.steps == 40
        assert loop.is_alive() is True


def test_controller_exception_pauses_instead_of_killing_the_thread():
    class Boom(FakeSession):
        def advance_controller(self):
            raise RuntimeError("jax recompile failed")

    sess = Boom()
    sess.controller_rate_hz = 1000.0
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=10, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "sim", "cmd": "play", "n": 1})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "controller"
        assert loop.playing is False
        assert loop.is_alive() is True       # a JAX error must not kill the session
    assert loop.playing is False


def test_bad_command_reports_but_does_not_pause_playback():
    """A malformed/rejected command is a client-input problem, not evidence the physics
    state is untrustworthy -- unlike divergence/controller/render errors, it must NOT pause
    playback. SimLoop is shared across viewers: pausing here would let one client's bad or
    version-skewed message freeze the session for every other viewer -- a denial of service
    via a single malformed message."""

    class BadCtrl(FakeSession):
        def set_ctrl(self, values):
            raise KeyError("unknown actuator 'nope'")

    sess = BadCtrl()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "sim", "cmd": "play", "n": 1})
        assert wait_until(lambda: sess.steps >= 1)
        loop.submit({"t": "ctrl", "set": {"nope": 1.0}})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "command"
        assert loop.is_alive() is True
        assert loop.playing is True  # playback must continue despite the bad command
        steps_after_error = sess.steps
        assert wait_until(lambda: sess.steps > steps_after_error)


def test_malformed_command_without_type_field_reports_error_without_killing_thread():
    """A command dict with no 't' key fails inside coalesce()/_drain() -- before any
    individual command is ever applied to the session. That must not kill the thread, and
    (being a client-input problem like the bad-command case above) must not pause
    playback either."""
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "sim", "cmd": "play", "n": 1})
        assert wait_until(lambda: sess.steps >= 1)
        loop.submit({"no_type_field": "oops"})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "command"
        assert loop.is_alive() is True
        assert loop.playing is True
        steps_after_error = sess.steps
        assert wait_until(lambda: sess.steps > steps_after_error)


def test_render_failure_pauses_and_reports_instead_of_killing_the_thread():
    """A render-side failure (e.g. a GL error) must pause and report too, not just be
    swallowed while the loop keeps trying to step blind."""

    class BadRender(FakeSession):
        def __init__(self):
            super().__init__()
            self._fail_at = 3

        def render(self):
            self.renders += 1
            if self.renders >= self._fail_at:
                raise RuntimeError("GL context lost")
            return np.zeros((1, 1, 3), np.uint8)

    sess = BadRender()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    with running(loop):
        loop.submit({"t": "sim", "cmd": "play", "n": 1})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "render"
        assert loop.playing is False
        assert loop.is_alive() is True


def test_wait_for_frame_blocks_until_timeout_when_no_frame_yet():
    """Before any frame is ever published, _seq == 0 and the 'never seen a frame' sentinel
    is last_seq=-1 -- so a naive '_seq <= last_seq' pre-wait check (0 <= -1 is False) would
    skip the wait entirely and return None immediately. That would busy-spin a caller
    polling in a loop (the expected usage pattern, see the watcher above) at native call
    rate. Assert this actually blocks for ~timeout, not that it merely returns None."""
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    # Deliberately never started: no frame has been published, so this exercises the
    # first-ever-frame case directly and deterministically (nothing will ever notify).
    start = time.monotonic()
    got = loop.wait_for_frame(-1, timeout=0.3)
    elapsed = time.monotonic() - start
    assert got is None
    assert elapsed >= 0.25  # actually waited out (most of) the timeout, not an instant return


def test_latest_frame_slot_drops_intermediates():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=200, substeps_per_frame=1, idle_pause_s=None)
    seen = []

    def watcher():
        seq = -1
        deadline = time.time() + 2.0
        while time.time() < deadline:
            got = loop.wait_for_frame(seq, timeout=0.2)
            if got is None:
                continue
            seq, _, _ = got
            seen.append(seq)
            time.sleep(0.05)  # deliberately slow client

    t = threading.Thread(target=watcher)
    t.start()
    _run_briefly(loop, lambda: sess.renders > 40, timeout=2.5)
    t.join(timeout=3.0)
    assert len(seen) >= 2
    # a slow client must skip ahead, never walk every seq in order
    assert max(np.diff(seen)) > 1


def test_resize_is_applied_once():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    loop.submit({"t": "stream", "width": 80, "height": 48})
    assert _run_briefly(loop, lambda: sess.resizes)
    assert sess.resizes == [(80, 48)]


def test_idle_auto_pause_after_last_client_leaves():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=0.1)
    loop.submit({"t": "sim", "cmd": "play", "n": 1})
    assert _run_briefly(loop, lambda: loop.playing is False, timeout=3.0)


# -- the published scene slot ------------------------------------------------


class SceneSession(FakeSession):
    """FakeSession whose scene message tracks a mutable settings dict, the way a real Session's
    tracks ``viz.vis_state``."""

    def __init__(self):
        super().__init__()
        self.vis_state = {"floor": {"alpha": 1.0}}
        self.scene_calls = 0

    def scene_message(self):
        self.scene_calls += 1
        # Real Session.scene_message deep-copies vis_state for exactly this reason.
        return {
            "t": "scene",
            "nu": 0,
            "settings": {k: dict(v) for k, v in self.vis_state.items()},
        }


def test_scene_is_published_before_any_frame_exists():
    """/ws sends the scene as its very first message, which happens long before the first
    frame on a slow-starting model -- so the slot must be seeded at construction."""
    sess = SceneSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    assert loop.latest() is None
    assert loop.scene()["t"] == "scene"


def test_published_scene_is_refreshed_each_tick_and_is_not_a_live_view():
    sess = SceneSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    seeded = loop.scene()
    sess.vis_state["floor"]["alpha"] = 0.25
    # The already-published dict is a snapshot: a later mutation cannot reach into it, which
    # is what stops json.dumps on a request thread from tripping over a concurrent write.
    assert seeded["settings"]["floor"]["alpha"] == 1.0
    # And the next tick republishes, so the slot does not go stale.
    assert _run_briefly(loop, lambda: sess.renders >= 2)
    assert loop.scene()["settings"]["floor"]["alpha"] == 0.25


def test_stop_closes_the_session():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    _run_briefly(loop, lambda: sess.renders >= 1)
    assert sess.closed is True


# -- replay mode --------------------------------------------------------------


def make_source(n_clips=2, n_frames=10, nq=3):
    q = np.arange(n_clips * n_frames * nq, dtype=np.float32)
    return ArrayTrajectorySource(q.reshape(n_clips, n_frames, nq))


@contextlib.contextmanager
def running_replay_loop(source=None, session_cls=FakeSession, **kw):
    session = session_cls()
    loop = SimLoop(session, source=source or make_source(), fps_cap=1000.0, idle_pause_s=None, **kw)
    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        yield session, loop
    finally:
        loop.stop()
        thread.join(timeout=2.0)


def test_replay_mode_is_detected_from_the_source():
    session = FakeSession()
    assert SimLoop(session).replay_mode is False
    assert SimLoop(session, source=make_source()).replay_mode is True


def test_replay_never_steps_physics():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "play": True})
        time.sleep(0.2)
        assert session.steps == 0, "replay mode must not call session.step()"
        assert len(session.qpos_writes) > 1, "replay mode must write qpos each tick"


def test_scrub_writes_that_exact_frame_without_playing():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "frame": 7})
        time.sleep(0.15)
        assert loop.replay_state()["frame"] == 7
        expected = make_source().qpos(0, 7)
        np.testing.assert_array_equal(session.qpos_writes[-1], expected)
        assert loop.playing is False


def test_playback_advances_by_stride():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "frame": 0, "stride": 3, "play": True})
        time.sleep(0.3)
        loop.submit({"t": "replay", "play": False})
        time.sleep(0.1)
        frames = [int(w[0] / 3) for w in session.qpos_writes]  # dof0 encodes frame index
        deltas = {b - a for a, b in zip(frames, frames[1:]) if b > a}
        assert deltas == {3}, f"expected stride-3 advance, saw deltas {deltas}"


def test_playback_loops_within_trim_when_loop_is_true():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "trim": [2, 5], "frame": 2, "loop": True, "play": True})
        # Thread.start() blocks until the child thread has begun running, so the loop's very
        # first tick -- with the constructor's initial `_replay_dirty` -- can write frame 0
        # before this command is even drained. Wait for the trim to actually land, then only
        # look at writes from that point on: what this test cares about is playback staying
        # in bounds, not the one-off startup frame that precedes any command.
        assert wait_until(lambda: loop.replay_state()["in"] == 2)
        start = len(session.qpos_writes)
        time.sleep(0.4)
        st = loop.replay_state()
        assert 2 <= st["frame"] <= 5
        assert all(2 <= int(w[0] / 3) <= 5 for w in session.qpos_writes[start:])


def test_playback_stops_at_out_when_loop_is_false():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "trim": [0, 3], "frame": 0, "loop": False, "play": True})
        time.sleep(0.4)
        assert loop.replay_state()["frame"] == 3
        assert loop.playing is False, "reaching `out` with loop=False must pause"


def test_a_reset_in_replay_mode_re_renders_the_frame_it_reports():
    """``sim reset`` snaps qpos to the rest pose; the playhead does not move.

    Without marking replay dirty, the next publish reports the old frame while the canvas
    shows the rest pose, and nothing re-renders until some later command happens to arrive --
    the same "reported frame != rendered pose" failure ``_advance_replay``'s write-then-publish
    order exists to prevent, reached through a different door.
    """
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "frame": 7})
        assert wait_until(lambda: loop.replay_state()["frame"] == 7)
        # Paused and clean: the loop keeps publishing but stops writing qpos, so any later
        # write can only have been caused by the reset below.
        time.sleep(0.1)
        settled = len(session.qpos_writes)
        time.sleep(0.1)
        assert len(session.qpos_writes) == settled, "paused replay should not keep writing"

        loop.submit({"t": "sim", "cmd": "reset", "n": 1})
        assert wait_until(lambda: session.resets >= 1)
        assert wait_until(lambda: len(session.qpos_writes) > settled), (
            "the reset left the rest pose on screen: nothing re-rendered the replay frame"
        )
        # The pose on screen is frame 7 again...
        np.testing.assert_array_equal(session.pose, make_source().qpos(0, 7))
        # ...and it is still frame 7 that is reported. Reset is about simulation state, not
        # about the cursor, so the playhead deliberately does not move.
        assert loop.replay_state()["frame"] == 7


def test_a_reset_outside_replay_mode_does_not_touch_qpos():
    """The counterpart: with no source, reset must stay exactly what it was."""
    session = FakeSession()
    loop = SimLoop(session, fps_cap=1000.0, idle_pause_s=None)
    loop.submit({"t": "sim", "cmd": "reset", "n": 1})
    assert _run_briefly(loop, lambda: session.resets >= 1)
    assert session.pose == "rest", "nothing should have re-rendered a replay frame"


def test_changing_clip_clamps_frame_and_resets_trim():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "frame": 9})
        time.sleep(0.1)
        loop.submit({"t": "replay", "clip": 1})
        time.sleep(0.1)
        st = loop.replay_state()
        assert st["clip"] == 1
        assert st["out"] == 9, "a new clip resets the trim to its full length"


def test_out_of_range_clip_reports_a_command_error_without_pausing():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "play": True})
        time.sleep(0.1)
        loop.submit({"t": "replay", "clip": 99})
        time.sleep(0.15)
        err = loop.error
        assert err is not None and err["kind"] == "command"
        assert err["paused"] is False
        assert loop.playing is True, "a bad command must not stop playback for everyone"


def test_ghost_toggle_swaps_the_model():
    """``make_source()`` returns a plain ``ArrayTrajectorySource``, which has no ``ghost``
    attribute -- this doubles as proof that a source lacking it is untouched and the toggle
    still works end to end (no AttributeError from the ``hasattr`` guard in ``loop.py``)."""
    with running_replay_loop() as (session, loop):
        assert not hasattr(loop._source, "ghost")
        loop.submit({"t": "replay", "ghost": True})
        time.sleep(0.15)
        assert session.model_swaps == ["alt"]
        loop.submit({"t": "replay", "ghost": False})
        time.sleep(0.15)
        assert session.model_swaps == ["alt", "primary"]


class GhostAwareSource(ArrayTrajectorySource):
    """Stands in for the real fly-side source, which exposes a ``ghost`` flag: off it returns
    101-wide qpos, on it returns ``concat(policy, reference)`` at 202 wide. SimLoop must flip
    this flag in lockstep with ``session.swap_model``, or a tick could observe a ghost-width
    model paired with a non-ghost-width source -- exactly what makes ``Session.set_qpos``
    raise on the next frame."""

    def __init__(self, qpos):
        super().__init__(qpos)
        self.ghost = False


def make_ghost_source(n_clips=2, n_frames=10, nq=3):
    q = np.arange(n_clips * n_frames * nq, dtype=np.float32)
    return GhostAwareSource(q.reshape(n_clips, n_frames, nq))


def test_ghost_toggle_also_flips_a_source_that_supports_it():
    session = FakeSession()
    source = make_ghost_source()
    loop = SimLoop(session, source=source, fps_cap=1000.0, idle_pause_s=None)
    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        loop.submit({"t": "replay", "ghost": True})
        assert wait_until(lambda: session.model_swaps == ["alt"])
        assert source.ghost is True, "the source's own ghost flag must flip too"
        loop.submit({"t": "replay", "ghost": False})
        assert wait_until(lambda: session.model_swaps == ["alt", "primary"])
        assert source.ghost is False
    finally:
        loop.stop()
        thread.join(timeout=2.0)


class BrokenGhostSession(FakeSession):
    """``swap_model`` always fails -- proves the ghost toggle leaves the loop's own flag,
    the source's flag, and the session's active model all agreeing on the PRE-toggle state
    when the swap itself fails, per finding 3's guarantee that no tick may ever observe a
    mismatched model/source pair (relocating that mismatch to a later tick, rather than
    preventing it, would not satisfy that guarantee)."""

    def swap_model(self, which):
        self.model_swaps.append(which)  # record the attempt for the assertion below
        raise RuntimeError("mesh upload failed")


def test_failed_ghost_swap_leaves_flags_and_model_consistent():
    session = BrokenGhostSession()
    source = make_ghost_source()
    loop = SimLoop(session, source=source, fps_cap=1000.0, idle_pause_s=None)
    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        loop.submit({"t": "replay", "ghost": True})
        assert wait_until(lambda: loop.error is not None)
        assert loop.error["kind"] == "command"
        assert loop.error["paused"] is False
        # swap_model was attempted (and recorded before it raised) but never succeeded --
        # and, because loop.py now swaps BEFORE flipping either flag, neither the loop's
        # own bookkeeping nor the source itself moved either. All three still agree on the
        # pre-toggle (non-ghost) state.
        assert session.model_swaps == ["alt"]
        assert loop.replay_state()["ghost"] is False
        assert source.ghost is False

        # A later, ordinary tick must still work normally: nothing was left inconsistent,
        # so a fresh scrub renders without a kind="replay" error (which is what a stale
        # ghost/source mismatch would produce on its very next write).
        writes_before = len(session.qpos_writes)
        loop.submit({"t": "replay", "frame": 3})
        assert wait_until(lambda: len(session.qpos_writes) > writes_before)
        assert loop.error["kind"] == "command", (
            "the original command error must not have been replaced by a new "
            "kind='replay' failure -- nothing should be wrong to fail on"
        )
    finally:
        loop.stop()
        thread.join(timeout=2.0)


def test_frame_meta_carries_replay_state_and_rollout_time():
    # make_source()'s default clip is only 10 frames long, so frame 250 needs a bigger
    # source here -- 10 frames would make frame 250 out of range and raise, never landing
    # in replay_state().
    big_source = make_source(n_frames=300)
    with running_replay_loop(source=big_source, frame_dt=1e-3) as (session, loop):
        loop.submit({"t": "replay", "frame": 250})
        time.sleep(0.15)
        got = loop.wait_for_frame(-1, timeout=1.0)
        assert got is not None
        _seq, _jpeg, meta = got
        assert meta["replay"]["frame"] == 250
        assert meta["replay"]["stride"] == 1
        assert meta["sim_time"] == pytest.approx(0.250)


def _poll_until_published_frame(loop, expected_frame, timeout=2.0):
    """Poll ``wait_for_frame`` until a published ``meta["replay"]["frame"]`` equals
    ``expected_frame`` (or the timeout elapses), returning that meta -- or ``None``.

    Anchors a caller's subsequent ``session.qpos_writes[-1]`` check to a report we have
    directly, freshly observed, rather than pairing it with a SEPARATE, later, unsynchronized
    read of ``qpos_writes`` that a slow test thread could grab one tick late (see the
    now-fixed ``test_reported_frame_matches_the_written_pose_while_playing`` below for why
    that particular shape of race is dangerous: it reproduces the exact ``published ==
    written + stride`` signature of the original bug well enough to pass against broken
    code).
    """
    seq = -1
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = loop.wait_for_frame(seq, timeout=0.3)
        if got is None:
            continue
        seq, _jpeg, meta = got
        replay = meta.get("replay")
        if replay is not None and replay["frame"] == expected_frame:
            return meta
    return None


def test_reported_frame_matches_the_written_pose_while_playing():
    """Regression for: ``_advance_replay`` writes ``qpos(frame)`` and THEN advances
    ``self._frame`` to the next one before returning, so a naive ``_publish`` that re-reads
    ``self._frame`` reports the frame the *next* tick will draw, one stride ahead of what is
    actually on screen right now.

    Compares SEQUENCES, not a single latest-vs-latest snapshot. Reading
    ``meta["replay"]["frame"]`` from ``wait_for_frame()`` and ``session.qpos_writes[-1]``
    from two separate, unsynchronized reads cannot tell "this publish reports the frame
    THIS tick just wrote" from "the loop has already advanced past it since we read the
    published meta" -- and that second, purely-timing-dependent scenario produces EXACTLY
    the same ``published == written + stride`` shape as the original bug, so a single-sample
    comparison can pass against broken code if the reads happen to land unluckily.

    The fix: capture a write count at a point PROVEN quiescent (paused, not dirty -- nothing
    can write again until the next command lands, so there is nothing left to race), collect
    an unbroken run of published frames while playing (gap-checked via ``seq``, so no publish
    -- and so no corresponding write -- was skipped), then compare that whole sequence
    against the correspondingly-positioned slice of the full, now-stable
    ``session.qpos_writes``. Under the bug every entry in that slice would be
    ``published[i] - stride``, which fails at position 0 regardless of exactly when either
    read happens to land relative to the loop thread's tick -- no timing can rescue it.
    """
    session = FakeSession()
    loop = SimLoop(
        session, source=make_source(n_frames=1000), fps_cap=200.0, idle_pause_s=None
    )
    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        # Phase 1: scrub to a known frame and wait for it to become quiescent (paused, and
        # -- since replay_state()["frame"] is only ever set AFTER the write that produced it
        # -- the corresponding write is guaranteed to have already landed too). Nothing
        # further will write until the `play` command below is drained, so the write count
        # captured next is a reliable alignment point, not a guess.
        loop.submit({"t": "replay", "frame": 0})
        assert wait_until(lambda: loop.replay_state()["frame"] == 0 and not loop.playing)
        pre_writes = len(session.qpos_writes)

        # Phase 2: play, and collect an unbroken run of published frames.
        loop.submit({"t": "replay", "play": True})
        published = []
        last_appended_seq = None
        seq = -1
        deadline = time.time() + 2.0
        while time.time() < deadline and len(published) < 12:
            got = loop.wait_for_frame(seq, timeout=0.5)
            if got is None:
                continue
            seq, _jpeg, meta = got
            replay = meta.get("replay")
            if replay is None or not replay["playing"]:
                continue
            if last_appended_seq is not None and seq != last_appended_seq + 1:
                raise AssertionError(
                    f"gap in published seq: {last_appended_seq} -> {seq}; a publish (and "
                    "so possibly a write) was missed, which would make the alignment below "
                    "unreliable -- rerun rather than trust a comparison built on a gap"
                )
            published.append(replay["frame"])
            last_appended_seq = seq

        # Phase 3: stop and let it quiesce before reading the now-stable write list.
        loop.submit({"t": "replay", "play": False})
        time.sleep(0.1)
    finally:
        loop.stop()
        thread.join(timeout=2.0)

    assert len(published) >= 8, "never observed enough published frames while playing"
    written = [int(w[0] / 3) for w in session.qpos_writes]
    aligned = written[pre_writes : pre_writes + len(published)]
    assert aligned == published, (
        f"published sequence {published} does not match the written sequence at the same "
        f"position {aligned} -- the report is racing ahead of (or behind) the actual poses"
    )


def test_step_advances_and_renders_the_next_frame():
    """Regression for: a single ``sim step`` used to write the frame already on screen and
    only silently move the cursor, so pressing Step produced no visible change at all. A
    step's contract is "show me the next frame", so it must move the cursor BEFORE writing
    -- the opposite order from continuous playback.

    Checked against the actual POSE (``session.qpos_writes``), not just the reported state:
    that is exactly the check whose absence let the original defect through review. The pose
    read below is anchored to a freshly-observed publish of the expected frame (see
    ``_poll_until_published_frame``), and safe to read at that point because a step is a
    one-shot write -- nothing advances again until another command arrives, so there is no
    race between "we just saw frame 6 published" and "read the pose it corresponds to".
    """
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "frame": 4, "stride": 2})
        assert wait_until(lambda: loop.replay_state()["frame"] == 4)

        loop.submit({"t": "sim", "cmd": "step", "n": 1})
        meta = _poll_until_published_frame(loop, 6)
        assert meta is not None, "step never published frame 6"
        assert loop.playing is False, "a step must not leave playback running"

        written_frame = int(session.qpos_writes[-1][0] / 3)
        assert written_frame == 6, (
            "one step at stride 2 must WRITE frame 6, not just move an internal cursor to it"
        )


def test_step_with_n_greater_than_one_advances_n_strides_and_writes_once():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "frame": 0, "stride": 1})
        assert wait_until(lambda: loop.replay_state()["frame"] == 0)
        writes_before = len(session.qpos_writes)

        loop.submit({"t": "sim", "cmd": "step", "n": 3})
        assert wait_until(lambda: int(session.qpos_writes[-1][0] / 3) == 3)
        # One tick, one publish -- n=3 must fold into a single write of the FINAL frame,
        # never one write per intermediate stride.
        assert len(session.qpos_writes) == writes_before + 1
        assert loop.playing is False


def test_step_at_out_wraps_to_in_when_looping():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "trim": [2, 5], "frame": 5, "loop": True})
        assert wait_until(lambda: loop.replay_state()["frame"] == 5)

        loop.submit({"t": "sim", "cmd": "step", "n": 1})
        meta = _poll_until_published_frame(loop, 2)
        assert meta is not None, "step never published frame 2"
        assert loop.playing is False
        assert int(session.qpos_writes[-1][0] / 3) == 2


def test_step_at_out_stays_at_out_when_not_looping():
    """Already paused, so this is "nothing moves": no error, no advance past `out`, and the
    same frame is simply re-rendered."""
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "trim": [2, 5], "frame": 5, "loop": False})
        assert wait_until(lambda: loop.replay_state()["frame"] == 5)
        writes_before = len(session.qpos_writes)

        loop.submit({"t": "sim", "cmd": "step", "n": 1})
        assert wait_until(lambda: len(session.qpos_writes) == writes_before + 1)
        assert int(session.qpos_writes[-1][0] / 3) == 5
        assert loop.replay_state()["frame"] == 5
        assert loop.error is None
        assert loop.playing is False


# -- joint locks ---------------------------------------------------------------


def test_locked_joint_holds_while_an_unlocked_neighbour_moves():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "lock", "set": {"joint1": 5.0}})
        loop.submit({"t": "replay", "play": True})
        time.sleep(0.3)
        loop.submit({"t": "replay", "play": False})
        time.sleep(0.1)
        writes = session.qpos_writes
        assert len(writes) > 3
        assert all(w[1] == 5.0 for w in writes[1:]), "the locked dof must hold"
        assert len({w[0] for w in writes}) > 1, "an unlocked dof must still move"


def test_null_lock_freezes_at_the_value_held_when_it_engaged():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "frame": 4})
        time.sleep(0.15)
        held = session.qpos_writes[-1][1]
        loop.submit({"t": "lock", "set": {"joint1": None}})
        loop.submit({"t": "replay", "frame": 8})
        time.sleep(0.15)
        assert session.qpos_writes[-1][1] == held


def test_clear_releases_every_lock():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "lock", "set": {"joint1": 5.0}})
        time.sleep(0.1)
        loop.submit({"t": "lock", "clear": True})
        time.sleep(0.1)
        assert loop.locks == {}
        # Not just the bookkeeping: a fresh write after `clear` must be the RAW frame, not a
        # stale locked array kept alive by something still holding the old value.
        loop.submit({"t": "replay", "frame": 2})
        time.sleep(0.15)
        assert session.qpos_writes[-1][1] == make_source().qpos(0, 2)[1]


def test_locks_ride_the_frame_meta():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "lock", "set": {"joint1": 5.0}})
        time.sleep(0.15)
        got = loop.wait_for_frame(-1, timeout=1.0)
        assert got is not None
        assert got[2]["locks"] == {"joint1": [5.0]}


def test_an_unknown_joint_reports_a_command_error_without_pausing():
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "replay", "play": True})
        time.sleep(0.1)
        loop.submit({"t": "lock", "set": {"not_a_joint": 1.0}})
        time.sleep(0.15)
        err = loop.error
        assert err is not None and err["kind"] == "command" and err["paused"] is False
        assert loop.playing is True


def test_the_joint_map_rebuilds_after_a_ghost_model_swap():
    """``_LOCK_ALT_MODEL`` renames the middle joint from ``joint1`` to ``joint1_ref`` -- so
    this only passes if the loop's cached joint map is actually rebuilt against the NEW
    model, not reused from the primary one (which has no ``joint1_ref`` at all, and would
    reject it as an unknown-joint command error just like the plain unknown-joint case
    above)."""
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "lock", "set": {"joint1_ref": 1.0}})
        time.sleep(0.1)
        assert loop.error is not None and loop.error["kind"] == "command"
        assert loop.locks == {}, "a name only the (not yet active) alt model has must be rejected"

        loop.submit({"t": "replay", "ghost": True})
        time.sleep(0.15)
        assert session.model_swaps == ["alt"]

        loop.submit({"t": "lock", "set": {"joint1_ref": 9.0}})
        # A `lock` command alone does not mark the playhead dirty (only a `replay` command
        # does), so a scrub is needed here to force a fresh write to actually observe the
        # lock take effect -- otherwise this would only be re-checking the write the ghost
        # swap's own dirty flag already produced, before the lock existed.
        loop.submit({"t": "replay", "frame": 5})
        time.sleep(0.15)
        assert loop.locks == {"joint1_ref": [9.0]}
        assert session.qpos_writes[-1][1] == 9.0, "the lock took effect on the next write"


# -- review round 1: locks must survive (or be safely dropped by) a model swap ------------


def test_a_stale_lock_does_not_survive_a_swap_that_drops_its_name():
    """Hole B (review round 1). ``joint1`` is locked, then a swap to ``_LOCK_ALT_MODEL``
    renames it to ``joint1_ref`` -- the old model's ``joint1`` does not exist on the new one
    at all. Before the fix the stale entry survived the swap, and the very next write's
    ``apply_locks`` call raised ``KeyError`` for a joint that no longer exists -- escaping as a
    PAUSED ``kind='replay'`` error that killed playback until the lock was cleared by hand.
    The fix releases every lock on a swap, so the stale entry cannot outlive the model it was
    resolved against."""
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "lock", "set": {"joint1": 5.0}})
        time.sleep(0.1)
        assert loop.locks == {"joint1": [5.0]}

        loop.submit({"t": "replay", "ghost": True})
        time.sleep(0.15)
        assert session.model_swaps == ["alt"]
        assert loop.locks == {}, "a lock naming a joint the new model dropped must not survive"
        assert loop.error is None or loop.error["kind"] != "replay", (
            "a dropped-name lock must never surface as a paused replay error"
        )

        # And playback is not stuck: a fresh write must actually land.
        writes_before = len(session.qpos_writes)
        loop.submit({"t": "replay", "frame": 3})
        assert wait_until(lambda: len(session.qpos_writes) > writes_before)


def test_a_null_lock_and_ghost_toggle_in_one_batch_cannot_pause_playback():
    """Hole A (review round 1). ``joint1`` is paired, via ``ghost_suffix``, with
    ``joint1_ref`` -- which exists only on the WIDE alt model (nq 3 -> 6), not the primary one.
    Submitting the ``ghost`` toggle and a ``None``-valued lock with no sleep between them is
    intended to land both in the SAME drain batch, so ``_apply`` processes the swap (which
    invalidates and lazily rebuilds the joint map) before ``_apply_lock`` runs.

    At that point ``pair_with_suffix`` sees the NEW, wide map (so it also expands to
    ``joint1_ref``), while ``self._last_written_qpos`` is still the frame written on the OLD,
    narrow (nq=3) model -- exactly the mismatch that makes ``resolve_lock_values`` slice past
    the end of the stale array and silently truncate to an empty list for ``joint1_ref``,
    instead of raising. Before the fix that mis-width entry was stored anyway and only failed
    later, inside the write path, as a PAUSED ``kind='replay'`` error. The fix (the width check
    on the resolved value, applied before anything is committed) must catch it HERE instead, as
    a non-pausing ``kind='command'`` error, with nothing partially locked.
    """
    with running_replay_loop(
        session_cls=WideGhostFakeSession, ghost_suffix="_ref"
    ) as (session, loop):
        loop.submit({"t": "replay", "ghost": True})
        loop.submit({"t": "lock", "set": {"joint1": None}})
        time.sleep(0.2)

        assert loop.error is not None, "the mis-width resolve must have been caught"
        assert loop.error["kind"] == "command", (
            f"a lock/swap interaction must never surface as kind='replay' (paused); "
            f"got {loop.error}"
        )
        assert loop.error["paused"] is False
        assert loop.locks == {}, "an all-or-nothing failure must not leave a partial lock"

        # And the loop is not stuck: a later, unrelated write still succeeds.
        writes_before = len(session.qpos_writes)
        loop.submit({"t": "replay", "frame": 3})
        assert wait_until(lambda: len(session.qpos_writes) > writes_before)


def test_lock_set_is_all_or_nothing():
    """One bad name among several good ones in the same ``set`` must commit NOTHING, not the
    valid subset -- otherwise a UI toggling several joints in one message gets partial state
    with no way to tell which half landed."""
    with running_replay_loop() as (session, loop):
        loop.submit({"t": "lock", "set": {"joint1": 5.0, "not_a_joint": 1.0}})
        time.sleep(0.1)
        assert loop.error is not None and loop.error["kind"] == "command"
        assert loop.locks == {}, "a partially-invalid lock.set must not commit the valid names"


# -- export --------------------------------------------------------------------


class FakeExportJob:
    """Stand-in for ExportJob: records what it was handed, never renders."""

    def __init__(self, qpos_frames, cmd):
        self.frames = np.asarray(qpos_frames)
        self.cmd = cmd
        self.started = False
        self.cancelled = False
        self._state = "pending"

    def start(self):
        self.started = True
        self._state = "rendering"

    def cancel(self):
        self.cancelled = True
        self._state = "cancelled"

    def is_alive(self):
        return self._state == "rendering"

    def finish(self):
        self._state = "done"

    def progress(self):
        return {"state": self._state, "done": 0, "total": len(self.frames),
                 "path": "x.mp4", "error": None, "note": None}


@contextlib.contextmanager
def replay_loop_with_export():
    session = FakeSession()
    made = []

    def factory(qpos_frames, cmd):
        job = FakeExportJob(qpos_frames, cmd)
        made.append(job)
        return job

    loop = SimLoop(session, source=make_source(n_frames=10), fps_cap=1000.0,
                   idle_pause_s=None, export_factory=factory)
    thread = threading.Thread(target=loop.run, daemon=True)
    thread.start()
    try:
        yield loop, made
    finally:
        loop.stop()
        thread.join(timeout=2.0)


def test_export_receives_the_trimmed_strided_frames():
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "trim": [2, 8], "stride": 3, "format": "mp4", "crf": 20})
        time.sleep(0.2)
        assert len(made) == 1
        job = made[0]
        assert job.started is True
        # frames 2, 5, 8 -- inclusive of `out`, in original frame units
        assert [int(f[0] / 3) for f in job.frames] == [2, 5, 8]


def test_export_defaults_to_the_current_trim_and_stride():
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "replay", "trim": [1, 4], "stride": 2})
        time.sleep(0.1)
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "format": "mp4", "crf": 20})
        time.sleep(0.2)
        assert [int(f[0] / 3) for f in made[0].frames] == [1, 3]


def test_second_concurrent_export_is_rejected_without_killing_the_first():
    with replay_loop_with_export() as (loop, made):
        cmd = {"t": "export", "width": 64, "height": 48, "fps": 30,
               "format": "mp4", "crf": 20}
        loop.submit(dict(cmd))
        time.sleep(0.15)
        loop.submit(dict(cmd))
        time.sleep(0.15)
        assert len(made) == 1, "a second export must not start"
        err = loop.error
        assert err is not None and "one export at a time" in err["msg"]
        assert err["paused"] is False
        assert made[0].cancelled is False


def test_export_cancel_cancels_the_running_job():
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "format": "mp4", "crf": 20})
        time.sleep(0.15)
        loop.submit({"t": "export_cancel"})
        time.sleep(0.15)
        assert made[0].cancelled is True


def test_a_finished_export_lets_a_new_one_start():
    with replay_loop_with_export() as (loop, made):
        cmd = {"t": "export", "width": 64, "height": 48, "fps": 30,
               "format": "mp4", "crf": 20}
        loop.submit(dict(cmd))
        time.sleep(0.15)
        made[0].finish()
        loop.submit(dict(cmd))
        time.sleep(0.15)
        assert len(made) == 2


def test_the_loop_hands_the_factory_its_own_clip_trim_and_stride():
    """The export request the FACTORY sees must be fully resolved by the loop.

    ``clip`` is not a wire field at all, and ``trim``/``stride`` are optional -- so a factory
    that reads them off the raw command can only guess. It guessed ``clip=0`` and
    ``trim=(0, n_frames-1)`` in strided units, which is how every auto-named export got the
    same filename and quietly overwrote the previous one, and how the provenance sidecar
    recorded ``"clip": null``.

    This drives a real ``export`` command through ``SimLoop`` rather than calling a factory by
    hand with a ``clip`` key no other party ever sets -- which is exactly why the old
    factory-side test stayed green while nothing upheld its contract.
    """
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "replay", "clip": 1, "trim": [2, 8], "stride": 3})
        time.sleep(0.1)
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "format": "mp4", "crf": 20})
        assert wait_until(lambda: made), f"no export ever started: {loop.error}"
        cmd = made[0].cmd
        assert cmd["clip"] == 1, "the factory was not told which clip it is exporting"
        assert cmd["trim"] == [2, 8], "trim must be resolved, in original frame units"
        assert cmd["stride"] == 3


def test_a_client_cannot_name_a_clip_for_export_that_the_loop_is_not_on():
    """The loop is the source of truth for which clip is being exported.

    ``protocol.parse_command`` does not even pass a ``clip`` through on an ``export``, but a
    hand-built command reaching ``submit`` must not be able to make the file's name and
    provenance disagree with the frames actually sliced out of the source.
    """
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "replay", "clip": 1})
        time.sleep(0.1)
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "format": "mp4", "crf": 20, "clip": 0})
        assert wait_until(lambda: made), f"no export ever started: {loop.error}"
        assert made[0].cmd["clip"] == 1


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"trim": [8, 2]}, "ordered"),
        ({"stride": 0}, "stride must be >= 1"),
    ],
)
def test_export_defends_against_a_hand_built_out_of_order_request(bad, message):
    """Defence-in-depth, mirroring ``_apply_replay``'s own trim/stride guards.

    ``protocol.py`` enforces both before a command reaches the loop, so these only fire for a
    caller that constructs an export command directly. Without them a reversed trim exports
    zero frames (``np.stack([])`` raising far from the cause) and ``stride=0`` surfaces as a
    bare "range() arg 3 must not be zero".
    """
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "format": "mp4", "crf": 20, **bad})
        assert wait_until(lambda: loop.error is not None), "the bad request was accepted"
        assert message in loop.error["msg"]
        assert loop.error["paused"] is False  # a bad command must not pause a shared session
        assert not made, "no job may start from a request that failed validation"


def test_export_progress_rides_the_frame_meta():
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "format": "mp4", "crf": 20})
        time.sleep(0.2)
        got = loop.wait_for_frame(-1, timeout=1.0)
        assert got is not None
        _seq, _jpeg, meta = got
        assert meta["export"]["state"] == "rendering"
        assert meta["export"]["total"] == 10


def test_the_export_slice_is_locked_too():
    with replay_loop_with_export() as (loop, made):
        loop.submit({"t": "lock", "set": {"joint1": 5.0}})
        time.sleep(0.1)
        loop.submit({"t": "export", "width": 64, "height": 48, "fps": 30,
                     "format": "mp4", "crf": 20})
        time.sleep(0.2)
        assert len(made) == 1
        assert (made[0].frames[:, 1] == 5.0).all(), "export must inherit the locks"
