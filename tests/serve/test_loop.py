"""SimLoop drives a Session on a thread. Tested against a fake session, so none of this
needs GL or a real model."""

import contextlib
import threading
import time

import numpy as np
import pytest

from mujoco_visualizer.serve.loop import SimLoop
from mujoco_visualizer.serve.session import Diverged


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
        self.resizes = []
        self.resets = 0
        self.mode = None
        self.closed = False
        self.controller_rate_hz = None
        self._diverge_after = diverge_after
        self._time = 0.0

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
        pass

    def set_qpos(self, qpos):
        pass

    def reset(self):
        self.resets += 1

    def resize(self, w, h):
        self.resizes.append((w, h))
        self.width, self.height = w, h

    def scene_message(self):
        return {"t": "scene", "nu": 0}

    def close(self):
        self.closed = True


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
