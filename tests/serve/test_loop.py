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


def test_bad_command_pauses_and_reports_instead_of_killing_the_thread():
    """A malformed/rejected command must pause and report, exactly like the physics-side
    errors above -- not just get logged while playing silently continues."""

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
        assert loop.playing is False
        assert loop.is_alive() is True
        renders_at_error = sess.renders
        assert wait_until(lambda: sess.renders > renders_at_error)


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


def test_stop_closes_the_session():
    sess = FakeSession()
    loop = SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)
    _run_briefly(loop, lambda: sess.renders >= 1)
    assert sess.closed is True
