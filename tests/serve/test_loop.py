"""SimLoop drives a Session on a thread. Tested against a fake session, so none of this
needs GL or a real model."""

import contextlib
import threading
import time

import numpy as np
import pytest

from mujoco_visualizer.serve.loop import SimLoop
from mujoco_visualizer.serve.replay import ArrayTrajectorySource
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
        self.qpos_writes = []
        self.model_swaps = []

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

    # -- added for replay mode --
    def set_qpos(self, qpos):
        self.qpos_writes.append(np.asarray(qpos).copy())

    def swap_model(self, which):
        self.model_swaps.append(which)

    def vis_state_snapshot(self):
        return {}

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


# -- replay mode --------------------------------------------------------------


def make_source(n_clips=2, n_frames=10, nq=3):
    q = np.arange(n_clips * n_frames * nq, dtype=np.float32)
    return ArrayTrajectorySource(q.reshape(n_clips, n_frames, nq))


@contextlib.contextmanager
def running_replay_loop(source=None, **kw):
    session = FakeSession()
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
