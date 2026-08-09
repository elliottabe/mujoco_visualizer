"""Route-level tests. The app is a thin translation layer, so these stay shallow: the
behaviour lives in Session/SimLoop and is tested there."""

import json

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("flask_sock")

from mujoco_visualizer.serve.app import _ws_loop, create_app  # noqa: E402


_SCENE = {"t": "scene", "nq": 3, "nu": 2, "controls": {"groups": []}}


class StubLoop:
    def __init__(self):
        self.submitted = []
        self.joined = 0
        self.left = 0
        self.error = None

    def submit(self, cmd):
        self.submitted.append(cmd)

    def latest(self):
        return (1, b"\xff\xd8jpeg", {"t": "frame_meta", "seq": 1})

    def wait_for_frame(self, last_seq, timeout=1.0):
        return None

    def scene(self):
        return dict(_SCENE)

    def client_joined(self):
        self.joined += 1

    def client_left(self):
        self.left += 1


class StubSession:
    """A Session the app must never call into: every route and the ws handler run on Flask
    request threads, and Session belongs to the simulation thread."""

    def scene_message(self):
        raise AssertionError(
            "app.py must not call Session.scene_message() from a request thread -- "
            "the scene comes from loop.scene(), published by the simulation thread"
        )


@pytest.fixture
def client():
    app = create_app(StubLoop(), StubSession())
    app.config["TESTING"] = True
    return app.test_client()


def test_index_serves_the_viewer_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<canvas" in resp.data


def test_scene_endpoint_returns_the_scene_message(client):
    resp = client.get("/api/scene")
    assert resp.status_code == 200
    body = json.loads(resp.data)
    assert body["t"] == "scene"
    assert body["nu"] == 2


def test_static_assets_are_served(client):
    assert client.get("/static/viewer.js").status_code == 200
    assert client.get("/static/viewer.css").status_code == 200


def test_scene_endpoint_still_answers_after_the_session_is_closed():
    """Session.close() drops the backend, so a request thread calling scene_message() would
    raise AttributeError and 500. Reading the loop's published slot keeps answering."""

    class ClosedSession(StubSession):
        pass

    loop = StubLoop()
    app = create_app(loop, ClosedSession())
    app.config["TESTING"] = True
    resp = app.test_client().get("/api/scene")
    assert resp.status_code == 200
    assert json.loads(resp.data)["t"] == "scene"


# -- _ws_loop -------------------------------------------------------------------------
#
# Flask's synchronous test client can't drive a real flask_sock connection, so the ws
# handler is unit-tested directly (it was extracted from the route as a standalone function
# for exactly this reason) against a fake socket object exposing send()/receive().


class _Disconnect(Exception):
    """Raised by FakeSockConn.send to end the otherwise-infinite _ws_loop, standing in for
    a real socket close / client vanishing mid-send."""


class FakeSockConn:
    """Stand-in for a flask_sock connection: a scripted receive() queue, a recorded send()
    list, and a configurable "disconnect after N sends" trigger so a test terminates
    deterministically instead of relying on _ws_loop's infinite polling loop."""

    def __init__(self, incoming=(), disconnect_after=None):
        self._incoming = list(incoming)
        self.sent = []
        self.disconnect_after = disconnect_after

    def receive(self, timeout=0):
        if self._incoming:
            return self._incoming.pop(0)
        return None

    def send(self, data):
        if self.disconnect_after is not None and len(self.sent) >= self.disconnect_after:
            raise _Disconnect("simulated client disconnect")
        self.sent.append(data)


class FrameLoop:
    """StubLoop variant that actually produces frame_meta/jpeg pairs (StubLoop's
    wait_for_frame always returns None) and a settable .error, so the frame-pairing and
    error-relay paths in _ws_loop can be exercised."""

    def __init__(self, error=None):
        self.submitted = []
        self.joined = 0
        self.left = 0
        self.error = error
        self._seq = 0

    def submit(self, cmd):
        self.submitted.append(cmd)

    def scene(self):
        return dict(_SCENE)

    def wait_for_frame(self, last_seq, timeout=1.0):
        self._seq += 1
        jpeg = b"\xff\xd8jpeg" + str(self._seq).encode()
        return self._seq, jpeg, {"t": "frame_meta", "seq": self._seq}

    def client_joined(self):
        self.joined += 1

    def client_left(self):
        self.left += 1


def _texts(sent):
    return [json.loads(m) for m in sent if isinstance(m, str)]


def test_ws_sends_scene_first_and_tracks_join_leave_exactly_once():
    loop = FrameLoop()
    conn = FakeSockConn(disconnect_after=1)  # cuts the connection right after the scene send
    _ws_loop(conn, loop, StubSession())
    assert loop.joined == 1
    assert loop.left == 1
    scene = json.loads(conn.sent[0])
    assert scene["t"] == "scene"


def test_ws_sends_frame_meta_then_binary_jpeg():
    loop = FrameLoop()
    conn = FakeSockConn(disconnect_after=3)  # scene, then exactly one meta+jpeg pair
    _ws_loop(conn, loop, StubSession())
    assert len(conn.sent) == 3
    meta = json.loads(conn.sent[1])
    assert meta["t"] == "frame_meta"
    assert conn.sent[2][:2] == b"\xff\xd8"


def test_ws_valid_command_is_forwarded_to_loop():
    loop = FrameLoop()
    conn = FakeSockConn(
        incoming=[json.dumps({"t": "sim", "cmd": "play"})],
        disconnect_after=2,  # scene, then the first frame_meta
    )
    _ws_loop(conn, loop, StubSession())
    assert loop.submitted == [{"t": "sim", "cmd": "play", "n": 1}]


def test_ws_command_error_replies_without_closing_the_connection():
    loop = FrameLoop()
    conn = FakeSockConn(
        incoming=[json.dumps({"t": "nonsense"})],
        disconnect_after=2,  # scene, then the error reply
    )
    _ws_loop(conn, loop, StubSession())
    assert loop.submitted == []  # rejected before ever reaching loop.submit
    texts = _texts(conn.sent)
    assert texts[1]["t"] == "error"
    assert texts[1]["kind"] == "command"
    # The loop kept going after the error reply (it went on to try a frame send next, which
    # is what actually severed conn) rather than treating the CommandError as fatal.
    assert loop.left == 1


def test_ws_error_relay_is_deduped_by_value_not_identity():
    """SimLoop rebuilds a fresh error dict every tick while a failure persists (see
    loop.py's _publish()); the client must see one error message, not one per tick."""
    loop = FrameLoop(error={"t": "error", "kind": "render", "msg": "boom", "paused": True})
    conn = FakeSockConn()

    calls = {"n": 0}
    real_wait = loop.wait_for_frame

    def limited_wait(last_seq, timeout=1.0):
        calls["n"] += 1
        if calls["n"] > 3:
            raise _Disconnect("stop the test")
        # A fresh dict, same content, exactly like SimLoop._publish() rebuilds every tick.
        loop.error = {"t": "error", "kind": "render", "msg": "boom", "paused": True}
        return real_wait(last_seq, timeout)

    loop.wait_for_frame = limited_wait
    _ws_loop(conn, loop, StubSession())

    errors = [m for m in _texts(conn.sent) if m["t"] == "error"]
    assert len(errors) == 1, f"expected exactly one error relay, got {len(errors)}"
    assert loop.left == 1


def test_errors_and_frame_warnings_live_in_separate_dom_elements(client):
    """Pinned at the asset level (there is no JS test harness here).

    frame_meta.warn used to be written into the same #warnbanner as server errors, so the very
    next JPEG's showWarn() hid a diverged/controller/render/command message under one frame
    interval -- the sim stopped with no reason on screen. The three honesty signals
    (#banner for backend_warning, #errbanner for errors, #warnbanner for per-frame warnings)
    must stay distinct elements.
    """
    page = client.get("/").data.decode()
    js = client.get("/static/viewer.js").data.decode()
    css = client.get("/static/viewer.css").data.decode()

    for element_id in ("banner", "errbanner", "warnbanner"):
        assert 'id="{0}"'.format(element_id) in page
    assert 'getElementById("errbanner")' in js
    assert "function showError(" in js
    # The scene-level backend warning still has its own setter, untouched by either of these.
    assert "function showBackendWarning(" in js
    # And the error branch no longer routes through the per-frame warning banner.
    assert "showWarn(`${msg.kind}" not in js
    assert "#errbanner" in css


# -- /api/clips and /api/series -------------------------------------------------------


class FakeLoop:
    """Minimal loop stub for clip/series tests."""

    def scene(self):
        return {}


class FakeClipInfo:
    def clips(self):
        return {"n_clips": 2, "columns": ["clip", "mean_reward"],
                "rows": [{"clip": 0, "mean_reward": 3.5},
                         {"clip": 1, "mean_reward": 3.9}]}

    def series(self, clip, key):
        if key not in ("reward", "joint_error"):
            raise KeyError(f"unknown series key {key!r}")
        if not 0 <= clip < 2:
            raise IndexError(f"clip {clip} out of range")
        return {"clip": clip, "key": key, "frames": [0, 1], "values": [0.1, 0.2],
                "keys": ["reward", "joint_error"]}


def test_api_clips_returns_the_provider_payload():
    app = create_app(FakeLoop(), None, clip_info=FakeClipInfo())
    resp = app.test_client().get("/api/clips")
    assert resp.status_code == 200
    assert resp.get_json()["n_clips"] == 2


def test_api_series_returns_a_trace():
    app = create_app(FakeLoop(), None, clip_info=FakeClipInfo())
    resp = app.test_client().get("/api/series?clip=1&key=reward")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["clip"] == 1 and body["values"] == [0.1, 0.2]


def test_api_series_rejects_a_bad_key_with_400():
    app = create_app(FakeLoop(), None, clip_info=FakeClipInfo())
    resp = app.test_client().get("/api/series?clip=0&key=nope")
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_series_rejects_a_bad_clip_with_400():
    app = create_app(FakeLoop(), None, clip_info=FakeClipInfo())
    assert app.test_client().get("/api/series?clip=9&key=reward").status_code == 400
    assert app.test_client().get("/api/series?clip=x&key=reward").status_code == 400


def test_clip_routes_404_without_a_provider():
    app = create_app(FakeLoop(), None)
    assert app.test_client().get("/api/clips").status_code == 404
    assert app.test_client().get("/api/series?clip=0&key=reward").status_code == 404
