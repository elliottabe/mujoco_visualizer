"""Route-level tests. The app is a thin translation layer, so these stay shallow: the
behaviour lives in Session/SimLoop and is tested there."""

import json

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("flask_sock")

from mujoco_visualizer.serve.app import _ws_loop, create_app  # noqa: E402


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

    def client_joined(self):
        self.joined += 1

    def client_left(self):
        self.left += 1


class StubSession:
    def scene_message(self):
        return {"t": "scene", "nq": 3, "nu": 2, "controls": {"groups": []}}


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
