"""Route-level tests. The app is a thin translation layer, so these stay shallow: the
behaviour lives in Session/SimLoop and is tested there."""

import json

import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("flask_sock")

from mujoco_visualizer.serve.app import create_app  # noqa: E402


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
