"""Flask app: the viewer page, the scene description, and the frame WebSocket.

Deliberately thin -- it validates, forwards to the SimLoop, and relays frames. All state
lives in Session and all scheduling in SimLoop.

Runs as its OWN process on its OWN port, not inside scripts/vnc_explorer/server.py. That
server runs ``threaded=False``, which serialises every request; one long-lived WebSocket
would block it completely.
"""

import argparse
import json
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, send_from_directory
from flask_sock import Sock

from mujoco_visualizer.serve.loop import SimLoop
from mujoco_visualizer.serve.protocol import CommandError, parse_command
from mujoco_visualizer.serve.session import Session

STATIC = Path(__file__).parent / "static"


def create_app(loop, session, extra_static: Optional[Path] = None) -> Flask:
    """Build the Flask app around a running (or runnable) *loop* and *session*.

    *extra_static* lets a host project (e.g. vnc_explorer) serve its own panel JS from
    ``/ext/<file>`` without vendoring it into this package.
    """
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    sock = Sock(app)

    @app.get("/")
    def index():
        return send_from_directory(str(STATIC), "index.html")

    @app.get("/api/scene")
    def scene():
        return jsonify(session.scene_message())

    if extra_static is not None:

        @app.get("/ext/<path:filename>")
        def ext(filename):
            return send_from_directory(str(extra_static), filename)

    @sock.route("/ws")
    def ws(sock_conn):
        """One connection: relay commands in, frames out.

        Frames are sent as a ``frame_meta`` JSON immediately followed by the binary JPEG.
        A single socket preserves order, so the pairing needs no framing header.
        """
        loop.client_joined()
        sock_conn.send(json.dumps(session.scene_message()))
        last_seq = -1
        last_error = None
        try:
            while True:
                # Non-blocking drain of whatever the client sent since the last frame.
                while True:
                    raw = sock_conn.receive(timeout=0)
                    if raw is None:
                        break
                    try:
                        loop.submit(parse_command(raw))
                    except CommandError as exc:
                        sock_conn.send(
                            json.dumps(
                                {"t": "error", "kind": "command", "msg": str(exc)}
                            )
                        )

                err = getattr(loop, "error", None)
                if err is not None and err is not last_error:
                    sock_conn.send(json.dumps(err))
                    last_error = err

                got = loop.wait_for_frame(last_seq, timeout=0.5)
                if got is None:
                    continue
                last_seq, jpeg, meta = got
                sock_conn.send(json.dumps(meta))
                sock_conn.send(jpeg)
        except Exception:
            # Client vanished mid-send, or the socket closed. Nothing to recover.
            pass
        finally:
            loop.client_left()

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True, help="MuJoCo XML to load")
    ap.add_argument("--anatomy", default=None, help="anatomy YAML/JSON for categories")
    ap.add_argument("--settings", default=None, help="settings JSON to apply at startup")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--substeps", type=int, default=260)
    a = ap.parse_args()

    # Session owns a thread-affine EGL/GL context (see serve/loop.py's module docstring:
    # "physics and rendering must share a thread"). So Session must be *built* -- which
    # creates that context via make_renderer() -- on the exact thread that will later call
    # render(), i.e. the SimLoop thread, not this one. Calling loop.run() directly (instead
    # of loop.start()) inside a thread we spawn ourselves keeps construction and rendering
    # on the same OS thread; a ready Event hands the constructed pair back to this thread
    # once they exist, so Flask never touches an unbuilt Session.
    ready = threading.Event()
    built: dict = {}

    def _build_and_run() -> None:
        session = Session(
            xml_path=a.xml,
            anatomy=a.anatomy,
            settings=a.settings,
            width=a.width,
            height=a.height,
        )
        loop = SimLoop(session, fps_cap=a.fps, substeps_per_frame=a.substeps)
        built["session"] = session
        built["loop"] = loop
        ready.set()
        loop.run()

    sim_thread = threading.Thread(target=_build_and_run, name="SimLoop", daemon=True)
    sim_thread.start()
    ready.wait()
    session, loop = built["session"], built["loop"]

    print(f"viewer ready on http://{a.host}:{a.port}")
    try:
        create_app(loop, session).run(host=a.host, port=a.port, threaded=True)
    finally:
        loop.stop()
        sim_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
