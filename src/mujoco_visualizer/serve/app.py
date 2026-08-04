"""Flask app: the viewer page, the scene description, and the frame WebSocket.

Deliberately thin -- it validates, forwards to the SimLoop, and relays frames. All state
lives in Session and all scheduling in SimLoop.

Runs as its OWN process on its OWN port, not inside scripts/vnc_explorer/server.py. That
server runs ``threaded=False``, which serialises every request; one long-lived WebSocket
would block it completely.

IMPORTANT -- headless GL: mujoco picks its GL backend from ``MUJOCO_GL`` at ``import mujoco``
time. Running ``python -m mujoco_visualizer.serve.app`` fully imports the parent
``mujoco_visualizer`` package -- and therefore ``mujoco`` -- before a single line of this
module runs, so ``serve/__init__.py``'s own ``os.environ.setdefault("MUJOCO_GL", "egl")`` is
already too late on a bare invocation with no ``MUJOCO_GL`` in the shell. Export it yourself
before launching, e.g.::

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m mujoco_visualizer.serve.app --xml model.xml

Use ``osmesa`` instead of ``egl`` on a machine with no GPU/EGL device. This is deliberately
NOT patched by setting the variable earlier inside ``mujoco_visualizer/__init__.py``: that
would change the default GL backend for every consumer of the package, including the
dearpygui/ipywidgets desktop GUIs and macOS (where EGL does not exist) -- too broad a change
to make as a side effect of this server.
"""

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, send_from_directory
from flask_sock import Sock

from mujoco_visualizer.serve.loop import SimLoop
from mujoco_visualizer.serve.protocol import CommandError, parse_command
from mujoco_visualizer.serve.session import Session

STATIC = Path(__file__).parent / "static"

# How long main() waits for the SimLoop thread to either finish building Session/SimLoop or
# report a construction failure, before giving up and exiting instead of hanging forever.
_STARTUP_TIMEOUT_S = 30.0

_EPILOG = """\
IMPORTANT: on a headless machine, export MUJOCO_GL (and PYOPENGL_PLATFORM) BEFORE running
this command -- mujoco picks its GL backend at import time, which happens before any of this
script's own code runs:

    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m mujoco_visualizer.serve.app --xml model.xml

Use 'osmesa' instead of 'egl' if the machine has no GPU/EGL device.
"""


# Substrings (checked lower-cased) that show up in the various ways a headless-GL-backend
# failure surfaces across mujoco/PyOpenGL/GLFW, keyed off the EXCEPTION TEXT rather than
# os.environ -- see _mujoco_gl_hint for why the environment can't be trusted here.
_GL_ERROR_MARKERS = ("opengl", "egl", "glx", "glfw", "gladloadgl")


def _mujoco_gl_hint(exc: Exception) -> str:
    """If *exc* looks like a headless-GL-backend failure, spell out the fix.

    Deliberately keyed off the exception text, not ``os.environ.get("MUJOCO_GL")``: by the
    time this runs, ``mujoco_visualizer.serve`` (this package) has already been imported,
    which means its own ``__init__.py`` has already run ``os.environ.setdefault("MUJOCO_GL",
    "egl")`` -- too late to change which GL backend ``mujoco`` actually picked (see this
    module's docstring), but early enough that ``os.environ`` now reads ``"egl"`` even on a
    run that crashed for lacking it at the moment that mattered. So checking the environment
    here would silently swallow the exact case this hint exists to catch.
    """
    text = str(exc).lower()
    if not any(marker in text for marker in _GL_ERROR_MARKERS):
        return ""
    return (
        "\nThis looks like a headless GL-backend failure. MUJOCO_GL must be exported BEFORE "
        "this process starts (mujoco picks its backend at import time, before any of this "
        "script's own code runs -- see this module's docstring). Re-run as:\n"
        "    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m mujoco_visualizer.serve.app ...\n"
        "(use 'osmesa' instead of 'egl' if this machine has no GPU/EGL device)"
    )


def _ws_loop(sock_conn, loop, session=None) -> None:
    """One connection: relay commands in, frames out.

    Frames are sent as a ``frame_meta`` JSON immediately followed by the binary JPEG. A
    single socket preserves order, so the pairing needs no framing header.

    Extracted from the ``/ws`` route (rather than left as a closure inside it) so it can be
    unit-tested directly against a fake socket object -- Flask's synchronous test client
    can't drive a real flask_sock connection.

    The scene description comes from ``loop.scene()``, NOT from ``session.scene_message()``:
    this function runs on a Flask request thread while the simulation thread mutates the
    Session, and ``scene_message()`` reads ``viz.vis_state`` (which that thread rewrites via
    apply_render/load_settings/set_camera). Serialising it from here could raise "dictionary
    changed size during iteration" inside ``json.dumps`` -- and this function's bare ``except``
    would silently turn that into a dropped connection. *session* is accepted but unused, for
    call-compatibility.
    """
    loop.client_joined()
    sock_conn.send(json.dumps(loop.scene()))
    last_seq = -1
    # Compared by VALUE (kind, msg), not identity: SimLoop._publish() builds a brand-new
    # error dict every tick while a failure persists (see loop.py), so an identity check
    # would resend the same error up to fps_cap times a second. Resetting to None once the
    # error clears means a *later, distinct* occurrence of the same (kind, msg) is still
    # reported -- only a persisting, unchanged error is suppressed after its first send.
    last_error_sig = None
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
                        json.dumps({"t": "error", "kind": "command", "msg": str(exc)})
                    )

            err = getattr(loop, "error", None)
            if err is None:
                last_error_sig = None
            else:
                sig = (err.get("kind"), err.get("msg"))
                if sig != last_error_sig:
                    sock_conn.send(json.dumps(err))
                    last_error_sig = sig

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


def create_app(loop, session, extra_static: Optional[Path] = None) -> Flask:
    """Build the Flask app around a running (or runnable) *loop* and *session*.

    *extra_static* lets a host project (e.g. vnc_explorer) serve its own panel JS from
    ``/ext/<file>`` without vendoring it into this package.

    *session* is held for ownership/lifetime only. Nothing here calls into it: every route
    runs on a Flask request thread, and the Session belongs to the simulation thread (see
    ``_ws_loop``). Scene data comes from ``loop.scene()``, which the simulation thread
    publishes.
    """
    app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")
    sock = Sock(app)

    @app.get("/")
    def index():
        return send_from_directory(str(STATIC), "index.html")

    @app.get("/api/scene")
    def scene():
        # loop.scene() is seeded at SimLoop construction and republished with every frame, so
        # it answers before the first frame and keeps answering after Session.close() -- where
        # calling session.scene_message() would raise AttributeError on the dropped backend.
        return jsonify(loop.scene())

    if extra_static is not None:

        @app.get("/ext/<path:filename>")
        def ext(filename):
            return send_from_directory(str(extra_static), filename)

    @sock.route("/ws")
    def ws(sock_conn):
        _ws_loop(sock_conn, loop, session)

    return app


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    #
    # Construction failure (bad --xml, a GL error such as the MUJOCO_GL issue documented
    # above, ...) is caught rather than left to kill the thread silently: without this, the
    # main thread's ready.wait() below would block forever with zero diagnostic -- strictly
    # worse than a loud crash.
    ready = threading.Event()
    built: dict = {}

    def _build_and_run() -> None:
        try:
            session = Session(
                xml_path=a.xml,
                anatomy=a.anatomy,
                settings=a.settings,
                width=a.width,
                height=a.height,
            )
            loop = SimLoop(session, fps_cap=a.fps, substeps_per_frame=a.substeps)
        except Exception as exc:
            built["error"] = exc
            return
        else:
            built["session"] = session
            built["loop"] = loop
        finally:
            ready.set()
        loop.run()

    sim_thread = threading.Thread(target=_build_and_run, name="SimLoop", daemon=True)
    sim_thread.start()

    if not ready.wait(timeout=_STARTUP_TIMEOUT_S):
        print(
            f"ERROR: timed out after {_STARTUP_TIMEOUT_S}s waiting for the simulation "
            "session to start; aborting.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if "error" in built:
        exc = built["error"]
        print(f"ERROR: failed to start the simulation session: {exc}", file=sys.stderr)
        hint = _mujoco_gl_hint(exc)
        if hint:
            print(hint, file=sys.stderr)
        raise SystemExit(1)

    session, loop = built["session"], built["loop"]

    print(f"viewer ready on http://{a.host}:{a.port}")
    try:
        create_app(loop, session).run(host=a.host, port=a.port, threaded=True)
    finally:
        loop.stop()
        sim_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
