"""Render a trajectory to a video file on a background thread.

Runs on its OWN thread with its OWN GL context and its OWN ``MjModel``. Two threads each
holding an independent EGL context was measured to work: a 4K export and a 640x480 preview
ran concurrently at 63.6 ms and 21.2 ms per frame respectively (47 fps preview, a 3.4x
slowdown from 6.2 ms solo). The spec's "one thread owns physics and rendering" rule is about
sharing ONE context, which this does not do.

The job never reads the rollout file and never touches the preview's state: it is handed a
plain qpos array, a model, and a snapshot of ``vis_state``. That is what lets the user keep
editing colours while a video renders with the look they pressed the button on. ``ExportJob``
deep-copies the model itself in ``__init__`` -- it does not trust the caller to have done so
-- because it mutates the copy's offscreen framebuffer size (see ``_make_visualizer``), and a
mutation reaching a live preview model would corrupt the very thing this design is supposed
to keep editable.

Frames stream to the writer one at a time. ``Visualizer.render_video`` accumulates every
frame and stacks them, which for 1588 frames at 3840x2160 is 39 GB.

Directory contract differs by format, and it is deliberate, not an oversight: MP4 export
requires ``path.parent`` to already exist (a missing one is reported as a failed job, not
silently created -- see ``_render_mp4``); PNG-sequence export creates ``path`` itself as the
sequence's own output directory, because that creation is inherent to what a sequence export
is (see ``_render_png``).

Cleanup after a cancel or a failure differs by format for the same reason: the MP4 is one
file and is deleted outright, while a PNG sequence has only the frames this job wrote deleted
-- the directory may be one the user named and may hold other things (see
``_cleanup_partial``).
"""

import copy
import json
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import mujoco
import numpy as np

__all__ = ["even_dims", "mp4_writer_kwargs", "ExportJob"]


def _actuator_names(model: mujoco.MjModel) -> List[str]:
    """Every actuator name on *model*, in id order. Mirrors ``Session._actuator_names`` (serve/
    session.py) -- duplicated rather than imported, because importing ``serve.session`` here
    would pull in ``Session`` (and its ``simplejpeg``/backend/controls dependencies) for a
    thread that, by design (see the module docstring), never touches a ``Session`` at all.
    An unnamed actuator gets a placeholder rather than ``None``, so it can still occupy a slot
    in :func:`~mujoco_visualizer.visualizer.build_ctrl_name_map`'s name lists without ever
    matching a real name."""
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}"
        for i in range(model.nu)
    ]


def even_dims(width: int, height: int) -> Tuple[int, int, Optional[str]]:
    """Round *width*/*height* down to even, returning a note when anything changed.

    Mandatory before opening an H.264 writer: ``yuv420p`` subsamples chroma by two, so
    libx264 refuses an odd dimension outright -- through imageio that surfaces as
    ``OSError: Broken pipe``, which tells the user nothing. Rounding DOWN rather than up
    guarantees the output never exceeds the requested size.
    """
    w, h = int(width), int(height)
    ew, eh = w - (w % 2), h - (h % 2)
    if (ew, eh) == (w, h):
        return ew, eh, None
    return ew, eh, (
        f"{w}x{h} is not even; H.264 requires even dimensions, so this was rendered at "
        f"{ew}x{eh}"
    )


def mp4_writer_kwargs(fps: float, crf: int = 20) -> Dict:
    """imageio writer kwargs verified (via ffprobe) to play in VSCode, browsers, QuickTime.

    ``macro_block_size=1`` is the load-bearing one: imageio defaults to 16 and silently
    rewrote a requested 1920x1080 to 1920x1088. ``+faststart`` puts the moov atom before
    mdat so a player can begin without fetching the whole file.
    """
    return {
        "fps": fps,
        "codec": "libx264",
        "macro_block_size": 1,
        "pixelformat": "yuv420p",
        "output_params": [
            "-movflags", "+faststart",
            "-profile:v", "high",
            "-crf", str(int(crf)),
        ],
    }


class ExportJob(threading.Thread):
    """Render *qpos_frames* to *path* at *width* x *height*, reporting progress.

    Guarantees:

    - The caller's ``model`` is never mutated. ``__init__`` deep-copies it immediately, so
      even a caller that hands in its live preview model (rather than a copy) is safe --
      this class owns the invariant rather than trusting every call site to uphold it.
    - MP4 export requires ``path.parent`` to already exist; it is not created
      automatically, and a missing one fails the job with ``state="failed"`` rather than
      guessing where to make directories. PNG-sequence export is the opposite: ``path`` is
      the sequence's own directory and IS created.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        anatomy,
        vis_state: Dict,
        qpos_frames: Sequence[np.ndarray],
        *,
        path: Union[str, Path],
        width: int,
        height: int,
        fps: float,
        camera=None,
        fmt: str = "mp4",
        crf: int = 20,
        frame_dt: float = 1e-3,
        meta: Optional[Dict] = None,
        ctrl_frames: Optional[Sequence[np.ndarray]] = None,
        primary_actuator_names: Optional[Sequence[str]] = None,
        modify_scene_fns: Optional[Sequence[Callable]] = None,
    ):
        super().__init__(name="ExportJob", daemon=True)
        # Deep-copy rather than trust the caller: _make_visualizer mutates this model's
        # offscreen framebuffer size, and a bare reference to a live preview model would
        # let that mutation reach the thing this job exists to leave editable. Cheap next
        # to the render itself, and it makes the no-mutation guarantee true unconditionally
        # rather than "true if every caller remembers to copy first". Tendon-activation
        # visualisation (see _apply_tendon_activation_frame below) mutates
        # model.tendon_rgba/tendon_width, and this same deep copy is what confines that
        # mutation to this job's own model rather than the caller's.
        self._model = copy.deepcopy(model)
        self._anatomy = anatomy
        self._vis_state = vis_state
        self._frames = np.asarray(qpos_frames, dtype=np.float64)
        self._path = Path(path)
        self._width, self._height, self._note = even_dims(width, height)
        self._fps = float(fps)
        self._camera = camera
        self._fmt = fmt
        self._crf = int(crf)
        # Accepted but not yet consumed by any renderer here -- kept for forward
        # compatibility with a later task's per-frame overlay that needs real elapsed time
        # (e.g. a timestamp burned into the frame), and recorded in the sidecar below since
        # it's free provenance.
        self._frame_dt = float(frame_dt)
        self._meta = dict(meta or {})

        # Extra per-frame scene decoration (e.g. recorded force-sensor arrows), forwarded
        # verbatim to Visualizer.render_with's existing modify_scene_fns -- see
        # Session.scene_modifiers for the identical LIVE-path seam this mirrors. A plain list
        # copy (or None), never the caller's own list object: this job runs on its own thread
        # and must not observe the caller mutating its list mid-export any more than it may
        # observe vis_state changing (see vis_state's own deepcopy in _make_visualizer).
        self._modify_scene_fns = (
            list(modify_scene_fns) if modify_scene_fns is not None else None
        )

        # ctrl_frames is VISUALISATION-ONLY, exactly like Session._vis_ctrl (see
        # Session.set_qpos's docstring for why data.ctrl is never written from a recorded
        # ctrl during replay): it drives _apply_tendon_activation_frame below and nothing
        # else. It never reaches data.ctrl and is never read by mj_forward.
        #
        # Fail loudly at construction, not partway through a render: a job that discovers a
        # shape mismatch on frame 400 of 1000 would have already written 400 frames of a file
        # a caller then has to know to distrust or clean up. Two distinct checks, both against
        # what the caller actually supplied wrong (row count vs qpos_frames; row width vs the
        # primary actuator names), so the message names the real mismatch rather than a single
        # generic "shape wrong".
        if ctrl_frames is not None:
            ctrl_arr = np.asarray(ctrl_frames, dtype=np.float64)
            if ctrl_arr.ndim != 2:
                raise ValueError(
                    "ExportJob: ctrl_frames must be 2D (n_frames, n_actuators); got shape "
                    f"{ctrl_arr.shape}"
                )
            if len(ctrl_arr) != len(self._frames):
                raise ValueError(
                    f"ExportJob: ctrl_frames has {len(ctrl_arr)} rows but qpos_frames has "
                    f"{len(self._frames)} -- they must be parallel, one row per exported frame"
                )
            # The order a ctrl_frames row is assumed to arrive in: PRIMARY-actuator order,
            # exactly like Session._primary_actuator_names -- never this job's own (possibly
            # doubled, differently-ordered) self._model. Defaults to self._model's own
            # actuator names for the common case of exporting from a model that IS the
            # primary model (no reference-ghost overlay active), where "primary order" and
            # "this model's order" are the same thing by construction.
            primary_names = (
                list(primary_actuator_names)
                if primary_actuator_names is not None
                else _actuator_names(self._model)
            )
            if ctrl_arr.shape[1] != len(primary_names):
                raise ValueError(
                    f"ExportJob: ctrl_frames rows have width {ctrl_arr.shape[1]}, expected "
                    f"{len(primary_names)} (len(primary_actuator_names) if given, else this "
                    "model's own actuator count)"
                )
            from mujoco_visualizer.visualizer import build_ctrl_name_map

            # Matched by NAME against self._model -- never by position. self._model may be
            # the reference-ghost pair (nu doubles, 272 -> 544 on the real models), so a
            # primary-ordered ctrl row cannot be trusted to already line up with this job's
            # own actuator order; see build_ctrl_name_map's own docstring for the real
            # data-corruption bug a positional assumption caused on this branch.
            self._ctrl_frames = ctrl_arr
            self._ctrl_map = build_ctrl_name_map(primary_names, self._model)
        else:
            self._ctrl_frames = None
            self._ctrl_map = None

        self._lock = threading.Lock()
        self._state = "pending"
        self._done = 0
        self._error: Optional[str] = None
        self._cancel = threading.Event()
        # Every PNG this job actually wrote, so a cancel/failure can remove exactly its own
        # frames and nothing else (see _cleanup_partial). Touched only by the job thread.
        self._png_written: List[Path] = []

    # -- public surface ------------------------------------------------------------

    def cancel(self) -> None:
        self._cancel.set()

    def progress(self) -> Dict:
        with self._lock:
            return {
                "state": self._state,
                "done": self._done,
                "total": int(len(self._frames)),
                "path": str(self._path),
                "error": self._error,
                "note": self._note,
            }

    # -- the work -----------------------------------------------------------------

    def run(self) -> None:
        try:
            self._set_state("rendering")
            if self._fmt == "png":
                self._render_png()
            else:
                self._render_mp4()
        except Exception as exc:  # noqa: BLE001 - a job must never kill the process
            self._fail(f"{type(exc).__name__}: {exc}")
            self._cleanup_partial()
            return
        # NOTE: a cancel() landing after the last frame has rendered but before this check
        # runs is indistinguishable from "never cancelled" -- the job reports "cancelled"
        # and deletes what is in fact a complete file. Harmless (the caller asked to
        # cancel and gets no file, which is what cancelling means to them) but worth
        # documenting rather than leaving as a silent surprise.
        if self._cancel.is_set():
            self._set_state("cancelled")
            self._cleanup_partial()
            return
        # The render succeeded and the video/frames are already on disk and usable even if
        # the sidecar write below fails (permission error, full disk, unwritable path).
        # Marking a good export "failed" over a missing provenance .json would throw away
        # more than it protects, so a sidecar failure is recorded as an error but the job
        # still reaches "done" -- never left hanging in "rendering" (see Finding 1: without
        # its own try/except, an exception here escaped both surrounding try/except blocks
        # and left progress() reporting "rendering" forever).
        sidecar_error = None
        try:
            self._write_sidecar()
        except Exception as exc:  # noqa: BLE001 - see note above
            sidecar_error = f"sidecar write failed: {type(exc).__name__}: {exc}"
        self._finish_done(sidecar_error)

    def _make_visualizer(self):
        """Build the Visualizer on THIS thread: it creates the GL context.

        ``anatomy`` is passed through rather than re-derived. Without it, a model handed in
        bare gets one auto-derived category per top-level body, and every category colour in
        the snapshotted ``vis_state`` would miss -- an export that silently does not match
        the preview it was launched from.
        """
        from mujoco_visualizer.visualizer import Visualizer

        # mujoco.Renderer refuses to render larger than the model's declared offscreen
        # framebuffer (vis.global_.offwidth/offheight, default 640x480) -- it raises
        # ValueError rather than resizing. A model that doesn't happen to declare a huge
        # buffer would otherwise reject any export above that default, including a 4K
        # export on an ordinary model. Bump it up (never down) to fit what was requested.
        # Safe to mutate in place: self._model is this job's own deep copy (see __init__),
        # never the caller's original object.
        if self._model.vis.global_.offwidth < self._width:
            self._model.vis.global_.offwidth = self._width
        if self._model.vis.global_.offheight < self._height:
            self._model.vis.global_.offheight = self._height

        viz = Visualizer(model=self._model, anatomy=self._anatomy)
        if self._vis_state:
            viz.vis_state = copy.deepcopy(self._vis_state)

        # Tendon-activation visualisation state, built once here rather than per frame --
        # mirrors Session._rebuild_tendon_state, computed against THIS job's own model copy
        # (never the caller's). Skipped entirely when no ctrl_frames were supplied: there is
        # nothing to drive activation from, so this job must behave exactly as it did before
        # this feature existed (no tendon mutation at all).
        if self._ctrl_frames is not None:
            from mujoco_visualizer.visualizer import (
                build_actuator_tendon_map,
                default_tendon_ctrl_full_scale,
            )

            self._tendon_act_to_ten, self._tendon_base_rgba = build_actuator_tendon_map(
                self._model, getattr(viz, "actuator_color_fn", None)
            )
            self._tendon_default_ctrl_full_scale = default_tendon_ctrl_full_scale(
                self._model, self._tendon_act_to_ten
            )
        return viz

    def _apply_tendon_activation_frame(self, viz, frame_idx: int) -> None:
        """Colour/thicken ``self._model``'s muscle tendons for *frame_idx*, honouring the
        ``vis_state['tendons']`` group in the snapshot this job was constructed with -- the
        export counterpart of ``Session._apply_tendon_activation_vis``.

        Unlike the live path, there is no toggling mid-export (``vis_state`` is a single fixed
        snapshot for the whole job -- see ``_make_visualizer``), so there is nothing to
        restore-on-disable here: when the group is off, this method simply does nothing, and
        ``self._model``'s tendon_rgba/tendon_width stay at the values this job's own deep copy
        started with (never the caller's model -- see ``__init__``).
        """
        tendons = viz.vis_state.get("tendons", {})
        if not tendons.get("enabled", False):
            return
        from mujoco_visualizer.visualizer import apply_tendon_activation

        # ctrl_frames is primary-actuator-ordered; scattered into this model's own actuator
        # order via self._ctrl_map (see __init__) -- by NAME, never by position, exactly like
        # Session.set_qpos. An unmatched primary name (-1) is skipped, never wrapped via
        # numpy's negative-index behaviour onto this model's last actuator.
        vis_ctrl = np.zeros(self._model.nu, dtype=np.float64)
        valid = self._ctrl_map >= 0
        vis_ctrl[self._ctrl_map[valid]] = self._ctrl_frames[frame_idx][valid]
        apply_tendon_activation(
            self._model,
            vis_ctrl,
            self._tendon_act_to_ten,
            self._tendon_base_rgba,
            tendon_width=tendons.get("max_width", 0.003),
            tendon_min_width=tendons.get("min_width", 0.0005),
            tendon_alpha_min=tendons.get("min_alpha", 0.05),
            tendon_baseline=tendons.get("baseline", 0.0),
            ctrl_max=tendons.get("ctrl_full_scale", self._tendon_default_ctrl_full_scale),
        )

    def _iter_rendered(self, viz, renderer):
        for i, qpos in enumerate(self._frames):
            if self._cancel.is_set():
                return
            viz.data.qpos[:] = qpos
            mujoco.mj_forward(viz.model, viz.data)
            if self._ctrl_frames is not None:
                self._apply_tendon_activation_frame(viz, i)
            yield viz.render_with(
                renderer,
                camera=self._camera,
                frame_idx=i,
                modify_scene_fns=self._modify_scene_fns,
            )
            with self._lock:
                self._done = i + 1

    def _render_mp4(self) -> None:
        import imageio

        # The parent directory is NOT auto-created here (unlike the PNG sequence dir
        # below, which the job itself owns): a missing destination is deliberately a
        # failure the job reports rather than papers over -- see
        # test_unwritable_destination_fails_the_job_not_the_process.
        viz = self._make_visualizer()
        renderer = viz.make_renderer(height=self._height, width=self._width)
        try:
            with imageio.get_writer(
                str(self._path), **mp4_writer_kwargs(self._fps, self._crf)
            ) as writer:
                for frame in self._iter_rendered(viz, renderer):
                    writer.append_data(frame)
        finally:
            renderer.close()

    def _render_png(self) -> None:
        import imageio

        self._path.mkdir(parents=True, exist_ok=True)
        viz = self._make_visualizer()
        renderer = viz.make_renderer(height=self._height, width=self._width)
        try:
            for i, frame in enumerate(self._iter_rendered(viz, renderer)):
                target = self._path / f"frame_{i:05d}.png"
                imageio.imwrite(str(target), frame)
                # Recorded, not re-derived from self._done, so cleanup deletes exactly the
                # files that exist: _done is incremented by the generator *after* the body
                # above runs, so the two disagree by one frame at any cancel point.
                self._png_written.append(target)
        finally:
            renderer.close()

    def _write_sidecar(self) -> None:
        """Provenance beside the output, so a figure can be re-made without guessing.

        ``tendon_activation_applied``/``scene_modifiers_applied`` are FACTUAL records of what
        this job actually did, not a claim about what the result looks like -- an exported
        video whose sidecar does not say what overlays it carries is exactly the kind of
        artifact that gets misread in a paper six months later. ``tendon_activation_applied``
        is true only when BOTH ``ctrl_frames`` were supplied AND ``vis_state['tendons']
        ['enabled']`` was true in the snapshot this job rendered with -- ctrl_frames alone
        (tendons left disabled) applied nothing, and this field says exactly that.
        """
        tendons_enabled = bool((self._vis_state or {}).get("tendons", {}).get("enabled", False))
        payload = dict(self._meta)
        payload.update({
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "frame_dt": self._frame_dt,
            "format": self._fmt,
            "crf": self._crf,
            "n_frames": int(len(self._frames)),
            "note": self._note,
            "ctrl_frames_provided": self._ctrl_frames is not None,
            "tendon_activation_applied": self._ctrl_frames is not None and tendons_enabled,
            "scene_modifiers_applied": int(len(self._modify_scene_fns))
            if self._modify_scene_fns
            else 0,
        })
        if self._fmt == "png":
            target = self._path / "export.json"
        else:
            target = self._path.with_suffix(self._path.suffix + ".json")
        target.write_text(json.dumps(payload, indent=2, sort_keys=True))

    def _cleanup_partial(self) -> None:
        """A truncated MP4 that plays for two frames is worse than no file.

        The PNG case is not exempt, it is just narrower. ``_render_png`` uses
        ``mkdir(exist_ok=True)`` and writes ``frame_00000.png`` upwards, so a cancelled or
        failed run left its frames behind -- and a later, shorter export into the same
        directory then sat on top of the previous run's higher-numbered frames, so anything
        globbing ``frame_*.png`` (ffmpeg, a figure script) spliced two different renders into
        one sequence without a word. Only the files THIS job wrote are removed: the directory
        itself, the sidecar, and any unrelated contents are left alone, because a sequence
        directory a user pointed at explicitly is not ours to empty.
        """
        if self._fmt == "png":
            for target in self._png_written:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    pass
            self._png_written.clear()
            return
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass

    def _set_state(self, state: str) -> None:
        with self._lock:
            self._state = state

    def _fail(self, message: str) -> None:
        with self._lock:
            self._state = "failed"
            self._error = message

    def _finish_done(self, error: Optional[str]) -> None:
        """Reach the terminal "done" state, optionally with a non-fatal *error* attached.

        Used when the render itself succeeded but the sidecar write did not: the output
        is real and usable, so the state is "done" rather than "failed", but the error is
        still surfaced rather than swallowed.
        """
        with self._lock:
            self._state = "done"
            self._error = error
