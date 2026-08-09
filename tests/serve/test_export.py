"""ExportJob: renders a handed-in qpos array to a video on its own thread and GL context.

The dimension and container assertions here are regressions for two verified traps:
imageio's default macro_block_size=16 silently rewrites 1920x1080 to 1920x1088, and odd
dimensions crash libx264 with OSError: Broken pipe. A viewer whose purpose is publication
figures must not quietly change the resolution it was asked for.
"""

import json
import subprocess

import numpy as np
import pytest

from mujoco_visualizer.serve.export import ExportJob, even_dims, mp4_writer_kwargs

_MODEL_XML = """
<mujoco><worldbody>
  <light pos="0 0 2"/>
  <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1" rgba=".8 .3 .2 1"/></body>
</worldbody></mujoco>
"""


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    streams = json.loads(out.stdout)["streams"]
    return next(s for s in streams if s["codec_type"] == "video")


# --- pure helpers ----------------------------------------------------------------

@pytest.mark.parametrize("w,h,expect", [
    (1920, 1080, (1920, 1080)),
    (1921, 1081, (1920, 1080)),
    (641, 480, (640, 480)),
    (3840, 2160, (3840, 2160)),
])
def test_even_dims_rounds_down(w, h, expect):
    got_w, got_h, note = even_dims(w, h)
    assert (got_w, got_h) == expect
    if (w, h) == expect:
        assert note is None
    else:
        assert note and str(expect[0]) in note and str(expect[1]) in note


def test_mp4_writer_kwargs_is_the_verified_recipe():
    kw = mp4_writer_kwargs(fps=30, crf=18)
    assert kw["codec"] == "libx264"
    assert kw["macro_block_size"] == 1
    assert kw["pixelformat"] == "yuv420p"
    assert kw["fps"] == 30
    params = kw["output_params"]
    assert "-movflags" in params and "+faststart" in params
    assert params[params.index("-crf") + 1] == "18"


# --- the job ---------------------------------------------------------------------

def make_job(tmp_path, frames=6, **kw):
    import mujoco

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    qpos = np.linspace(0, 1, frames, dtype=np.float64).reshape(frames, model.nq)
    defaults = dict(path=tmp_path / "out.mp4", width=64, height=48, fps=10)
    defaults.update(kw)
    return ExportJob(model, None, {}, qpos, **defaults)


@pytest.mark.gl
def test_export_writes_a_playable_mp4(tmp_path):
    job = make_job(tmp_path)
    job.start()
    job.join(timeout=120)
    prog = job.progress()
    assert prog["state"] == "done", prog
    assert prog["done"] == prog["total"] == 6
    info = probe(tmp_path / "out.mp4")
    assert info["codec_name"] == "h264"
    assert info["pix_fmt"] == "yuv420p"
    assert (info["width"], info["height"]) == (64, 48)


@pytest.mark.gl
def test_requested_1080_stays_1080_not_1088(tmp_path):
    """The regression that motivated macro_block_size=1."""
    job = make_job(tmp_path, frames=3, width=1920, height=1080, path=tmp_path / "hd.mp4")
    job.start()
    job.join(timeout=300)
    assert job.progress()["state"] == "done"
    info = probe(tmp_path / "hd.mp4")
    assert (info["width"], info["height"]) == (1920, 1080)


@pytest.mark.gl
def test_odd_dimensions_are_rounded_and_reported_not_crashed(tmp_path):
    job = make_job(tmp_path, width=65, height=49, path=tmp_path / "odd.mp4")
    job.start()
    job.join(timeout=120)
    prog = job.progress()
    assert prog["state"] == "done", prog
    assert prog["note"] and "64" in prog["note"]
    info = probe(tmp_path / "odd.mp4")
    assert (info["width"], info["height"]) == (64, 48)


@pytest.mark.gl
def test_moov_precedes_mdat_so_browsers_can_start_playing(tmp_path):
    job = make_job(tmp_path, frames=4, path=tmp_path / "fast.mp4")
    job.start()
    job.join(timeout=120)
    head = (tmp_path / "fast.mp4").read_bytes()
    assert head.find(b"moov") < head.find(b"mdat")


@pytest.mark.gl
def test_cancel_stops_early_and_removes_the_partial_file(tmp_path):
    job = make_job(tmp_path, frames=400, width=320, height=240,
                   path=tmp_path / "cancelled.mp4")
    job.start()
    while job.progress()["done"] < 2 and job.is_alive():
        pass
    job.cancel()
    job.join(timeout=120)
    prog = job.progress()
    assert prog["state"] == "cancelled"
    assert prog["done"] < 400
    assert not (tmp_path / "cancelled.mp4").exists()


@pytest.mark.gl
def test_png_sequence_writes_numbered_frames(tmp_path):
    outdir = tmp_path / "seq"
    job = make_job(tmp_path, frames=3, fmt="png", path=outdir)
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done"
    assert sorted(p.name for p in outdir.glob("*.png")) == [
        "frame_00000.png", "frame_00001.png", "frame_00002.png",
    ]


@pytest.mark.gl
def test_sidecar_records_provenance(tmp_path):
    job = make_job(tmp_path, frames=2, meta={"clip": 42, "stride": 10})
    job.start()
    job.join(timeout=120)
    side = json.loads((tmp_path / "out.mp4.json").read_text())
    assert side["clip"] == 42 and side["stride"] == 10
    assert side["width"] == 64 and side["height"] == 48 and side["fps"] == 10
    assert side["n_frames"] == 2


@pytest.mark.gl
def test_unwritable_destination_fails_the_job_not_the_process(tmp_path):
    job = make_job(tmp_path, path=tmp_path / "nope" / "deep" / "out.mp4")
    job.start()
    job.join(timeout=120)
    prog = job.progress()
    assert prog["state"] == "failed"
    assert prog["error"]


# --- fix round 1 regressions -------------------------------------------------------

@pytest.mark.gl
def test_sidecar_failure_does_not_hang_the_job(tmp_path, monkeypatch):
    """A broken sidecar write must not leave progress() stuck reporting "rendering"."""
    job = make_job(tmp_path, frames=2)
    monkeypatch.setattr(
        job, "_write_sidecar",
        lambda: (_ for _ in ()).throw(OSError("disk full")),
    )
    job.start()
    job.join(timeout=120)
    prog = job.progress()
    assert prog["state"] != "rendering", prog
    # The render itself succeeded and the video is on disk and usable, so a failed
    # provenance write is surfaced as an error rather than flipping the whole job to
    # "failed" -- see the comment in ExportJob.run().
    assert prog["state"] == "done"
    assert prog["error"] and "disk full" in prog["error"]
    assert (tmp_path / "out.mp4").exists()


@pytest.mark.gl
def test_export_never_mutates_the_callers_model(tmp_path):
    """_make_visualizer bumps the offscreen framebuffer size -- it must do so on the
    job's own deep copy, never on the model object the caller passed in."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    orig_offwidth = model.vis.global_.offwidth
    orig_offheight = model.vis.global_.offheight
    assert (orig_offwidth, orig_offheight) == (640, 480)  # the default, and < requested below

    qpos = np.linspace(0, 1, 2, dtype=np.float64).reshape(2, model.nq)
    job = ExportJob(
        model, None, {}, qpos,
        path=tmp_path / "wide.mp4", width=800, height=600, fps=10,
    )
    job.start()
    job.join(timeout=120)

    assert job.progress()["state"] == "done"
    assert model.vis.global_.offwidth == orig_offwidth
    assert model.vis.global_.offheight == orig_offheight
