"""ExportJob: renders a handed-in qpos array to a video on its own thread and GL context.

The dimension and container assertions here are regressions for two verified traps:
imageio's default macro_block_size=16 silently rewrites 1920x1080 to 1920x1088, and odd
dimensions crash libx264 with OSError: Broken pipe. A viewer whose purpose is publication
figures must not quietly change the resolution it was asked for.
"""

import copy
import json
import subprocess

import mujoco
import numpy as np
import pytest

from mujoco_visualizer.serve.export import ExportJob, even_dims, mp4_writer_kwargs


def _vis_state_with_tendons(model, tendons):
    """A REALISTIC full ``vis_state`` snapshot (as ``Session.vis_state_snapshot`` would hand
    to a real ``ExportJob``) with ``tendons`` overridden -- never a bare ``{"tendons": ...}``
    dict. ``ExportJob._make_visualizer`` replaces ``viz.vis_state`` WHOLESALE with whatever
    snapshot it is given (``viz.vis_state = copy.deepcopy(self._vis_state)``), so a partial
    dict here would silently wipe every other required key (``alpha``, ``floor``, ...) that
    ``Visualizer.render_with`` reads unconditionally -- this constructs a real ``Visualizer``
    (no GL context; that is only created by ``make_renderer``) purely to get its own
    fully-populated defaults, then overrides just the one group under test."""
    from mujoco_visualizer.visualizer import Visualizer

    viz = Visualizer(model=model)
    vis_state = copy.deepcopy(viz.vis_state)
    viz.close()
    vis_state["tendons"].update(tendons)
    return vis_state


_MODEL_XML = """
<mujoco><worldbody>
  <light pos="0 0 2"/>
  <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1" rgba=".8 .3 .2 1"/></body>
</worldbody></mujoco>
"""

# Two hinge-jointed, motor-actuated bodies -- used by every ctrl_frames test below, since
# _MODEL_XML above has no actuators at all (nu == 0), which cannot exercise a width mismatch.
_ACTUATED_MODEL_XML = """
<mujoco><worldbody>
  <light pos="0 0 2"/>
  <body name="b1"><joint name="j1" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1" rgba=".8 .3 .2 1"/></body>
  <body name="b2"><joint name="j2" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".1 .1 .1" rgba=".2 .3 .8 1"/></body>
</worldbody>
<actuator>
  <motor name="m1" joint="j1"/>
  <motor name="m2" joint="j2"/>
</actuator>
</mujoco>
"""

# A single spatial tendon driven by one motor -- used by the tendon-activation pixel tests.
# rgba alpha=1 and a real (if small) width in the MJCF itself, deliberately, so "tendon
# activation was never applied" (ctrl_frames=None) still renders a REAL, visible tendon at
# its own default appearance -- the two pixel-diff tests below are about activation CHANGING
# that appearance, not about a tendon being invisible without this feature.
_TENDON_MODEL_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <site name="anchor_a" pos="-0.3 0 0.4" size="0.01"/>
    <body name="box_a" pos="-0.3 0 0.6">
      <joint name="slide_a" type="slide" axis="0 0 1"/>
      <geom name="box_a_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_a" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_a" width="0.003" rgba="1 0 0 1">
      <site site="anchor_a"/><site site="tip_a"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="m_a" tendon="t_a" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# Named exactly like tests/serve/test_session.py's own _CTRL_PRIMARY_XML/_CTRL_ALT_XML pair
# (same names, same deliberately scrambled order, same "_ref" un-driven half standing in for
# a reference-ghost overlay's kinematic actuators) -- so the ExportJob-side wiring test below
# exercises the identical "nu doubles, order scrambles" shape that motivated
# build_ctrl_name_map, rather than a fixture whose primary actuators happen to occupy the
# model's first nu slots (which a positional slice would pass by accident).
_CTRL_PRIMARY_NAMES = ["m_a", "m_b", "m_c"]

_CTRL_DOUBLED_ALT_XML = """
<mujoco><worldbody>
  <body name="bc"><joint name="jc" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".05 .05 .05"/></body>
  <body name="ba_ref"><joint name="ja_ref" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".05 .05 .05"/></body>
  <body name="bb"><joint name="jb" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".05 .05 .05"/></body>
  <body name="ba"><joint name="ja" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".05 .05 .05"/></body>
  <body name="bc_ref"><joint name="jc_ref" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".05 .05 .05"/></body>
  <body name="bb_ref"><joint name="jb_ref" type="hinge" axis="0 0 1"/>
    <geom type="box" size=".05 .05 .05"/></body>
</worldbody>
<actuator>
  <motor name="m_c" joint="jc"/>
  <motor name="m_a_ref" joint="ja_ref"/>
  <motor name="m_b" joint="jb"/>
  <motor name="m_a" joint="ja"/>
  <motor name="m_c_ref" joint="jc_ref"/>
  <motor name="m_b_ref" joint="jb_ref"/>
</actuator>
</mujoco>
"""

# Export-side twin of tests/serve/test_session.py's own
# _CTRL_PRIMARY_WITH_UNMATCHED_XML/_CTRL_ALT_ONE_ACTUATOR_XML pair: "m_missing" is a PRIMARY
# actuator name with no counterpart on this model at all. Ordered so the unmatched one is NOT
# first (mirrors the Session fixture's own reasoning), and driving a real tendon so the guard
# test below can observe the SCATTERED result (not just inspect the map), the same way the
# tendon pixel tests above do.
_CTRL_PRIMARY_NAMES_WITH_UNMATCHED = ["m_real", "m_missing"]

_CTRL_MODEL_ONE_REAL_ACTUATOR_WITH_TENDON_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <site name="anchor_r" pos="-0.2 0 0.4" size="0.01"/>
    <body name="box_r" pos="-0.2 0 0.6">
      <joint name="slide_r" type="slide" axis="0 0 1"/>
      <geom name="box_r_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_r" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_r" width="0.003" rgba="1 0 0 1">
      <site site="anchor_r"/><site site="tip_r"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="m_real" tendon="t_r" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
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
def test_a_cancelled_png_sequence_removes_the_frames_it_wrote(tmp_path):
    """§9 promises "partial file removed", and a half-written sequence is a partial file.

    Left behind, those frames are worse than a truncated MP4: ``_render_png`` uses
    ``mkdir(exist_ok=True)`` and always numbers from zero, so a later, shorter export into the
    same directory sits on top of the previous run's higher-numbered frames and anything
    globbing ``frame_*.png`` (ffmpeg, a figure script) silently splices two renders together.
    """
    outdir = tmp_path / "seq"
    job = make_job(tmp_path, frames=400, width=320, height=240, fmt="png", path=outdir)
    job.start()
    while job.progress()["done"] < 2 and job.is_alive():
        pass
    job.cancel()
    job.join(timeout=120)
    prog = job.progress()
    assert prog["state"] == "cancelled"
    assert prog["done"] < 400, "the job finished, so this proves nothing about a partial run"
    assert list(outdir.glob("frame_*.png")) == [], (
        "a cancelled PNG export left its frames on disk"
    )


@pytest.mark.gl
def test_a_failed_png_sequence_removes_the_frames_it_wrote(tmp_path, monkeypatch):
    """Same guarantee on the failure path, which is the one a user does not choose."""
    import imageio

    real_imwrite = imageio.imwrite
    calls = {"n": 0}

    def explode(path, frame, *a, **kw):
        calls["n"] += 1
        if calls["n"] > 2:
            raise OSError("disk full")
        return real_imwrite(path, frame, *a, **kw)

    monkeypatch.setattr(imageio, "imwrite", explode)
    outdir = tmp_path / "seq"
    job = make_job(tmp_path, frames=6, fmt="png", path=outdir)
    job.start()
    job.join(timeout=120)
    prog = job.progress()
    assert prog["state"] == "failed"
    assert "disk full" in prog["error"]
    assert list(outdir.glob("frame_*.png")) == []


@pytest.mark.gl
def test_cleanup_leaves_unrelated_files_in_a_sequence_directory_alone(tmp_path):
    """The sequence directory can be one the user named explicitly, so only the frames THIS
    job wrote are removed -- not the directory, and not its other contents."""
    outdir = tmp_path / "seq"
    outdir.mkdir()
    keep = outdir / "notes.txt"
    keep.write_text("mine")
    job = make_job(tmp_path, frames=400, width=320, height=240, fmt="png", path=outdir)
    job.start()
    while job.progress()["done"] < 2 and job.is_alive():
        pass
    job.cancel()
    job.join(timeout=120)
    assert job.progress()["state"] == "cancelled"
    assert keep.read_text() == "mine"
    assert outdir.is_dir()


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


# --- task 15c: tendon activation + modify_scene_fns reach the export ----------------------


@pytest.mark.gl
def test_tendon_activation_reaches_exported_pixels(tmp_path):
    """The test that matters most for this task: tendon activation must reach the actual
    rendered frame, not merely mutate ``vis_state`` or be accepted as a constructor argument.
    Renders the SAME qpos twice -- once with ``ctrl_frames`` and ``tendons.enabled=True`` at
    a width/colour that cannot be confused with the MJCF's own declared tendon appearance,
    once with no ``ctrl_frames`` at all (tendons therefore never touched, exactly the
    pre-existing behaviour) -- and asserts the two PNG outputs differ in actual pixels."""
    model = mujoco.MjModel.from_xml_string(_TENDON_MODEL_XML)
    qpos = model.qpos0.copy().reshape(1, -1)

    off_dir = tmp_path / "off"
    off = ExportJob(
        model, None, {}, qpos,
        path=off_dir, fmt="png", width=128, height=96, fps=10,
    )
    off.start()
    off.join(timeout=120)
    assert off.progress()["state"] == "done", off.progress()

    on_dir = tmp_path / "on"
    on = ExportJob(
        model, None,
        _vis_state_with_tendons(model, {
            "enabled": True, "max_width": 0.05, "min_width": 0.001,
            "min_alpha": 0.05, "baseline": 0.0, "ctrl_full_scale": 1.0,
        }),
        qpos,
        path=on_dir, fmt="png", width=128, height=96, fps=10,
        ctrl_frames=np.array([[1.0]]),
        primary_actuator_names=["m_a"],
    )
    on.start()
    on.join(timeout=120)
    assert on.progress()["state"] == "done", on.progress()

    import imageio.v2 as imageio

    off_frame = imageio.imread(off_dir / "frame_00000.png")
    on_frame = imageio.imread(on_dir / "frame_00000.png")
    assert not np.array_equal(off_frame, on_frame), (
        "ExportJob accepted ctrl_frames/tendons but tendon activation never reached the "
        "rendered pixels"
    )


@pytest.mark.gl
def test_export_never_mutates_the_callers_model_tendon_state(tmp_path):
    """Tendon-activation visualisation writes ``model.tendon_rgba``/``tendon_width`` in
    place -- exactly the kind of mutation the existing offwidth/offheight no-mutation
    guarantee (see ``test_export_never_mutates_the_callers_model`` above) already protects.
    This proves that guarantee extends to the new mutation: it must land only on ``ExportJob``'s
    own deep copy, never on the ``model`` object the caller passed in."""
    model = mujoco.MjModel.from_xml_string(_TENDON_MODEL_XML)
    orig_rgba = model.tendon_rgba.copy()
    orig_width = model.tendon_width.copy()

    qpos = model.qpos0.copy().reshape(1, -1)
    job = ExportJob(
        model, None,
        _vis_state_with_tendons(model, {
            "enabled": True, "max_width": 0.05, "min_width": 0.001,
            "min_alpha": 0.05, "baseline": 0.0, "ctrl_full_scale": 1.0,
        }),
        qpos,
        path=tmp_path / "out.mp4", width=64, height=48, fps=10,
        ctrl_frames=np.array([[1.0]]),
        primary_actuator_names=["m_a"],
    )
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done"

    assert np.array_equal(model.tendon_rgba, orig_rgba)
    assert np.array_equal(model.tendon_width, orig_width)


@pytest.mark.gl
def test_export_forwards_modify_scene_fns_and_they_reach_the_rendered_pixels(tmp_path):
    """Companion to the tendon test above, for the second overlay this task wires up: a
    ``modify_scene_fns`` callable that draws an extra geom must actually reach the frame the
    renderer produces, not merely be accepted and stored. See also
    ``tests/serve/test_force_arrows_seam.py::
    test_export_now_forwards_scene_modifiers_and_they_reach_the_rendered_pixels`` for the
    seam-level counterpart of this same guarantee."""
    import mujoco

    from mujoco_visualizer.visualizer import add_arrow_to_scene

    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    qpos = model.qpos0.copy().reshape(1, -1)

    def modifier(scene, data=None, frame_idx=0):
        add_arrow_to_scene(scene, [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], radius=0.05)

    plain_dir = tmp_path / "plain"
    plain = ExportJob(
        model, None, {}, qpos, path=plain_dir, fmt="png", width=128, height=96, fps=10,
    )
    plain.start()
    plain.join(timeout=120)
    assert plain.progress()["state"] == "done", plain.progress()

    modded_dir = tmp_path / "modded"
    modded = ExportJob(
        model, None, {}, qpos, path=modded_dir, fmt="png", width=128, height=96, fps=10,
        modify_scene_fns=[modifier],
    )
    modded.start()
    modded.join(timeout=120)
    assert modded.progress()["state"] == "done", modded.progress()

    import imageio.v2 as imageio

    plain_frame = imageio.imread(plain_dir / "frame_00000.png")
    modded_frame = imageio.imread(modded_dir / "frame_00000.png")
    assert not np.array_equal(plain_frame, modded_frame), (
        "ExportJob accepted modify_scene_fns but the callable never reached the rendered "
        "pixels"
    )


# --- mismatched ctrl_frames lengths fail loudly at construction ---------------------------


def test_ctrl_frames_row_count_mismatch_raises_at_construction(tmp_path):
    """ExportJob's chosen contract: a shape mismatch is a construction-time ValueError, never
    a partially-rendered file the caller has to notice and clean up."""
    model = mujoco.MjModel.from_xml_string(_ACTUATED_MODEL_XML)  # nq == 2
    qpos = np.linspace(0, 1, 6 * model.nq, dtype=np.float64).reshape(6, model.nq)
    with pytest.raises(ValueError, match="ctrl_frames has 5 rows but qpos_frames has 6"):
        ExportJob(
            model, None, {}, qpos,
            path=tmp_path / "out.mp4", width=64, height=48, fps=10,
            ctrl_frames=np.zeros((5, model.nu)),
        )


def test_ctrl_frames_row_width_mismatch_raises_at_construction(tmp_path):
    model = mujoco.MjModel.from_xml_string(_ACTUATED_MODEL_XML)  # nu == 2
    qpos = np.linspace(0, 1, 4 * model.nq, dtype=np.float64).reshape(4, model.nq)
    with pytest.raises(ValueError, match="ctrl_frames rows have width 3, expected 2"):
        ExportJob(
            model, None, {}, qpos,
            path=tmp_path / "out.mp4", width=64, height=48, fps=10,
            ctrl_frames=np.zeros((4, 3)),
            primary_actuator_names=["m1", "m2"],
        )


def test_ctrl_frames_must_be_two_dimensional(tmp_path):
    model = mujoco.MjModel.from_xml_string(_ACTUATED_MODEL_XML)
    qpos = np.linspace(0, 1, 4 * model.nq, dtype=np.float64).reshape(4, model.nq)
    with pytest.raises(ValueError, match="ctrl_frames must be 2D"):
        ExportJob(
            model, None, {}, qpos,
            path=tmp_path / "out.mp4", width=64, height=48, fps=10,
            ctrl_frames=np.zeros(4),
        )


def test_ctrl_frames_without_primary_actuator_names_raises_at_construction(tmp_path):
    """Fix round 1: there is no safe default for ``primary_actuator_names`` -- a model whose
    ``nu`` happens to match the primary's but whose actuator order genuinely differs (e.g.
    ``ctrl_frames`` recorded against a different model version with the same actuator count
    but reordered names) cannot be detected from shape alone, so it must be supplied
    explicitly whenever ``ctrl_frames`` is."""
    model = mujoco.MjModel.from_xml_string(_ACTUATED_MODEL_XML)  # nu == 2
    qpos = np.linspace(0, 1, 4 * model.nq, dtype=np.float64).reshape(4, model.nq)
    with pytest.raises(ValueError, match="ctrl_frames requires primary_actuator_names"):
        ExportJob(
            model, None, {}, qpos,
            path=tmp_path / "out.mp4", width=64, height=48, fps=10,
            ctrl_frames=np.zeros((4, 2)),
        )


# --- ctrl_frames are matched by NAME, never by position ------------------------------------


def test_ctrl_frames_are_matched_by_name_not_position_on_a_doubled_export_model(tmp_path):
    """The width problem for real, directly against ``ExportJob``: the model handed to it may
    be the reference-ghost pair (``nu`` doubles, and the surviving names are declared in a
    genuinely different order -- see ``_CTRL_DOUBLED_ALT_XML`` above). A ``ctrl_frames`` row
    ordered by the PRIMARY model's own actuator order must land on the matching name wherever
    that name's id actually sits on THIS job's model, never at a fixed positional prefix.
    Deliberately inspects ``job._ctrl_map`` directly (no GL, no render needed) rather than
    running the whole job -- the pixel-level proof that the map is actually USED lives in
    ``test_tendon_activation_reaches_exported_pixels`` above; this is the proof that the map
    ITSELF is right on a model shaped like the real doubled one."""
    alt = mujoco.MjModel.from_xml_string(_CTRL_DOUBLED_ALT_XML)
    qpos = np.zeros((1, alt.nq), dtype=np.float64)
    job = ExportJob(
        alt, None, {}, qpos,
        path=tmp_path / "out.mp4", width=64, height=48, fps=10,
        ctrl_frames=np.array([[1.0, 2.0, 3.0]]),
        primary_actuator_names=_CTRL_PRIMARY_NAMES,
    )
    alt_id_of = {
        mujoco.mj_id2name(alt, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
        for i in range(alt.nu)
    }
    assert list(job._ctrl_map) == [
        alt_id_of["m_a"], alt_id_of["m_b"], alt_id_of["m_c"],
    ]


@pytest.mark.gl
def test_ctrl_frames_skip_an_unmatched_primary_name_rather_than_misassigning(tmp_path):
    """Export-side twin of ``tests/serve/test_session.py::
    test_ctrl_map_skips_an_unmatched_primary_name_rather_than_wrapping_onto_the_last_actuator``
    -- this task's whole premise is that ``Session`` and ``ExportJob`` apply the SAME matching
    rule, so an unmatched primary name must be handled identically on both paths. ``m_missing``
    has no actuator on this model at all, so ``_ctrl_map``'s entry for it must be -1 and must
    be SKIPPED when scattering ``ctrl_frames`` into this model's own actuator order -- never
    wrapped via numpy's negative-index behaviour onto ``m_real``'s slot, and never a shape
    mismatch either (dropping the mask on the RHS of the scatter assignment, rather than only
    the LHS, makes the LHS index array and the RHS values array different lengths whenever a
    -1 is actually present -- which is exactly the "forgot the mask on one side" slip
    ``build_ctrl_name_map``'s own docstring warns about). Runs a REAL job end to end and reads
    back the job's own tendon width (not the stored map) so either kind of slip is caught:
    a wrong value if regressed some other way, or an outright job failure for this specific
    slip, which a bare ``_ctrl_map`` inspection could not distinguish from success."""
    model = mujoco.MjModel.from_xml_string(_CTRL_MODEL_ONE_REAL_ACTUATOR_WITH_TENDON_XML)
    assert model.nu == 1  # only "m_real" -- "m_missing" has no actuator on this model at all
    qpos = model.qpos0.copy().reshape(1, -1)

    job = ExportJob(
        model, None,
        _vis_state_with_tendons(model, {
            "enabled": True, "max_width": 0.05, "min_width": 0.001,
            "min_alpha": 0.05, "baseline": 0.0, "ctrl_full_scale": 1.0,
        }),
        qpos,
        path=tmp_path / "out.mp4", width=64, height=48, fps=10,
        ctrl_frames=np.array([[1.0, 2.0]]),  # m_real=1.0, m_missing=2.0 (unmatched)
        primary_actuator_names=_CTRL_PRIMARY_NAMES_WITH_UNMATCHED,
    )
    assert list(job._ctrl_map) == [0, -1]

    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done", job.progress()

    t_r = mujoco.mj_name2id(job._model, mujoco.mjtObj.mjOBJ_TENDON, "t_r")
    assert job._model.tendon_width[t_r] == pytest.approx(0.05), (
        "m_real's own ctrl_frames value (1.0, full scale) did not drive its tendon to "
        "max_width -- either m_missing's unmatched value corrupted the scatter, or this "
        "silently did not raise the way a dropped RHS mask would"
    )


# --- sidecar provenance: what overlays were actually applied --------------------------------


@pytest.mark.gl
def test_sidecar_records_tendon_activation_and_scene_modifier_provenance(tmp_path):
    model = mujoco.MjModel.from_xml_string(_ACTUATED_MODEL_XML)
    qpos = np.zeros((2, model.nq), dtype=np.float64)

    def modifier(scene, data=None, frame_idx=0):
        pass

    job = ExportJob(
        model, None,
        _vis_state_with_tendons(model, {
            "enabled": True, "max_width": 0.01, "min_width": 0.001,
            "min_alpha": 0.05, "baseline": 0.0, "ctrl_full_scale": 1.0,
        }),
        qpos,
        path=tmp_path / "out.mp4", width=64, height=48, fps=10,
        ctrl_frames=np.zeros((2, model.nu)),
        primary_actuator_names=["m1", "m2"],
        modify_scene_fns=[modifier, modifier],
    )
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done", job.progress()

    side = json.loads((tmp_path / "out.mp4.json").read_text())
    assert side["ctrl_frames_provided"] is True
    assert side["tendon_activation_applied"] is True
    assert side["scene_modifiers_applied"] == 2


@pytest.mark.gl
def test_sidecar_tendon_activation_applied_is_false_when_tendons_disabled(tmp_path):
    """``ctrl_frames`` being supplied is not the same claim as tendon activation having been
    applied: the ``vis_state`` snapshot's own ``tendons.enabled`` (default False) governs
    whether anything was actually drawn from it."""
    model = mujoco.MjModel.from_xml_string(_ACTUATED_MODEL_XML)
    qpos = np.zeros((2, model.nq), dtype=np.float64)

    job = ExportJob(
        model, None, {},  # tendons not mentioned -> disabled
        qpos,
        path=tmp_path / "out.mp4", width=64, height=48, fps=10,
        ctrl_frames=np.zeros((2, model.nu)),
        primary_actuator_names=["m1", "m2"],
    )
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done", job.progress()

    side = json.loads((tmp_path / "out.mp4.json").read_text())
    assert side["ctrl_frames_provided"] is True
    assert side["tendon_activation_applied"] is False
    assert side["scene_modifiers_applied"] == 0


# --- task 5: explicit actuator_color_fn, the export/preview parity fix --------------------


_TENDON_VIS_ON = {
    "enabled": True, "max_width": 0.05, "min_width": 0.001,
    "min_alpha": 0.05, "baseline": 0.0, "ctrl_full_scale": 1.0,
}


def test_export_honours_an_explicit_actuator_color_fn(tmp_path):
    """Export parity. ExportJob builds its OWN Visualizer from a deep copy, so an
    actuator_color_fn monkey-patched onto the live session's viz could never reach it -- which
    is why the getattr(viz, "actuator_color_fn", None) this replaced was structurally dead, and
    an exported video stayed red while the preview showed colours."""
    model = mujoco.MjModel.from_xml_string(_TENDON_MODEL_XML)
    qpos = model.qpos0.copy().reshape(1, -1)
    job = ExportJob(
        model, None,
        _vis_state_with_tendons(model, _TENDON_VIS_ON),
        qpos,
        path=tmp_path / "coloured", fmt="png", width=128, height=96, fps=10,
        ctrl_frames=np.array([[1.0]]),
        primary_actuator_names=["m_a"],
        actuator_color_fn=lambda name: "#0000ff",
    )
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done", job.progress()
    assert list(job._tendon_base_rgba[0, :3]) == pytest.approx([0.0, 0.0, 1.0])


def test_export_without_a_color_fn_still_uses_the_red_fallback(tmp_path):
    model = mujoco.MjModel.from_xml_string(_TENDON_MODEL_XML)
    qpos = model.qpos0.copy().reshape(1, -1)
    job = ExportJob(
        model, None,
        _vis_state_with_tendons(model, _TENDON_VIS_ON),
        qpos,
        path=tmp_path / "red", fmt="png", width=128, height=96, fps=10,
        ctrl_frames=np.array([[1.0]]),
        primary_actuator_names=["m_a"],
    )
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done", job.progress()
    assert list(job._tendon_base_rgba[0, :3]) == pytest.approx([0.85, 0.15, 0.15])


def test_live_and_export_agree_on_base_rgba_for_the_same_scheme(tmp_path):
    """Spec §8 test 7. The preview and the exported video must derive tendon colours from the
    same array; this is the assertion that fails if the two paths are ever given different
    colour functions or resolve the scheme differently."""
    from mujoco_visualizer.serve.session import Session

    colour_fn = lambda name: "#0000ff"  # noqa: E731 -- one expression, used twice below

    model = mujoco.MjModel.from_xml_string(_TENDON_MODEL_XML)
    live = Session(
        model=model, width=128, height=96,
        actuator_color_schemes={"s": {"color": colour_fn, "group": lambda n: "g"}},
    )
    try:
        live.viz.vis_state["tendons"].update(_TENDON_VIS_ON)
        live.viz.vis_state["tendons"]["color_by"] = "s"
        live.render()
        live_rgba = live._tendon_base_rgba.copy()
    finally:
        live.close()

    export_model = mujoco.MjModel.from_xml_string(_TENDON_MODEL_XML)
    job = ExportJob(
        export_model, None,
        _vis_state_with_tendons(export_model, _TENDON_VIS_ON),
        export_model.qpos0.copy().reshape(1, -1),
        path=tmp_path / "parity", fmt="png", width=128, height=96, fps=10,
        ctrl_frames=np.array([[1.0]]),
        primary_actuator_names=["m_a"],
        actuator_color_fn=colour_fn,
    )
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done", job.progress()
    assert list(job._tendon_base_rgba.flatten()) == pytest.approx(list(live_rgba.flatten()))


# --- the exclusion filter must reach the EXPORT path, not just the preview ------------------
#
# `m_a_ref`/`t_a_ref` stand in for the reference ghost's duplicate half: an actuator that
# exists on the model being rendered but that no primary ctrl column drives. Declared FIRST so
# a positional-prefix assumption cannot pass by accident, and given the MJCF's own opaque
# `rgba="1 0 0 1"` so "the filter did nothing" cannot be mistaken for "the tendon was already
# invisible".
_TENDON_DOUBLED_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <site name="anchor_ref" pos="0.3 0 0.4" size="0.01"/>
    <body name="box_ref" pos="0.3 0 0.6">
      <joint name="slide_ref" type="slide" axis="0 0 1"/>
      <geom name="box_ref_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_ref" pos="0 0 0" size="0.01"/>
    </body>
    <site name="anchor_a" pos="-0.3 0 0.4" size="0.01"/>
    <body name="box_a" pos="-0.3 0 0.6">
      <joint name="slide_a" type="slide" axis="0 0 1"/>
      <geom name="box_a_geom" type="box" size="0.05 0.05 0.05"/>
      <site name="tip_a" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_a_ref" width="0.003" rgba="1 0 0 1">
      <site site="anchor_ref"/><site site="tip_ref"/>
    </spatial>
    <spatial name="t_a" width="0.003" rgba="1 0 0 1">
      <site site="anchor_a"/><site site="tip_a"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="m_a_ref" tendon="t_a_ref" ctrlrange="-1 1"/>
    <motor name="m_a" tendon="t_a" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


@pytest.mark.gl
def test_export_hides_the_tendons_of_actuators_no_ctrl_column_drives(tmp_path):
    """Fix 1. ``Session._rebuild_tendon_colors`` narrows the actuator->tendon map to actuators
    some primary ctrl column drives; ``ExportJob`` rebuilds the same map from the same model and
    used to apply NO such filter, so exporting with the reference ghost active wrote ~260 dim
    but palette-COLOURED duplicate tendons into the video that the preview never showed.

    Expectation if the fix is correct: with a colour fn supplied and a doubled model whose
    ``m_a_ref`` matches no name in ``primary_actuator_names``, the exported frames must show
    ``t_a_ref`` at alpha 0.0 (hidden) while ``t_a`` -- driven at full scale -- is opaque; and the
    whole rendered alpha vector must equal, element for element, what a live ``Session`` on the
    same model produces. Both halves are asserted: the alpha values themselves (so removing
    ``driven_ids`` from ``build_actuator_tendon_map`` fails here with 0.05, the ``min_alpha``
    floor, instead of 0.0) and the live/export parity (so a filter that came back on ONE caller
    only still fails).
    """
    from mujoco_visualizer.serve.session import Session

    colour_fn = lambda name: "#0000ff"  # noqa: E731 -- one expression, used on both paths

    # --- live: the same doubled model, reached through a swap so _ctrl_map is the alt one -----
    live = Session(
        model=mujoco.MjModel.from_xml_string(_TENDON_MODEL_XML),
        alt_model=mujoco.MjModel.from_xml_string(_TENDON_DOUBLED_XML),
        width=128, height=96,
        actuator_color_schemes={"s": {"color": colour_fn, "group": lambda n: "g"}},
    )
    try:
        live.swap_model("alt")
        live.viz.vis_state["tendons"].update(_TENDON_VIS_ON)
        live.viz.vis_state["tendons"]["color_by"] = "s"
        live._vis_ctrl[:] = 0.0
        live._vis_ctrl[mujoco.mj_name2id(
            live.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_a"
        )] = 1.0
        live.render()
        live_alpha = live.model.tendon_rgba[:, 3].copy()
    finally:
        live.close()

    # --- export: a fresh copy of that same doubled model, 2 PNG frames ----------------------
    export_model = mujoco.MjModel.from_xml_string(_TENDON_DOUBLED_XML)
    qpos = np.repeat(export_model.qpos0.copy().reshape(1, -1), 2, axis=0)
    job = ExportJob(
        export_model, None,
        _vis_state_with_tendons(export_model, _TENDON_VIS_ON),
        qpos,
        path=tmp_path / "ghostfilter", fmt="png", width=128, height=96, fps=10,
        ctrl_frames=np.array([[1.0], [1.0]]),  # m_a only; m_a_ref is not a primary name
        primary_actuator_names=["m_a"],
        actuator_color_fn=colour_fn,
    )
    job.start()
    job.join(timeout=120)
    assert job.progress()["state"] == "done", job.progress()
    assert sorted(p.name for p in tmp_path.glob("ghostfilter/*.png")), "no frames were written"

    t_a = mujoco.mj_name2id(job._model, mujoco.mjtObj.mjOBJ_TENDON, "t_a")
    t_a_ref = mujoco.mj_name2id(job._model, mujoco.mjtObj.mjOBJ_TENDON, "t_a_ref")
    assert job._model.tendon_rgba[t_a_ref, 3] == pytest.approx(0.0), (
        "the exported frames drew t_a_ref, whose actuator m_a_ref no primary ctrl column "
        "drives -- its activation is structurally zero, so build_actuator_tendon_map's "
        "driven_ids filter must have hidden it"
    )
    assert job._model.tendon_rgba[t_a, 3] == pytest.approx(1.0), (
        "m_a was driven at full scale, so its own tendon must be opaque -- the filter went "
        "too far if this is dim or hidden"
    )
    assert list(job._model.tendon_rgba[:, 3]) == pytest.approx(list(live_alpha)), (
        "the exported frames' tendon alphas disagree with what the live Session shows for the "
        "same model -- the two paths have drifted again"
    )
