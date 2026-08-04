"""Renderer reuse: same pixels, far fewer contexts, and the skybox still updates.

Constructing a Renderer re-uploads every mesh to the GPU -- measured 399 ms/frame on the fly
model versus 8.6 ms reused. These tests pin that caching it is not a behaviour change, and
in particular that skybox edits still reach the screen: MuJoCo uploads textures when the
render context is built, so a cached context needs an explicit re-upload.
"""

import mujoco
import numpy as np
import pytest

from mujoco_visualizer import Visualizer
from mujoco_visualizer import visualizer as vmod

# The texture MUST be named "skybox" -- that is the name render_settings looks up.
_XML = """
<mujoco>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="1 0 0" rgb2="0 0 1" width="8" height="48"/>
  </asset>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
    <body name="box" pos="0 0 0.2">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" rgba="0.2 0.8 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def viz(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_XML)
    v = Visualizer(str(p))
    yield v
    v.close()


def _sky_camera(viz):
    """Free camera with the horizon in frame, so the skybox occupies real pixels."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.2]
    cam.distance = 2.0
    cam.azimuth, cam.elevation = 90.0, 5.0
    return cam


def _set_sky(viz, color):
    viz.vis_state["skybox"].update(show=True, sky_top=color, sky_bot=color)


# -- parity ----------------------------------------------------------------


@pytest.mark.gl
def test_render_with_matches_render_frame(viz):
    expected = viz.render_frame(viz.model.qpos0, height=120, width=160)
    renderer = viz.make_renderer(height=120, width=160)
    try:
        viz.data.qpos[:] = viz.model.qpos0
        mujoco.mj_forward(viz.model, viz.data)
        got = viz.render_with(renderer)
    finally:
        renderer.close()
    assert got.shape == (120, 160, 3)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, expected)


@pytest.mark.gl
def test_render_with_does_not_touch_qpos(viz):
    renderer = viz.make_renderer(height=120, width=160)
    try:
        viz.data.qpos[0] = 0.0
        mujoco.mj_forward(viz.model, viz.data)
        low = viz.render_with(renderer)
        viz.data.qpos[0] = 0.4
        mujoco.mj_forward(viz.model, viz.data)
        high = viz.render_with(renderer)
        assert viz.data.qpos[0] == pytest.approx(0.4)
        assert not np.array_equal(low, high)
    finally:
        renderer.close()


@pytest.mark.gl
def test_repeated_render_frame_is_deterministic(viz):
    first = viz.render_frame(viz.model.qpos0, height=64, width=64)
    for _ in range(5):
        again = viz.render_frame(viz.model.qpos0, height=64, width=64)
    np.testing.assert_array_equal(again, first)


# -- caching ---------------------------------------------------------------


@pytest.mark.gl
def test_render_frame_builds_only_one_renderer(viz, monkeypatch):
    """The whole point: 399 ms of mesh upload must happen once, not per call."""
    calls = []
    real = viz.make_renderer

    def counting(height=480, width=640):
        calls.append((height, width))
        return real(height=height, width=width)

    monkeypatch.setattr(viz, "make_renderer", counting)
    for _ in range(4):
        viz.render_frame(viz.model.qpos0, height=120, width=160)
    assert len(calls) == 1


@pytest.mark.gl
def test_changing_resolution_rebuilds_once_and_frees_the_old(viz, monkeypatch):
    calls = []
    real = viz.make_renderer

    def counting(height=480, width=640):
        calls.append((height, width))
        return real(height=height, width=width)

    monkeypatch.setattr(viz, "make_renderer", counting)
    viz.render_frame(viz.model.qpos0, height=120, width=160)
    viz.render_frame(viz.model.qpos0, height=64, width=64)
    viz.render_frame(viz.model.qpos0, height=64, width=64)
    assert calls == [(120, 160), (64, 64)]
    assert viz.render_frame(viz.model.qpos0, height=64, width=64).shape == (64, 64, 3)


@pytest.mark.gl
def test_close_releases_the_cached_renderer(viz):
    viz.render_frame(viz.model.qpos0, height=64, width=64)
    viz.close()
    assert viz._renderer_cache is None
    viz.close()  # idempotent


# -- the correctness trap caching introduces -------------------------------


@pytest.mark.gl
def test_skybox_edits_survive_renderer_reuse(viz):
    """Verified regression: without an explicit re-upload this diff is 0, not 255."""
    cam = _sky_camera(viz)
    _set_sky(viz, "#ff0000")
    red = viz.render_frame(viz.model.qpos0, camera=cam, height=200, width=200)
    _set_sky(viz, "#00ff00")
    green = viz.render_frame(viz.model.qpos0, camera=cam, height=200, width=200)
    assert np.abs(red.astype(int) - green.astype(int)).max() > 200


@pytest.mark.gl
def test_skybox_is_not_regenerated_when_settings_are_unchanged(viz, monkeypatch):
    """Regenerating the texture costs ~4 ms, ~45% of a 640x480 render."""
    calls = []
    real = vmod._make_sky_pixels

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(vmod, "_make_sky_pixels", counting)
    _set_sky(viz, "#123456")
    viz.render_frame(viz.model.qpos0, height=64, width=64)
    before = len(calls)
    for _ in range(5):
        viz.render_frame(viz.model.qpos0, height=64, width=64)
    assert len(calls) == before, "skybox regenerated despite unchanged settings"


@pytest.mark.gl
def test_changing_sky_settings_does_regenerate(viz, monkeypatch):
    calls = []
    real = vmod._make_sky_pixels

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(vmod, "_make_sky_pixels", counting)
    _set_sky(viz, "#111111")
    viz.render_frame(viz.model.qpos0, height=64, width=64)
    n = len(calls)
    _set_sky(viz, "#eeeeee")
    viz.render_frame(viz.model.qpos0, height=64, width=64)
    assert len(calls) == n + 1


def test_apply_all_reports_whether_the_sky_changed(viz):
    """No GL needed: this is the signal render_with uses to decide on a re-upload."""
    _set_sky(viz, "#abcdef")
    assert viz._apply_all() is True
    assert viz._apply_all() is False


# -- render_video DRY refactor ---------------------------------------------


@pytest.mark.gl
def test_render_video_matches_per_frame_rendering(viz):
    qposes = np.tile(viz.model.qpos0, (3, 1))
    qposes[1, 0] = 0.2
    qposes[2, 0] = 0.4
    video = viz.render_video(qposes, height=64, width=64)
    assert video.shape == (3, 64, 64, 3)
    for i, qpos in enumerate(qposes):
        np.testing.assert_array_equal(
            video[i], viz.render_frame(qpos, height=64, width=64)
        )


@pytest.mark.gl
def test_render_video_passes_frame_idx_to_scene_modifiers(viz):
    seen = []

    def spy(scene, data=None, frame_idx=0):
        seen.append(frame_idx)

    qposes = np.tile(viz.model.qpos0, (3, 1))
    viz.render_video(qposes, height=32, width=32, modify_scene_fns=[spy])
    assert seen == [0, 1, 2]
