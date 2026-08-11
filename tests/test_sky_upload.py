"""The skybox upload signal must survive a caller that applies settings before ``render_with``.

``_apply_sky_props`` regenerates ``model.tex_data`` and latches ``_sky_fingerprint`` together,
and ``render_with`` is the only site that calls ``mjr_uploadTexture``. Before this fix, it
uploaded only when its OWN ``_apply_all()`` call returned True -- so any other caller that ran
``_apply_all()`` first (``Visualizer.load_settings`` does, and ``Session.reset_render_settings``
used to) consumed the single True the fingerprint would ever produce, and the regenerated sky
never reached the GPU.

No fixture in ``test_vis_flags.py`` builds a model with a skybox texture, so this lives in its
own module rather than bolted onto that file's tendon-only fixture.
"""

import mujoco
import pytest

from mujoco_visualizer import Visualizer

# The texture MUST be named "skybox" -- that is the name render_settings looks up.
_SKYBOX_XML = """
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


def _make_viz_with_skybox(tmp_path):
    """Build a Visualizer whose model actually declares a skybox texture.

    Without this, ``_apply_sky_props`` returns early at the ``_skybox_tex_id < 0`` guard and
    any test built on top of it passes vacuously. Asserting here means a missing texture can
    never pass silently.
    """
    path = tmp_path / "skybox.xml"
    path.write_text(_SKYBOX_XML)
    viz = Visualizer(str(path))
    assert viz._skybox_tex_id >= 0, "model has no 'skybox' texture -- test would prove nothing"
    return viz


@pytest.mark.gl
def test_a_sky_change_applied_outside_render_with_still_uploads(tmp_path):
    """A caller that applies settings itself must not consume the one upload signal.

    `_apply_sky_props` regenerates `model.tex_data` and latches `_sky_fingerprint` together, and
    `render_with` is the only site that calls `mjr_uploadTexture`. So when ANY other caller runs
    `_apply_all()` first -- `Visualizer.load_settings` does, and `Session.reset_render_settings`
    used to -- the fingerprint already matches by the time `render_with` asks, its `_apply_all()`
    returns False, and the regenerated sky is never uploaded to the live context. The canvas then
    keeps rendering the OLD sky indefinitely, while `vis_state` and `model.tex_data` both say it
    changed.

    Asserted on `_sky_needs_upload` rather than on pixels because the flag is the contract:
    pixel comparison needs a real GL context and a skybox large enough to sample reliably, and
    it would pass for the wrong reason on a model whose skybox texture is absent.
    """
    viz = _make_viz_with_skybox(tmp_path)
    renderer = viz.make_renderer(height=64, width=64)
    try:
        viz.render_with(renderer)                  # settle: uploads whatever the launch sky is
        assert viz._sky_needs_upload is False, "a settled render must leave nothing pending"

        viz.vis_state["skybox"]["sky_top"] = "#ff0000"
        viz._apply_all()                           # an outside caller applies, as load_settings does
        assert viz._sky_needs_upload is True, (
            "the regenerated texture was not marked as needing an upload, so render_with will "
            "never send it and the live context keeps the old sky"
        )

        viz.render_with(renderer)
        assert viz._sky_needs_upload is False, "render_with must clear the flag once it uploads"
    finally:
        renderer.close()
        viz.close()
