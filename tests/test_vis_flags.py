"""``vis_flags`` entries must reach an actual MjvOption flag.

A flag key that nothing reads is worse than a missing one: every writer (the settings JSON
presets, the serve layer's ``export.tendons`` field, the widget GUIs, a benchmark sweeping
render cost) looks like it is configuring the renderer and is not, and a test that seeds the
key itself and reads it back cannot tell the difference. These tests therefore assert against
the real :class:`Visualizer`'s ``vis_state`` and the ``MjvOption`` it builds, never a
hand-made dict.

``tendon`` in particular defaults to True, because ``mjVIS_TENDON`` is already ON in a bare
``MjvOption()``: any other default would silently change what every existing caller renders.
"""

import mujoco
import numpy as np
import pytest

from mujoco_visualizer import Visualizer
from mujoco_visualizer.render_settings import build_scene_option

# A spatial tendon routed through two sites, so mjVIS_TENDON has something to draw. Without a
# tendon in the model the flag is unobservable and a render-level test would pass vacuously.
_TENDON_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>
    <site name="anchor" pos="-0.3 0 0.4" size="0.01"/>
    <body name="box" pos="0.3 0 0.4">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" rgba="0.2 0.8 0.2 1"/>
      <site name="tip" pos="0 0 0" size="0.01"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="cord" width="0.01" rgba="1 0 0 1">
      <site site="anchor"/>
      <site site="tip"/>
    </spatial>
  </tendon>
</mujoco>
"""


@pytest.fixture
def viz(tmp_path):
    path = tmp_path / "tendon.xml"
    path.write_text(_TENDON_XML)
    v = Visualizer(str(path))
    yield v
    v.close()


def test_mjvis_tendon_defaults_on_in_a_bare_mjvoption():
    """The premise for the default below. If MuJoCo ever flips this, the default here must
    change with it or every existing render silently loses (or gains) its tendons."""
    assert int(mujoco.MjvOption().flags[mujoco.mjtVisFlag.mjVIS_TENDON]) == 1


def test_the_default_vis_state_carries_a_tendon_flag_that_is_on(viz):
    assert viz.vis_state["vis_flags"]["tendon"] is True


def test_the_tendon_flag_reaches_the_scene_option(viz):
    """The claim the old test could not make: the value in ``vis_state`` is what MuJoCo is
    told, read back off the real ``MjvOption`` rather than off the dict it came from."""
    flag = mujoco.mjtVisFlag.mjVIS_TENDON
    assert int(viz._build_scene_option().flags[flag]) == 1

    viz.vis_state["vis_flags"]["tendon"] = False
    assert int(viz._build_scene_option().flags[flag]) == 0

    viz.vis_state["vis_flags"]["tendon"] = True
    assert int(viz._build_scene_option().flags[flag]) == 1


def test_a_settings_dict_with_no_tendon_key_still_leaves_tendons_on():
    """Behaviour preservation for the settings-file path: every preset on disk predates this
    key, so a missing one must mean "unchanged", not "off"."""
    flag = mujoco.mjtVisFlag.mjVIS_TENDON
    assert int(build_scene_option({"vis_flags": {}}).flags[flag]) == 1
    assert int(build_scene_option({"vis_flags": {"tendon": False}}).flags[flag]) == 0


def test_the_two_scene_option_builders_agree_about_tendons(viz):
    """``Visualizer._build_scene_option`` and ``render_settings.build_scene_option`` read the
    same vis_flags dicts (settings JSON, vis_state). A flag honoured by one and ignored by the
    other means one settings preset renders two different ways."""
    flag = mujoco.mjtVisFlag.mjVIS_TENDON
    for value in (True, False):
        viz.vis_state["vis_flags"]["tendon"] = value
        assert (
            int(viz._build_scene_option().flags[flag])
            == int(build_scene_option(viz.vis_state).flags[flag])
            == int(value)
        )


@pytest.mark.gl
def test_turning_tendons_off_removes_them_from_the_rendered_scene(viz):
    """Not just the option object: the drawn scene. Scene geom count is the deterministic
    witness -- pixel differences on this GPU carry a 1/255 noise floor, but geom counts do
    not move on their own."""
    renderer = viz.make_renderer(height=120, width=160)
    try:
        mujoco.mj_forward(viz.model, viz.data)

        viz.render_with(renderer)
        with_tendons = int(renderer.scene.ngeom)

        viz.vis_state["vis_flags"]["tendon"] = False
        frame_off = viz.render_with(renderer)
        without_tendons = int(renderer.scene.ngeom)

        viz.vis_state["vis_flags"]["tendon"] = True
        frame_on = viz.render_with(renderer)
    finally:
        renderer.close()

    assert without_tendons < with_tendons, (
        f"turning tendons off changed nothing in the scene ({with_tendons} geoms either "
        "way) -- the flag is inert"
    )
    # ...and it is visible, not merely absent from the geom list.
    assert np.abs(frame_on.astype(int) - frame_off.astype(int)).max() > 8
