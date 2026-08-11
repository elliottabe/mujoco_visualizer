"""Turning a render flag off and back on must return the image to what it was.

`render_with` writes mjRND_SHADOW / mjRND_WIREFRAME / mjRND_SKYBOX onto `renderer.scene.flags`,
and `renderer.update_scene()` does not reset them. The live viewer reuses one renderer for every
frame, so a one-directional write moves a flag permanently: ticking wireframe in the Settings tab
could not be unticked, and `reset_render_settings` restored `vis_state` while the canvas kept
rendering the old flags.

Asserted on PIXELS rather than on `vis_state`, because `vis_state` was always correct -- that is
precisely why eight passing tests and a clean 13/13 root round-trip missed this. The only witness
is the rendered image.
"""

import numpy as np
import pytest

from mujoco_visualizer.serve.session import Session

# A skybox texture is required or the mjRND_SKYBOX leg proves nothing: with no skybox asset the
# flag has no visible effect and the case would pass while broken.
_XML = """
<mujoco>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1=".4 .5 .7" rgb2="0 0 0" width="8" height="8"/>
  </asset>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.2 0 0" size="0.03"/>
    </body>
  </worldbody>
  <actuator><motor name="m0" joint="j0" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


@pytest.fixture
def sess(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_XML)
    s = Session(xml_path=str(p), width=160, height=120)
    yield s
    s.close()


def _max_abs_diff(a, b):
    return int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())


@pytest.mark.parametrize(
    "key,flipped,original,elevation",
    [
        # No single camera puts all three effects in frame, so each case aims its own.
        # The default free camera (az=180, el=-30, fovy=45) has its whole frustum below
        # horizontal, so the sky is never visible and a skybox toggle is a no-op; raising the
        # elevation brings sky in but takes the floor shadow out. Measured, per flag, with a
        # FRESH session each time -- measuring them in sequence on one session confounds them,
        # because these flags are exactly the thing that sticks.
        ("vis_flags.wireframe", True, False, None),   # toggled diff ~162 at the default camera
        ("vis_flags.shadows", False, True, None),     # toggled diff ~100 at the default camera
        ("skybox.show", False, True, 10.0),           # toggled diff ~123 at elevation +10
    ],
)
def test_a_render_flag_put_back_restores_the_image(sess, key, flipped, original, elevation):
    """One renderer, reused -- which is what the live viewer does and what makes this bite."""
    if elevation is not None:
        # SHORT wire name: set_camera translates az/el/dist to vis_state's long names.
        sess.set_camera(el=elevation)
    base = sess.render()

    sess.apply_render({key: flipped})
    changed = sess.render()
    assert _max_abs_diff(changed, base) > 0, (
        f"toggling {key} changed nothing on screen, so the restore below would prove nothing "
        f"-- the fixture is not exercising this flag"
    )

    sess.apply_render({key: original})
    restored = sess.render()
    assert _max_abs_diff(restored, base) == 0, (
        f"{key} was put back but the image did not return: the flag is written "
        f"one-directionally onto renderer.scene.flags, and update_scene() does not reset it, so "
        f"a reused renderer keeps the changed value forever"
    )
