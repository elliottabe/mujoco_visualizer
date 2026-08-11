"""``settings_epoch`` counts wholesale ``vis_state`` writes, so a client can tell that a
settings command was applied even when it changed nothing.

Without it, the viewer's `refreshFromServer` -- which polls /api/scene until the `settings`
blob differs -- reports a no-op reset as "the server reported no change within 2.0s". True,
and it reads as a failure. The same misreport applies to re-loading the preset already
selected.
"""

import os

import pytest

from mujoco_visualizer.serve.session import Session

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j0" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
    </body>
  </worldbody>
  <actuator><motor name="m0" joint="j0" ctrlrange="-1 1"/></actuator>
</mujoco>
"""


@pytest.fixture
def sess(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_XML)
    s = Session(xml_path=str(p), width=64, height=64)
    yield s
    s.close()


def test_epoch_starts_at_zero_and_is_published(sess):
    assert sess.settings_epoch == 0
    assert sess.scene_message()["settings_epoch"] == 0


def test_a_no_op_reset_still_bumps_the_epoch(sess):
    """T6. The whole reason the counter exists: `changed` is False here, and the client must
    STILL be able to see that the command was applied."""
    assert sess.reset_render_settings() is False
    assert sess.settings_epoch == 1
    assert sess.scene_message()["settings_epoch"] == 1


def test_a_reset_that_changes_something_bumps_the_epoch_once(sess):
    sess.apply_render({"floor.reflectance": 0.87})
    assert sess.reset_render_settings() is True
    assert sess.settings_epoch == 1


def test_loading_a_preset_bumps_the_epoch(sess):
    """`load_settings` is the other wholesale vis_state write, and re-loading the
    already-selected preset is the same no-op-looking case."""
    sess.load_settings("Default")
    assert sess.settings_epoch == 1
    sess.load_settings("Default")
    assert sess.settings_epoch == 2


def test_apply_render_does_not_bump_the_epoch(sess):
    """A per-key edit is not a wholesale write. The client already sees those through the
    control it just moved; counting them would make the epoch change constantly while a slider
    is being dragged and defeat its purpose as a change signal."""
    sess.apply_render({"floor.reflectance": 0.5})
    assert sess.settings_epoch == 0
