"""What a saved settings preset carries beyond the look: the armed camera path and the
engaged joint locks.

Both were unreachable from a preset before this. ``camera_presets`` did round-trip (a saved
preset file holds them), but the PATH over those presets lived in ``Session._camera_path``,
outside ``vis_state``, and ``Session.load_settings`` actively disarmed it -- so a shot built
out of six saved cameras could not be reloaded, only rebuilt. Locks were further out still,
in ``SimLoop._locks``, which the settings file has no view of at all.

The two live at different layers and are tested that way: the camera path against a bare
``Session``, the locks through the ``SimLoop`` that owns them.
"""

import copy
import json

import mujoco
import pytest

from mujoco_visualizer.serve.loop import SimLoop
from mujoco_visualizer.serve.session import RESET_KEYS, Session

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="thorax" pos="0 0 0.5">
      <joint name="j_coxa_T1_left" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      <body name="b1" pos="0.1 0 0">
        <joint name="j_coxa_T1_right" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def sess(tmp_path):
    s = Session(model=mujoco.MjModel.from_xml_string(_XML), width=64, height=48,
                user_settings_dir=tmp_path / "user_settings")
    yield s
    s.close()


def _two_presets(session, **overrides):
    """Two saved camera presets a path can be armed over.

    ``save_camera_preset`` snapshots the CURRENT free camera, so the two are made distinct by
    moving the camera between saves -- a path over two identical presets would interpolate
    correctly no matter what, and could not tell a restored path from a default one.
    """
    session.set_camera(mode="free", azimuth=10.0, elevation=-20.0, distance=0.5, **overrides)
    session.save_camera_preset("shot_a")
    session.set_camera(mode="free", azimuth=90.0, elevation=-5.0, distance=1.5, **overrides)
    session.save_camera_preset("shot_b")


# -- the camera path -------------------------------------------------------------------


def test_a_saved_preset_carries_the_armed_camera_path(sess, tmp_path):
    _two_presets(sess)
    sess.set_camera_path(["shot_a", "shot_b"], weights=[2.0], loop=False)

    dest = sess.save_settings_as("with_path")

    saved = json.loads(dest.read_text())
    assert saved["camera_path"] == {
        "cameras": ["shot_a", "shot_b"], "weights": [2.0], "loop": False
    }
    # The presets it names must travel in the SAME file, or the path describes cameras the
    # loading session has never heard of.
    assert set(saved["camera_presets"]) >= {"shot_a", "shot_b"}


def test_loading_a_preset_re_arms_its_camera_path(sess):
    _two_presets(sess)
    # loop=True closes the path, so a 2-camera loop has TWO segments, not one.
    sess.set_camera_path(["shot_a", "shot_b"], weights=[3.0, 1.0], loop=True)
    sess.save_settings_as("with_path")
    sess.set_camera_path([])
    assert sess.camera_path is None

    sess.load_settings("with_path")

    assert sess.camera_path == {
        "cameras": ["shot_a", "shot_b"], "weights": [3.0, 1.0], "loop": True
    }
    # Armed, not merely recorded: the path must actually drive the rendered cameras.
    assert len(sess.camera_list_for(8)) == 8


def test_loading_a_preset_that_has_no_path_still_disarms(sess):
    """Every preset written before this feature -- including all the bundled ones -- has no
    camera_path key, and must keep behaving exactly as it did: a load replaces the camera
    wholesale, so an armed path over the camera that was just replaced describes a shot the
    user did not ask for."""
    _two_presets(sess)
    sess.save_settings_as("no_path")
    raw = json.loads((sess.user_settings_dir / "no_path.json").read_text())
    del raw["camera_path"]
    (sess.user_settings_dir / "no_path.json").write_text(json.dumps(raw))
    sess.set_camera_path(["shot_a", "shot_b"])

    sess.load_settings("no_path")

    assert sess.camera_path is None


def test_a_path_naming_a_preset_the_file_lacks_is_dropped_not_raised(sess):
    """The colours are applied before the path is re-armed, so a bad path cannot be allowed to
    abort the load -- that would leave a half-applied preset. It is dropped, and the reason is
    returned so the client can say so."""
    _two_presets(sess)
    sess.set_camera_path(["shot_a", "shot_b"])
    sess.save_settings_as("stale_path")
    raw = json.loads((sess.user_settings_dir / "stale_path.json").read_text())
    raw["camera_path"]["cameras"] = ["shot_a", "shot_gone"]
    raw["alpha"] = 0.37
    (sess.user_settings_dir / "stale_path.json").write_text(json.dumps(raw))

    notes = sess.load_settings("stale_path")

    assert sess.camera_path is None
    assert sess.viz.vis_state["alpha"] == 0.37, "the rest of the preset must still apply"
    assert any("shot_gone" in n for n in notes), notes


def test_reset_does_not_delete_the_camera_path_or_the_locks():
    """Same rule camera/camera_presets already follow: Reset restores the LOOK, and a shot the
    user composed is not part of the look."""
    assert "camera_path" not in RESET_KEYS
    assert "locks" not in RESET_KEYS


# -- the locks -------------------------------------------------------------------------


@pytest.fixture
def loop(sess):
    return SimLoop(sess, fps_cap=60, substeps_per_frame=1, idle_pause_s=None)


def _apply(loop, cmd):
    """Drive one command straight through ``SimLoop._apply``.

    The same entry ``run()`` uses per command, called without starting the thread: these
    assertions are about what a command does to lock state, and a running loop would add
    timing to a question that has none.
    """
    loop._apply(cmd)


def test_a_saved_preset_carries_the_engaged_locks(loop, sess):
    _apply(loop, {"t": "lock", "set": {"j_coxa_T1_left": 0.25}})

    _apply(loop, {"t": "settings", "save": "with_locks"})

    saved = json.loads((sess.user_settings_dir / "with_locks.json").read_text())
    assert saved["locks"] == {"j_coxa_T1_left": [0.25]}


def test_loading_a_preset_re_applies_its_locks(loop, sess):
    _apply(loop, {"t": "lock", "set": {"j_coxa_T1_left": 0.25,
                                       "j_coxa_T1_right": -0.5}})
    _apply(loop, {"t": "settings", "save": "with_locks"})
    _apply(loop, {"t": "lock", "clear": True})
    assert loop.locks == {}

    _apply(loop, {"t": "settings", "load": "with_locks"})

    assert loop.locks == {"j_coxa_T1_left": [0.25], "j_coxa_T1_right": [-0.5]}


def test_loading_replaces_the_lock_set_rather_than_merging_it(loop, sess):
    """A preset describes a complete state. Merging would leave a joint locked that the
    preset says nothing about, which is not a state the user ever saved."""
    _apply(loop, {"t": "lock", "set": {"j_coxa_T1_left": 0.25}})
    _apply(loop, {"t": "settings", "save": "only_left"})
    _apply(loop, {"t": "lock", "clear": True})
    _apply(loop, {"t": "lock", "set": {"j_coxa_T1_right": 0.9}})

    _apply(loop, {"t": "settings", "load": "only_left"})

    assert loop.locks == {"j_coxa_T1_left": [0.25]}


def test_a_lock_on_a_joint_this_model_lacks_is_dropped_and_reported(loop, sess):
    """Frozen VALUES are saved, so a preset is tied to the model it was saved on. Loading it
    on another must keep the locks that still resolve and drop the rest by name -- not refuse
    the whole set, which would make a preset useless the moment one joint is renamed."""
    _apply(loop, {"t": "lock", "set": {"j_coxa_T1_left": 0.25}})
    _apply(loop, {"t": "settings", "save": "mixed"})
    raw = json.loads((sess.user_settings_dir / "mixed.json").read_text())
    raw["locks"]["j_wing_hinge"] = [0.1]
    (sess.user_settings_dir / "mixed.json").write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="j_wing_hinge"):
        _apply(loop, {"t": "settings", "load": "mixed"})

    assert loop.locks == {"j_coxa_T1_left": [0.25]}, (
        "the joints that DO exist must still be locked -- the report is a note about what was "
        "dropped, raised after the install, not a refusal of the load"
    )


def test_a_lock_whose_width_does_not_match_the_joint_is_dropped(loop, sess):
    """A hinge takes one value. Three means the preset was saved against a free joint of the
    same name -- installing it would slice past the joint and corrupt its neighbours."""
    _apply(loop, {"t": "lock", "set": {"j_coxa_T1_left": 0.25}})
    _apply(loop, {"t": "settings", "save": "wide"})
    raw = json.loads((sess.user_settings_dir / "wide.json").read_text())
    raw["locks"]["j_coxa_T1_right"] = [0.1, 0.2, 0.3]
    (sess.user_settings_dir / "wide.json").write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="j_coxa_T1_right"):
        _apply(loop, {"t": "settings", "load": "wide"})

    assert loop.locks == {"j_coxa_T1_left": [0.25]}


def test_a_preset_with_no_locks_key_clears_none(loop, sess):
    """Back-compat: every bundled preset predates this key. Loading one must not silently
    release locks the user engaged -- it says nothing about locks, so it changes nothing."""
    _apply(loop, {"t": "settings", "save": "look_only"})
    raw = json.loads((sess.user_settings_dir / "look_only.json").read_text())
    del raw["locks"]
    (sess.user_settings_dir / "look_only.json").write_text(json.dumps(raw))
    _apply(loop, {"t": "lock", "set": {"j_coxa_T1_left": 0.25}})

    _apply(loop, {"t": "settings", "load": "look_only"})

    assert loop.locks == {"j_coxa_T1_left": [0.25]}
