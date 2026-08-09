"""``settings.save`` writes a NAMED preset into a user-supplied directory -- never into the
package's own bundled ``settings/`` directory. The security shape mirrors ``settings.load``:
a name is checked against ``^[A-Za-z0-9_-]{1,64}$`` before it ever reaches a filesystem call,
because it becomes a filename an ``open(..., 'w')`` on the server creates.

Everything here runs headless: no GL, no jax/mjx.
"""

import copy

import pytest

from mujoco_visualizer.render_settings import PRESET_NAME_RE, list_available_settings
from mujoco_visualizer.serve.session import Session

_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j_coxa_T1_left" type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      <body name="b1" pos="0.1 0 0">
        <joint name="j_coxa_T1_right" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="coxa_T1_left"  joint="j_coxa_T1_left"  ctrlrange="-1 1"/>
    <motor name="coxa_T1_right" joint="j_coxa_T1_right" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def xml(tmp_path):
    p = tmp_path / "m.xml"
    p.write_text(_XML)
    return str(p)


@pytest.fixture
def sess(xml, tmp_path):
    user_dir = tmp_path / "user_settings"
    s = Session(xml_path=xml, width=64, height=64, user_settings_dir=user_dir)
    yield s
    s.close()


# -- the NAME pattern itself (also exercised through parse_command in test_protocol.py) -----


@pytest.mark.parametrize("value", ["my_fig-2", "a", "A0_-", "a" * 64])
def test_preset_name_pattern_accepts_well_formed_names(value):
    assert PRESET_NAME_RE.match(value)


@pytest.mark.parametrize(
    "value",
    ["../x", "a/b", "", "x.json", "a" * 65],
)
def test_preset_name_pattern_rejects_paths_and_out_of_range_names(value):
    assert PRESET_NAME_RE.match(value) is None


# -- list_available_settings -----------------------------------------------------------------


def test_list_available_settings_with_no_user_dir_is_bundled_only():
    entries = list_available_settings()
    assert entries  # the package ships several
    assert all(d["origin"] == "bundled" for d in entries)


def test_list_available_settings_merges_a_user_dir(tmp_path):
    (tmp_path / "my_look.json").write_text("{}")
    entries = list_available_settings(tmp_path)
    names_by_origin = {(d["name"], d["origin"]) for d in entries}
    assert ("my_look", "user") in names_by_origin
    # bundled presets are still listed alongside it
    assert any(origin == "bundled" for _, origin in names_by_origin)


def test_list_available_settings_a_missing_user_dir_does_not_raise(tmp_path):
    missing = tmp_path / "does_not_exist_yet"
    entries = list_available_settings(missing)
    assert all(d["origin"] == "bundled" for d in entries)


def test_name_collision_keeps_both_entries_distinguished_by_origin(tmp_path):
    """Decision: a name collision between a bundled and a user preset is not resolved by
    list_available_settings -- both are listed, distinguished by 'origin'. (Session.load_settings
    picks a precedence for LOADING; this function's job is just to report what exists.)"""
    bundled_name = list_available_settings()[0]["name"]
    (tmp_path / f"{bundled_name}.json").write_text("{}")

    entries = list_available_settings(tmp_path)
    matches = [d for d in entries if d["name"] == bundled_name]
    assert len(matches) == 2
    assert {d["origin"] for d in matches} == {"bundled", "user"}


# -- Session.save_settings_as / load_settings round trip --------------------------------------


def test_save_writes_into_the_user_dir_not_the_package_dir(sess):
    from mujoco_visualizer.render_settings import _SETTINGS_DIR

    path = sess.save_settings_as("my_look")

    assert path == sess.user_settings_dir / "my_look.json"
    assert path.is_file()
    assert _SETTINGS_DIR not in path.parents
    assert not (_SETTINGS_DIR / "my_look.json").exists()


def test_save_creates_the_user_dir_if_missing(xml, tmp_path):
    user_dir = tmp_path / "nested" / "presets"
    assert not user_dir.exists()
    s = Session(xml_path=xml, width=64, height=64, user_settings_dir=user_dir)
    try:
        s.save_settings_as("first")
        assert user_dir.is_dir()
        assert (user_dir / "first.json").is_file()
    finally:
        s.close()


def test_save_requires_a_user_settings_dir(xml):
    s = Session(xml_path=xml, width=64, height=64)  # no user_settings_dir
    try:
        with pytest.raises(ValueError):
            s.save_settings_as("my_look")
    finally:
        s.close()


@pytest.mark.parametrize("value", ["../x", "a/b", "", "x.json", "a" * 65])
def test_save_settings_as_rejects_a_bad_name_even_called_directly(sess, value):
    """Defence in depth: save_settings_as must not trust that protocol.parse_command already
    validated the name -- it is a public method any caller (test, script, future non-wire
    entry point) can call directly."""
    with pytest.raises(ValueError):
        sess.save_settings_as(value)
    # And, crucially, nothing was written anywhere reachable.
    if sess.user_settings_dir.exists():
        assert list(sess.user_settings_dir.iterdir()) == []


def test_save_list_load_round_trip(sess):
    path = sess.save_settings_as("roundtrip")
    assert path.is_file()

    entries = list_available_settings(sess.user_settings_dir)
    assert {"name": "roundtrip", "origin": "user"} in entries

    sess.load_settings("roundtrip")  # must not raise


def test_load_settings_still_refuses_an_unknown_name_with_a_user_dir_configured(sess):
    with pytest.raises(ValueError):
        sess.load_settings("this_name_does_not_exist_anywhere")


def test_load_settings_prefers_the_user_preset_on_a_name_collision(sess):
    """Decision: on a name collision, LOADING resolves to the user's own save, not the
    bundled preset underneath it -- see Session.load_settings's docstring for the reasoning.
    Distinguished from the bundled version by a marker only the saved-then-edited file has.
    """
    bundled_name = list_available_settings()[0]["name"]
    sess.save_settings_as(bundled_name)
    # Tag the just-saved user copy so we can tell which file actually got read.
    user_path = sess.user_settings_dir / f"{bundled_name}.json"
    import json

    data = json.loads(user_path.read_text())
    data["_marker"] = "this-is-the-user-copy"
    user_path.write_text(json.dumps(data))

    sess.load_settings(bundled_name)  # must not raise, and must not explode on the extra key


def test_unwritable_user_dir_raises_and_leaves_no_partial_file(sess):
    sess.user_settings_dir.mkdir(parents=True, exist_ok=True)
    sess.user_settings_dir.chmod(0o500)  # read + execute, no write
    try:
        with pytest.raises(OSError):
            sess.save_settings_as("wont_work")
    finally:
        sess.user_settings_dir.chmod(0o700)  # restore so tmp_path teardown can clean up

    assert list(sess.user_settings_dir.iterdir()) == []


def test_save_settings_as_atomically_replaces_no_partial_file_on_a_mid_write_failure(
    sess, monkeypatch
):
    """A failure INSIDE Visualizer.save_settings (not just a permissions error opening the
    file) must not leave a temp file behind either -- the temp-then-replace path is cleaned
    up on ANY exception, not just the ones caught by chmod-based tests."""
    from mujoco_visualizer import visualizer as visualizer_mod

    def _boom(self, json_path):
        # Simulate a failure partway through -- e.g. an encoding error, a full disk hit
        # mid-dump. The real function has already opened the file by this point in a full
        # reproduction; here it's enough that save_settings raises at all.
        raise RuntimeError("simulated failure mid-save")

    monkeypatch.setattr(visualizer_mod.Visualizer, "save_settings", _boom)

    with pytest.raises(RuntimeError):
        sess.save_settings_as("doomed")

    assert list(sess.user_settings_dir.iterdir()) == []


# -- whole vis_state round trip (not a hand-picked subset of keys) ----------------------------


def test_save_load_round_trips_the_whole_vis_state(sess):
    """Visualizer.save_settings/load_settings both enumerate top-level vis_state groups
    EXPLICITLY (colors, geom_colors, alpha, vis_flags, geom_groups, site_groups, camera,
    lighting, floor, skybox, camera_presets, ghost, ...). A group added to one list but not
    the other -- or to neither -- round-trips as a silent loss: the save succeeds, the load
    succeeds, and only the value is gone. This exact shape (four tests green while the thing
    they guarded was broken) is why this asserts on EVERY key present before the save, not a
    hand-picked few that already worked.
    """
    before = copy.deepcopy(sess.viz.vis_state)
    assert before  # sanity: there is something to compare

    sess.save_settings_as("whole_state_probe")
    sess.load_settings("whole_state_probe")
    after = sess.viz.vis_state

    for key in before:
        assert key in after, f"{key!r} was present before the round trip, missing after"
        assert after[key] == before[key], (
            f"{key!r} changed across a save -> load round trip: "
            f"before={before[key]!r} after={after[key]!r}"
        )


def test_save_load_round_trip_catches_a_group_that_is_not_wired_into_either_list(xml, tmp_path):
    """Proof that the whole-state test above is not decorative: a top-level vis_state group
    that is NOT one of save_settings/load_settings' explicit keys (standing in for the next
    one somebody adds, the way 'ghost' was added on a sibling branch) does NOT survive a save
    -> load round trip. If this assertion ever starts failing because someone made save/load
    fully generic, that is an improvement -- update this test to match, don't delete it.

    Uses two INDEPENDENT sessions (save from one, load into another) rather than reusing one:
    reloading into the same live object that already holds the probe value in memory would
    leave it looking "present" regardless of whether the file round-tripped it -- load_settings
    only ever merges keys the loaded file actually has, it never clears ones it doesn't. A
    fresh session has no such leftover value to be fooled by.
    """
    user_dir = tmp_path / "user_settings"
    saver = Session(xml_path=xml, width=64, height=64, user_settings_dir=user_dir)
    try:
        saver.viz.vis_state["_unwired_probe_group"] = {"x": 1.0}
        saver.save_settings_as("unwired_probe")
    finally:
        saver.close()

    loader = Session(xml_path=xml, width=64, height=64, user_settings_dir=user_dir)
    try:
        loader.load_settings("unwired_probe")
        assert "_unwired_probe_group" not in loader.viz.vis_state
    finally:
        loader.close()
