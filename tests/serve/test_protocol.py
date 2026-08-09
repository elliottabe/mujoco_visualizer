"""Command parsing is pure data - no model, no GL, no server."""

import json

import pytest

from mujoco_visualizer.serve.protocol import CommandError, coalesce, parse_command


def test_parses_a_json_string():
    cmd = parse_command('{"t":"sim","cmd":"play"}')
    assert cmd == {"t": "sim", "cmd": "play", "n": 1}


def test_parses_a_dict():
    assert parse_command({"t": "mode", "ctrl": "additive"})["ctrl"] == "additive"


def test_rejects_malformed_json():
    with pytest.raises(CommandError):
        parse_command("{not json")


def test_rejects_unknown_type():
    with pytest.raises(CommandError):
        parse_command({"t": "launch_missiles"})


def test_rejects_missing_type():
    with pytest.raises(CommandError):
        parse_command({"cmd": "play"})


def test_rejects_wrong_value_type():
    with pytest.raises(CommandError):
        parse_command({"t": "ctrl", "set": "not a mapping"})


def test_rejects_non_numeric_ctrl_value():
    with pytest.raises(CommandError):
        parse_command({"t": "ctrl", "set": {"a": "loud"}})


def test_rejects_bad_sim_command():
    with pytest.raises(CommandError):
        parse_command({"t": "sim", "cmd": "explode"})


def test_rejects_bad_mode():
    with pytest.raises(CommandError):
        parse_command({"t": "mode", "ctrl": "sideways"})


def test_stream_bounds_are_enforced():
    with pytest.raises(CommandError):
        parse_command({"t": "stream", "width": 0})
    with pytest.raises(CommandError):
        parse_command({"t": "stream", "quality": 101})


def test_rejects_boolean_in_numeric_field():
    """Booleans are instances of int, so must be explicitly rejected."""
    with pytest.raises(CommandError):
        parse_command({"t": "camera", "az": True})
    with pytest.raises(CommandError):
        parse_command({"t": "ctrl_group", "group": "g", "gain": False})


def test_rejects_boolean_in_lookat():
    """camera.lookat elements must reject booleans like all numeric fields."""
    with pytest.raises(CommandError):
        parse_command({"t": "camera", "lookat": [1.0, 2.0, True]})


def test_rejects_non_numeric_lookat_element():
    """camera.lookat elements must raise CommandError, not bare ValueError."""
    with pytest.raises(CommandError):
        parse_command({"t": "camera", "lookat": [1.0, 2.0, "bad"]})


def test_coalesce_keeps_only_the_last_camera():
    cmds = [
        {"t": "camera", "az": 1.0},
        {"t": "camera", "az": 2.0},
        {"t": "camera", "az": 3.0},
    ]
    assert coalesce(cmds) == [{"t": "camera", "az": 3.0}]


def test_coalesce_merges_ctrl_sets_with_later_keys_winning():
    cmds = [
        {"t": "ctrl", "set": {"a": 0.1, "b": 0.2}},
        {"t": "ctrl", "set": {"a": 0.9}},
    ]
    assert coalesce(cmds) == [{"t": "ctrl", "set": {"a": 0.9, "b": 0.2}}]


def test_coalesce_keeps_last_gain_per_group_independently():
    cmds = [
        {"t": "ctrl_group", "group": "leg.T1.left", "gain": 0.1},
        {"t": "ctrl_group", "group": "leg.T1.right", "gain": 0.5},
        {"t": "ctrl_group", "group": "leg.T1.left", "gain": 0.9},
    ]
    got = coalesce(cmds)
    assert len(got) == 2
    assert {c["group"]: c["gain"] for c in got} == {
        "leg.T1.left": 0.9,
        "leg.T1.right": 0.5,
    }


def test_coalesce_preserves_every_sim_event_in_order():
    cmds = [
        {"t": "sim", "cmd": "step", "n": 1},
        {"t": "sim", "cmd": "step", "n": 1},
        {"t": "sim", "cmd": "reset", "n": 1},
    ]
    assert coalesce(cmds) == cmds


def test_coalesce_is_stable_for_unrelated_types():
    cmds = [
        {"t": "ctrl", "set": {"a": 0.1}},
        {"t": "sim", "cmd": "play", "n": 1},
        {"t": "camera", "az": 1.0},
    ]
    assert coalesce(cmds) == cmds


def test_round_trips_through_json():
    original = {"t": "render", "set": {"floor.alpha": 0.5}}
    assert parse_command(json.dumps(original))["set"] == {"floor.alpha": 0.5}


# -- settings.load is a name from a whitelist, never a path ---------------------------------


def test_settings_load_accepts_a_bundled_preset_name():
    from mujoco_visualizer import list_available_settings

    name = list_available_settings()[0]
    assert parse_command({"t": "settings", "load": name}) == {"t": "settings", "load": name}


@pytest.mark.parametrize(
    "value",
    [
        "/etc/passwd",
        "../../../../etc/hostname",
        "./settings/Default.json",
        "not_a_preset",
    ],
)
def test_settings_load_rejects_anything_not_in_the_whitelist(value):
    """Visualizer._resolve_settings_path's first branch is `if Path(x).is_file()`, so an
    unvalidated wire value makes the server open() and json.load() any path a client names --
    and then echo the whitelisted keys straight back in the scene message. The default bind is
    127.0.0.1, but --host widens it."""
    with pytest.raises(CommandError) as exc:
        parse_command({"t": "settings", "load": value})
    assert value in str(exc.value)  # the rejection names the offending value


def test_settings_load_rejects_a_real_readable_file_outside_the_settings_dir(tmp_path):
    """The path being genuinely readable is the whole point: an is_file() check would accept
    this one."""
    victim = tmp_path / "secret.json"
    victim.write_text('{"alpha": 1.0}')
    with pytest.raises(CommandError):
        parse_command({"t": "settings", "load": str(victim)})


# --- replay -----------------------------------------------------------------------

def test_replay_accepts_the_full_key_set():
    cmd = parse_command(
        {"t": "replay", "clip": 42, "frame": 412, "play": True,
         "stride": 10, "trim": [120, 900], "loop": False, "ghost": True}
    )
    assert cmd == {"t": "replay", "clip": 42, "frame": 412, "play": True,
                   "stride": 10, "trim": [120, 900], "loop": False, "ghost": True}


def test_replay_needs_at_least_one_field():
    with pytest.raises(CommandError, match="at least one"):
        parse_command({"t": "replay"})


def test_replay_stride_must_be_at_least_one():
    with pytest.raises(CommandError, match="stride"):
        parse_command({"t": "replay", "stride": 0})


def test_replay_trim_must_be_two_ordered_ints():
    with pytest.raises(CommandError, match="two"):
        parse_command({"t": "replay", "trim": [5]})
    with pytest.raises(CommandError, match="in .* out|ordered"):
        parse_command({"t": "replay", "trim": [900, 120]})


def test_replay_rejects_negative_frame():
    with pytest.raises(CommandError):
        parse_command({"t": "replay", "frame": -1})


def test_replay_booleans_must_be_boolean():
    with pytest.raises(CommandError, match="ghost"):
        parse_command({"t": "replay", "ghost": "yes"})


# --- export -----------------------------------------------------------------------

def test_export_requires_size_and_fps():
    with pytest.raises(CommandError, match="width"):
        parse_command({"t": "export", "height": 1080, "fps": 30})


def test_export_defaults_format_and_crf():
    cmd = parse_command({"t": "export", "width": 1920, "height": 1080, "fps": 30})
    assert cmd["format"] == "mp4"
    assert cmd["crf"] == 20
    assert cmd["width"] == 1920 and cmd["height"] == 1080 and cmd["fps"] == 30.0


def test_export_rejects_unknown_format():
    with pytest.raises(CommandError, match="format"):
        parse_command({"t": "export", "width": 640, "height": 480, "fps": 30,
                       "format": "avi"})


def test_export_clamps_nothing_but_validates_range():
    with pytest.raises(CommandError):
        parse_command({"t": "export", "width": 8, "height": 480, "fps": 30})
    with pytest.raises(CommandError):
        parse_command({"t": "export", "width": 640, "height": 480, "fps": 0})


def test_export_cancel_parses():
    assert parse_command({"t": "export_cancel"}) == {"t": "export_cancel"}


def test_export_commands_are_events_not_last_wins():
    # Two exports queued in one tick must BOTH survive coalescing: the second is
    # rejected by the loop with "one at a time", which is a different, visible outcome
    # from silently discarding the first.
    out = coalesce([
        parse_command({"t": "export", "width": 640, "height": 480, "fps": 30}),
        parse_command({"t": "export_cancel"}),
    ])
    assert [c["t"] for c in out] == ["export", "export_cancel"]


def test_replay_still_last_wins_so_a_scrub_drag_applies_once():
    out = coalesce([
        parse_command({"t": "replay", "frame": 100}),
        parse_command({"t": "replay", "frame": 200}),
        parse_command({"t": "replay", "frame": 300}),
    ])
    assert out == [{"t": "replay", "frame": 300}]
