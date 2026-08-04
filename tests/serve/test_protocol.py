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
