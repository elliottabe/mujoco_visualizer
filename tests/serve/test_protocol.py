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

    name = list_available_settings()[0]["name"]
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


def test_settings_load_consults_the_user_settings_dir_when_given(tmp_path):
    """A name that exists only in the user directory is not in the bundled whitelist -- it
    must still be accepted when that directory is passed in, and still rejected when it
    isn't (e.g. a stale second connection with no per-session dir configured)."""
    (tmp_path / "my_look.json").write_text("{}")

    cmd = parse_command(
        {"t": "settings", "load": "my_look"}, user_settings_dir=tmp_path
    )
    assert cmd == {"t": "settings", "load": "my_look"}

    with pytest.raises(CommandError):
        parse_command({"t": "settings", "load": "my_look"})  # no user_settings_dir given


# -- settings.save is a NAME, checked against a pattern, never an existing-file whitelist ---


def test_settings_save_accepts_a_wellformed_name():
    cmd = parse_command({"t": "settings", "save": "my_fig-2"})
    assert cmd == {"t": "settings", "save": "my_fig-2"}


@pytest.mark.parametrize(
    "value",
    [
        "../x",       # path traversal
        "a/b",        # path separator
        "",           # empty
        "x.json",     # looks like a filename, not a name
        "a" * 65,     # over the 64-char cap
    ],
)
def test_settings_save_rejects_anything_that_is_not_a_bare_name(value):
    with pytest.raises(CommandError) as exc:
        parse_command({"t": "settings", "save": value})
    # The rejection quotes the pattern itself, not just a generic "invalid name" -- so the
    # error is actionable rather than requiring a source-code lookup to know what IS valid.
    assert r"^[A-Za-z0-9_-]{1,64}$" in str(exc.value)


def test_settings_save_accepts_the_boundary_lengths():
    """1 char and 64 chars are both valid -- only 0 and 65+ are rejected."""
    assert parse_command({"t": "settings", "save": "a"})["save"] == "a"
    assert parse_command({"t": "settings", "save": "a" * 64})["save"] == "a" * 64


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


def test_coalescing_a_replay_never_discards_a_field_the_later_one_omits():
    """The reason ``replay`` merges instead of last-wins.

    A scrub drag and any other replay control landing in the same tick (33 ms at the default
    fps -- holding ArrowRight while pressing ``]``, or ticking the ghost box mid-drag) used to
    collapse to just the later message, silently dropping the toggle with no error at all.
    """
    out = coalesce([
        parse_command({"t": "replay", "ghost": True}),
        parse_command({"t": "replay", "frame": 10}),
    ])
    assert out == [{"t": "replay", "ghost": True, "frame": 10}]


def test_coalescing_a_replay_still_lets_the_later_value_win_per_key():
    out = coalesce([
        parse_command({"t": "replay", "frame": 10, "stride": 2, "loop": True}),
        parse_command({"t": "replay", "frame": 40, "loop": False}),
    ])
    assert out == [{"t": "replay", "frame": 40, "stride": 2, "loop": False}]


def test_a_coalesced_replay_keeps_the_position_of_the_last_message():
    """The merged command must sit where the LAST replay sat, not the first: a ``sim`` event
    queued between the two is an ordered event, and moving the replay in front of it would
    reorder "seek, then step" into "step, then seek"."""
    out = coalesce([
        parse_command({"t": "replay", "ghost": True}),
        parse_command({"t": "sim", "cmd": "step"}),
        parse_command({"t": "replay", "frame": 10}),
    ])
    assert [c["t"] for c in out] == ["sim", "replay"]
    assert out[-1] == {"t": "replay", "ghost": True, "frame": 10}


def test_replay_load_is_rejected_rather_than_validated_and_ignored():
    """``_apply_replay`` never reads ``load``: the source is fixed when the server starts.

    Accepting the field made the server look like it honoured a request it silently dropped.
    """
    with pytest.raises(CommandError, match="fixed when the server starts"):
        parse_command({"t": "replay", "load": "/some/other/rollout.h5"})


def test_the_replay_error_message_no_longer_advertises_load():
    with pytest.raises(CommandError) as excinfo:
        parse_command({"t": "replay"})
    assert "load" not in str(excinfo.value)


def test_render_rejects_an_unknown_root():
    with pytest.raises(CommandError, match="colours"):
        parse_command({"t": "render", "set": {"colours.thorax": "#ff0000"}})


def test_render_accepts_every_known_root():
    from mujoco_visualizer.serve.protocol import _VIS_STATE_ROOTS
    for root in sorted(_VIS_STATE_ROOTS):
        cmd = parse_command({"t": "render", "set": {f"{root}.x": 1}})
        assert cmd["set"] == {f"{root}.x": 1}


def test_render_accepts_a_bare_key_whose_name_is_itself_a_known_root():
    """``alpha`` is a scalar at the top of ``vis_state``, not a container to descend into, so
    addressing it with no sub-key is legitimate -- exactly how ``apply_render`` already treats
    it (``parts[:-1]`` is empty, so the whole dotted string becomes the key written straight
    onto ``vis_state``). A single-segment key is therefore accepted whenever its name is
    itself a known root; only a name that matches no root at all is a validation error (see
    ``test_render_rejects_a_bare_key_with_no_root`` below)."""
    cmd = parse_command({"t": "render", "set": {"alpha": 0.5}})
    assert cmd["set"] == {"alpha": 0.5}


def test_render_rejects_a_bare_key_with_no_root():
    with pytest.raises(CommandError, match="bogus"):
        parse_command({"t": "render", "set": {"bogus": 0.5}})


def test_render_rejects_a_bare_geom_groups_key():
    """``geom_groups`` is a fixed-length list; a bare key with no ``.<index>`` would replace
    the whole list with whatever scalar the client sent. Unlike ``alpha`` (a scalar root),
    whole-root replacement here is destructive, so it is rejected -- and the message names the
    form the client should have sent instead."""
    with pytest.raises(CommandError, match=r"geom_groups.*\.<index>"):
        parse_command({"t": "render", "set": {"geom_groups": True}})


def test_render_rejects_a_bare_site_groups_key():
    with pytest.raises(CommandError, match=r"site_groups.*\.<index>"):
        parse_command({"t": "render", "set": {"site_groups": True}})


def test_render_still_accepts_a_bare_alpha_key():
    """The list-root guard must not overreach into scalar roots: ``alpha`` has no index to
    address and whole-root replacement is exactly what setting it means."""
    cmd = parse_command({"t": "render", "set": {"alpha": 0.5}})
    assert cmd["set"] == {"alpha": 0.5}


# --- lock -----------------------------------------------------------------------

def test_lock_accepts_scalar_list_and_null_values():
    cmd = parse_command({"t": "lock", "set": {"hinge_a": 0.5, "root.quat": [1, 0, 0, 0],
                                              "wing_yaw_left": None}})
    assert cmd["set"]["hinge_a"] == [0.5]
    assert cmd["set"]["root.quat"] == [1.0, 0.0, 0.0, 0.0]
    assert cmd["set"]["wing_yaw_left"] is None, "None means freeze at the current value"


def test_lock_clear_parses_alone():
    assert parse_command({"t": "lock", "clear": True}) == {"t": "lock", "clear": True}


def test_lock_needs_set_or_clear():
    with pytest.raises(CommandError, match="set.*clear"):
        parse_command({"t": "lock"})


def test_lock_rejects_non_numeric_and_non_finite_values():
    with pytest.raises(CommandError):
        parse_command({"t": "lock", "set": {"hinge_a": "0.5"}})
    with pytest.raises(CommandError, match="finite"):
        parse_command({"t": "lock", "set": {"hinge_a": float("inf")}})


def test_lock_rejects_a_boolean_value():
    with pytest.raises(CommandError):
        parse_command({"t": "lock", "set": {"hinge_a": True}})


def test_lock_merges_key_by_key_rather_than_replacing():
    out = coalesce([
        parse_command({"t": "lock", "set": {"a": 1.0}}),
        parse_command({"t": "lock", "set": {"b": 2.0}}),
    ])
    assert len(out) == 1
    assert out[0]["set"] == {"a": [1.0], "b": [2.0]}, "a group toggle sets many at once"


def test_lock_later_value_wins_per_key():
    out = coalesce([
        parse_command({"t": "lock", "set": {"a": 1.0}}),
        parse_command({"t": "lock", "set": {"a": 3.0}}),
    ])
    assert out[0]["set"] == {"a": [3.0]}


def test_lock_clear_is_not_swallowed_by_a_later_set():
    out = coalesce([
        parse_command({"t": "lock", "clear": True}),
        parse_command({"t": "lock", "set": {"a": 1.0}}),
    ])
    kinds = [(c.get("clear"), c.get("set")) for c in out]
    assert any(c is True for c, _ in kinds), "a clear must not vanish into a later set"


def test_lock_combined_set_and_clear_in_single_command():
    """A single command with both set and clear must preserve both fields."""
    out = coalesce([parse_command({"t": "lock", "set": {"a": 1.0}, "clear": True})])
    assert len(out) == 1
    assert out[0]["clear"] is True
    assert out[0]["set"] == {"a": [1.0]}


def test_lock_set_then_clear_yields_only_clear():
    """After a clear, earlier set values are discarded."""
    out = coalesce([
        parse_command({"t": "lock", "set": {"a": 1.0}}),
        parse_command({"t": "lock", "clear": True}),
    ])
    assert len(out) == 1
    assert out[0] == {"t": "lock", "clear": True}


def test_lock_clear_then_set_preserves_both():
    """Clear resets the accumulator, then set merges onto the empty dict."""
    out = coalesce([
        parse_command({"t": "lock", "clear": True}),
        parse_command({"t": "lock", "set": {"b": 2.0}}),
    ])
    assert len(out) == 1
    assert out[0]["clear"] is True
    assert out[0]["set"] == {"b": [2.0]}


def test_lock_combined_followed_by_set_only():
    """Combined command followed by set-only: clear applies first, then both sets merge."""
    out = coalesce([
        parse_command({"t": "lock", "set": {"a": 1.0}, "clear": True}),
        parse_command({"t": "lock", "set": {"b": 2.0}}),
    ])
    assert len(out) == 1
    assert out[0]["clear"] is True
    assert out[0]["set"] == {"a": [1.0], "b": [2.0]}


def test_lock_none_then_number_for_same_key():
    """Number overwrites None for the same key in coalesce."""
    out = coalesce([
        parse_command({"t": "lock", "set": {"a": None}}),
        parse_command({"t": "lock", "set": {"a": 1.0}}),
    ])
    assert len(out) == 1
    assert out[0]["set"] == {"a": [1.0]}, "later number wins over None"


def test_lock_number_then_none_for_same_key():
    """None survives coalesce when it overwrites a number for the same key."""
    out = coalesce([
        parse_command({"t": "lock", "set": {"a": 1.0}}),
        parse_command({"t": "lock", "set": {"a": None}}),
    ])
    assert len(out) == 1
    assert out[0]["set"]["a"] is None, "None must survive as None, not be coerced to [0.0]"
