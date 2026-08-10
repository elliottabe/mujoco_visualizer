"""Client command schema: validate before anything touches the simulation.

Two jobs. Validation, so a malformed message from the browser produces an error reply
instead of a half-applied state change. And coalescing, so a mouse drag that emitted a
dozen camera messages while the loop was busy stepping applies once -- without it, a fast
drag over a slow link builds a backlog of stale positions the camera then walks through.

Otherwise pure data: the one exception is ``settings``, which is checked against
``list_available_settings()`` (a directory listing) for ``load`` and against a name
whitelist for ``save``. That validation has to live in front of the simulation rather than
behind it, because a ``load`` value reaches ``open()``/``json.load()`` and a ``save`` value
becomes a filename ``open(..., 'w')`` creates -- validating either is exactly a validation job.
"""

import json
import math
from typing import Dict, List, Optional

from mujoco_visualizer.render_settings import PRESET_NAME_RE, list_available_settings

COMMANDS = frozenset(
    {
        "ctrl",
        "ctrl_group",
        "mode",
        "sim",
        "speed",
        "camera",
        "camera_preset",
        "render",
        "settings",
        "stream",
        "replay",
        "export",
        "export_cancel",
        "lock",
        "ping",
    }
)

_SIM_CMDS = frozenset({"play", "pause", "step", "reset"})
_MODES = frozenset({"absolute", "additive"})

# Types whose messages fully supersede an earlier one of the same type: only the last
# matters. ``ctrl``, ``ctrl_group``, ``replay``, ``lock`` and ``render`` are merged instead
# (see :func:`coalesce`). Keep this list exhaustive: a half-updated enumeration reads as
# current and is worse than a visibly stale one.
#
# ``replay`` is deliberately NOT here. Wholesale last-wins was correct when the command
# carried only ``{load, frame, play}``; it now carries eight independent fields, and dropping
# an earlier message wholesale silently discards every field the later one does not mention.
# Two commands in the same tick (33 ms at the default fps) is ordinary UI traffic -- holding
# ArrowRight while pressing ``]``, or ticking the ghost box mid scrub-drag -- and
# ``{ghost:true}`` followed by ``{frame:10}`` lost the ghost toggle with no error at all.
#
# ``render`` is deliberately NOT here either, for the same reason: its payload is
# ``{"set": {<dotted.key>: value}}``, a key-value delta, not a whole state. A settings panel
# with dozens of controls emitting two independent edits in one tick -- ``{colors.thorax:...}``
# then ``{alpha:...}`` -- must keep both, not have the second silently erase the first.
_LAST_WINS = frozenset({"mode", "speed", "camera", "settings", "stream"})

# Roots that exist in Visualizer.vis_state. A `render.set` key outside these was previously
# merged verbatim, creating a dead entry: the control appeared to do nothing and nothing said
# why. Validated here so a typo is a named error rather than a silent no-op. ``ghost`` is an
# exception: it does not exist in vis_state yet -- reserved here for a sibling task that adds
# it, so that task's keys are not rejected as unknown roots before it lands.
_VIS_STATE_ROOTS = frozenset({
    "colors", "geom_colors", "alpha", "vis_flags", "geom_groups", "site_groups",
    "camera", "camera_presets", "lighting", "floor", "skybox", "ghost",
    "geom_render_state", "forces", "tendons", "force_arrows",
})

# Roots that are fixed-length lists in vis_state (geom_groups/site_groups are boolean lists
# indexed by group id; build_scene_option reads them positionally). A BARE key naming one of
# these -- "geom_groups" with no ".<index>" -- would replace the whole list with whatever
# scalar the client sent, silently turning a list into e.g. a bool the next time anything reads
# it positionally. Scalar roots (alpha) and dict roots (floor, camera, ...) both survive
# whole-root replacement; these do not, so bare access to them is rejected here rather than at
# the write site, before it ever reaches vis_state.
_LIST_VALUED_ROOTS = frozenset({"geom_groups", "site_groups"})


class CommandError(ValueError):
    """A client command was malformed or out of range."""


def _num(cmd: Dict, key: str, default=None, lo=None, hi=None):
    if key not in cmd:
        if default is None:
            raise CommandError(f"{cmd.get('t')!r} requires {key!r}")
        return default
    value = cmd[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommandError(f"{key!r} must be a number, got {type(value).__name__}")
    value = float(value)
    if lo is not None and value < lo:
        raise CommandError(f"{key!r} must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise CommandError(f"{key!r} must be <= {hi}, got {value}")
    return value


def parse_command(raw, user_settings_dir: Optional[str] = None) -> Dict:
    """Validate one client command. Accepts a JSON string or a dict.

    *user_settings_dir*, when given, is consulted (alongside the bundled presets) to decide
    whether a ``settings.load`` name is valid -- so a preset a user has actually saved via
    ``settings.save`` is loadable, not just the ones shipped in the package. Omitting it
    (the default) validates ``load`` against the bundled presets only, which is the same
    behaviour this function always had before per-session user directories existed.

    Returns a normalised dict with defaults filled in. Raises :class:`CommandError`.
    """
    if isinstance(raw, (str, bytes)):
        try:
            cmd = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise CommandError(f"not valid JSON: {exc}") from exc
    else:
        cmd = raw
    if not isinstance(cmd, dict):
        raise CommandError(f"command must be an object, got {type(cmd).__name__}")

    kind = cmd.get("t")
    if kind is None:
        raise CommandError("command is missing 't'")
    if kind not in COMMANDS:
        raise CommandError(f"unknown command type {kind!r}")

    if kind == "ctrl":
        values = cmd.get("set")
        if not isinstance(values, dict):
            raise CommandError("'ctrl' requires a 'set' object")
        out = {}
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CommandError(f"ctrl value for {name!r} must be a number")
            out[str(name)] = float(value)
        return {"t": "ctrl", "set": out}

    if kind == "ctrl_group":
        group = cmd.get("group")
        if not isinstance(group, str) or not group:
            raise CommandError("'ctrl_group' requires a 'group' string")
        return {"t": "ctrl_group", "group": group, "gain": _num(cmd, "gain", 1.0)}

    if kind == "mode":
        mode = cmd.get("ctrl")
        if mode not in _MODES:
            raise CommandError(f"'mode.ctrl' must be one of {sorted(_MODES)}")
        return {"t": "mode", "ctrl": mode}

    if kind == "sim":
        action = cmd.get("cmd")
        if action not in _SIM_CMDS:
            raise CommandError(f"'sim.cmd' must be one of {sorted(_SIM_CMDS)}")
        return {"t": "sim", "cmd": action, "n": int(_num(cmd, "n", 1.0, lo=1, hi=1e6))}

    if kind == "speed":
        return {
            "t": "speed",
            "substeps_per_frame": int(
                _num(cmd, "substeps_per_frame", lo=1, hi=100000)
            ),
        }

    if kind == "camera":
        out = {"t": "camera"}
        named = cmd.get("named")
        if named is not None:
            if not isinstance(named, str):
                raise CommandError("'camera.named' must be a string")
            out["named"] = named
            return out
        for key, lo, hi in (
            ("az", -3600.0, 3600.0),
            ("el", -90.0, 90.0),
            ("dist", 1e-6, 1e6),
        ):
            if key in cmd:
                out[key] = _num(cmd, key, lo=lo, hi=hi)
        if "lookat" in cmd:
            lookat = cmd["lookat"]
            if not isinstance(lookat, (list, tuple)) or len(lookat) != 3:
                raise CommandError("'camera.lookat' must be 3 numbers")
            lookat_values = []
            for i, v in enumerate(lookat):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise CommandError(
                        f"'camera.lookat' element {i} must be a number, got {type(v).__name__}"
                    )
                lookat_values.append(float(v))
            out["lookat"] = lookat_values
        if "pan" in cmd:
            pan = cmd["pan"]
            if not isinstance(pan, (list, tuple)) or len(pan) != 2:
                raise CommandError("'camera.pan' must be 2 numbers [dx, dy]")
            pan_values = []
            for i, v in enumerate(pan):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise CommandError(
                        f"'camera.pan' element {i} must be a number, got {type(v).__name__}"
                    )
                pan_values.append(float(v))
            out["pan"] = pan_values
        if len(out) == 1:
            raise CommandError("'camera' needs 'named' or at least one of az/el/dist/lookat")
        return out

    if kind == "camera_preset":
        op = cmd.get("op")
        if op not in ("save", "delete"):
            raise CommandError("'camera_preset.op' must be 'save' or 'delete'")
        name = cmd.get("name")
        if not isinstance(name, str) or not PRESET_NAME_RE.match(name):
            raise CommandError(
                f"'camera_preset.name' must match {PRESET_NAME_RE.pattern}, got {name!r}"
            )
        return {"t": "camera_preset", "op": op, "name": name}

    if kind == "render":
        values = cmd.get("set")
        if not isinstance(values, dict):
            raise CommandError("'render' requires a 'set' object")
        for dotted in values:
            parts = str(dotted).split(".")
            root = parts[0]
            if root not in _VIS_STATE_ROOTS:
                raise CommandError(
                    f"'render.set' key {dotted!r} has unknown root {root!r}; "
                    f"expected one of {', '.join(sorted(_VIS_STATE_ROOTS))}"
                )
            if len(parts) == 1 and root in _LIST_VALUED_ROOTS:
                raise CommandError(
                    f"'render.set' key {dotted!r} would replace the whole list at {root!r}; "
                    f"address an element as {root}.<index>, not {root!r} bare"
                )
        return {"t": "render", "set": dict(values)}

    if kind == "settings":
        save_name = cmd.get("save")
        if save_name is not None:
            # A save name is never checked against an existing-file whitelist (there is
            # nothing to whitelist against -- the whole point is creating a new preset). It
            # is instead checked against a NAME PATTERN, for exactly the same reason
            # 'load' is checked against a directory listing below: this value becomes a
            # filename an `open(..., 'w')` on the server creates, so "../x", "a/b", and a
            # bare "" or path-with-suffix must be rejected before they ever get near a
            # filesystem call, not after.
            if not isinstance(save_name, str) or not PRESET_NAME_RE.match(save_name):
                raise CommandError(
                    "'settings.save' name must match {0!r}; got {1!r}".format(
                        PRESET_NAME_RE.pattern, save_name
                    )
                )
            return {"t": "settings", "save": save_name}

        name = cmd.get("load")
        if not isinstance(name, str) or not name:
            raise CommandError("'settings' requires a 'load' name")
        # Whitelisted by NAME against the bundled AND user presets, never accepted as a path.
        # Visualizer._resolve_settings_path's first branch is `if Path(x).is_file()`, which is
        # correct for its own callers (a user naming a settings file on the command line) but
        # means an unvalidated wire value makes the server open() and json.load() any path a
        # client names -- and then echo the whitelisted keys back in the scene message. The
        # bind address defaults to 127.0.0.1 but --host widens it.
        available = {d["name"] for d in list_available_settings(user_settings_dir)}
        if name not in available:
            raise CommandError(
                "'settings.load' must be one of the available settings presets; "
                "{0!r} is not (available: {1})".format(name, ", ".join(sorted(available)))
            )
        return {"t": "settings", "load": name}

    if kind == "stream":
        out = {"t": "stream"}
        if "width" in cmd:
            out["width"] = int(_num(cmd, "width", lo=16, hi=4096))
        if "height" in cmd:
            out["height"] = int(_num(cmd, "height", lo=16, hi=4096))
        if "fps" in cmd:
            out["fps"] = _num(cmd, "fps", lo=1, hi=120)
        if "quality" in cmd:
            out["quality"] = int(_num(cmd, "quality", lo=1, hi=100))
        if len(out) == 1:
            raise CommandError("'stream' needs at least one of width/height/fps/quality")
        return out

    if kind == "replay":
        out = {"t": "replay"}
        if "load" in cmd:
            # Rejected rather than validated-and-ignored. The loop's _apply_replay never
            # reads this field: the trajectory source is chosen once, at launch, and swapping
            # it mid-session would have to rebuild the model, the clip table and every cached
            # width. Accepting the key made the server look like it honoured a request it
            # silently dropped, which is worse than refusing it.
            raise CommandError(
                "'replay.load' is not supported: the trajectory source is fixed when the "
                "server starts. Use 'clip' to choose which clip of that source to replay."
            )
        if "clip" in cmd:
            out["clip"] = int(_num(cmd, "clip", lo=0, hi=1e9))
        if "frame" in cmd:
            out["frame"] = int(_num(cmd, "frame", lo=0, hi=1e9))
        if "stride" in cmd:
            out["stride"] = int(_num(cmd, "stride", lo=1, hi=100000))
        if "trim" in cmd:
            trim = cmd["trim"]
            if not isinstance(trim, (list, tuple)) or len(trim) != 2:
                raise CommandError("'replay.trim' must be two frame indices [in, out]")
            values = []
            for i, v in enumerate(trim):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise CommandError(
                        f"'replay.trim' element {i} must be a number, got "
                        f"{type(v).__name__}"
                    )
                if v < 0:
                    raise CommandError(f"'replay.trim' element {i} must be >= 0, got {v}")
                values.append(int(v))
            if values[0] > values[1]:
                raise CommandError(
                    f"'replay.trim' must be ordered [in, out]; got in={values[0]} > "
                    f"out={values[1]}"
                )
            out["trim"] = values
        for flag in ("play", "loop", "ghost"):
            if flag in cmd:
                if not isinstance(cmd[flag], bool):
                    raise CommandError(f"'replay.{flag}' must be a boolean")
                out[flag] = cmd[flag]
        if len(out) == 1:
            raise CommandError(
                "'replay' needs at least one of clip/frame/play/stride/trim/loop/ghost"
            )
        return out

    if kind == "export":
        # width/height/fps are required rather than defaulted: an export is a deliberate,
        # minutes-long, file-producing act, and inheriting the preview's 640x480 by accident
        # is a silently useless 4K-shaped request.
        for key in ("width", "height", "fps"):
            if key not in cmd:
                raise CommandError(f"'export' requires {key!r}")
        fmt = cmd.get("format", "mp4")
        if fmt not in ("mp4", "png"):
            raise CommandError(f"'export.format' must be 'mp4' or 'png', got {fmt!r}")
        out = {
            "t": "export",
            "width": int(_num(cmd, "width", lo=16, hi=8192)),
            "height": int(_num(cmd, "height", lo=16, hi=8192)),
            "fps": _num(cmd, "fps", lo=1, hi=240),
            "format": fmt,
            "crf": int(_num(cmd, "crf", 20.0, lo=0, hi=51)),
        }
        if "stride" in cmd:
            out["stride"] = int(_num(cmd, "stride", lo=1, hi=100000))
        if "trim" in cmd:
            # Same shape rules as replay.trim; reuse by re-parsing through this function so
            # the two can never drift apart.
            out["trim"] = parse_command({"t": "replay", "trim": cmd["trim"]})["trim"]
        for flag in ("shadows", "tendons"):
            if flag in cmd:
                if not isinstance(cmd[flag], bool):
                    raise CommandError(f"'export.{flag}' must be a boolean")
                out[flag] = cmd[flag]
        if "path" in cmd:
            if not isinstance(cmd["path"], str) or not cmd["path"]:
                raise CommandError("'export.path' must be a non-empty string")
            out["path"] = cmd["path"]
        return out

    if kind == "export_cancel":
        return {"t": "export_cancel"}

    if kind == "lock":
        out = {"t": "lock"}
        has_set = "set" in cmd
        has_clear = "clear" in cmd
        if not (has_set or has_clear):
            raise CommandError("'lock' requires at least one of 'set' or 'clear'")
        if has_set:
            values = cmd.get("set")
            if not isinstance(values, dict):
                raise CommandError("'lock' requires a 'set' object")
            normalized = {}
            for name, value in values.items():
                if value is None:
                    # None means freeze at the current value when lock engages
                    normalized[str(name)] = None
                elif isinstance(value, (list, tuple)):
                    # List of numbers
                    converted = []
                    for i, v in enumerate(value):
                        if isinstance(v, bool) or not isinstance(v, (int, float)):
                            raise CommandError(
                                f"lock value for {name!r} element {i} must be a number, "
                                f"got {type(v).__name__}"
                            )
                        float_val = float(v)
                        if not math.isfinite(float_val):
                            raise CommandError(
                                f"lock value for {name!r} element {i} must be finite, got {float_val}"
                            )
                        converted.append(float_val)
                    normalized[str(name)] = converted
                else:
                    # Scalar number
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise CommandError(
                            f"lock value for {name!r} must be a number or None, "
                            f"got {type(value).__name__}"
                        )
                    float_val = float(value)
                    if not math.isfinite(float_val):
                        raise CommandError(
                            f"lock value for {name!r} must be finite, got {float_val}"
                        )
                    # Normalize scalar to list
                    normalized[str(name)] = [float_val]
            out["set"] = normalized
        if has_clear:
            if not isinstance(cmd.get("clear"), bool):
                raise CommandError("'lock.clear' must be a boolean")
            out["clear"] = cmd.get("clear")
        return out

    return {"t": "ping"}


def coalesce(cmds: List[Dict]) -> List[Dict]:
    """Collapse redundant commands, keeping the order of the survivors.

    ``_LAST_WINS`` types keep only their final message. ``ctrl``, ``render`` and ``replay``
    commands merge key-by-key with later values winning. ``ctrl_group`` keeps the last gain
    per group. ``sim`` messages are events (play/pause/step/reset) and are all preserved in
    order.

    ``lock`` commands merge with special handling: ``clear`` is an ordered event that resets the
    running set accumulator, and is preserved in output. When a message carries both ``set`` and
    ``clear``, the ``clear`` applies first (resetting the accumulator) and then the ``set`` is
    merged onto it. The final output carries both fields if they were ever set, omitting an
    empty ``set`` and omitting ``clear`` if it was never encountered.

    ``replay`` merges rather than last-wins because its eight fields are independent knobs,
    not one value: a scrub drag emitting ``{frame:...}`` every few ms must still coalesce to a
    single seek (the reason it was collapsed in the first place), but a ``{ghost:true}`` that
    happens to share the tick must not vanish with the earlier frames. Later values still win
    per key, so the collapsed drag behaves exactly as before.

    ``lock.set`` merges for the same reason: a UI that toggles multiple joints in quick
    succession (e.g. a group checkbox covering nine legs) must apply all toggles, not just
    the last one.

    ``render.set`` merges for the same reason as ``lock.set``: its payload is a
    ``{<dotted.key>: value}`` delta, so a settings panel with dozens of controls emitting two
    edits to different keys within one tick (e.g. ``{colors.thorax:...}`` then
    ``{alpha:...}``) must keep both instead of the second replacing the whole ``set`` dict and
    silently reverting the first.
    """
    last_index: Dict[str, int] = {}
    merged_ctrl: Dict[str, float] = {}
    ctrl_index = None
    merged_replay: Dict = {}
    replay_index = None
    merged_lock_set: Dict = {}
    lock_clear_flag: bool = False
    lock_index = None
    merged_render_set: Dict = {}
    render_index = None
    group_index: Dict[str, int] = {}
    keep = [True] * len(cmds)

    for i, cmd in enumerate(cmds):
        kind = cmd["t"]
        if kind in _LAST_WINS:
            if kind in last_index:
                keep[last_index[kind]] = False
            last_index[kind] = i
        elif kind == "ctrl":
            merged_ctrl.update(cmd["set"])
            if ctrl_index is not None:
                keep[ctrl_index] = False
            ctrl_index = i
        elif kind == "replay":
            for key, value in cmd.items():
                if key != "t":
                    merged_replay[key] = value
            if replay_index is not None:
                keep[replay_index] = False
            replay_index = i
        elif kind == "lock":
            # Process clear first (resets accumulator), then merge set values key-by-key
            if cmd.get("clear"):
                lock_clear_flag = True
                merged_lock_set = {}
            if "set" in cmd:
                merged_lock_set.update(cmd["set"])
            if lock_index is not None:
                keep[lock_index] = False
            lock_index = i
        elif kind == "render":
            merged_render_set.update(cmd["set"])
            if render_index is not None:
                keep[render_index] = False
            render_index = i
        elif kind == "ctrl_group":
            group = cmd["group"]
            if group in group_index:
                keep[group_index[group]] = False
            group_index[group] = i

    out = []
    for i, cmd in enumerate(cmds):
        if not keep[i]:
            continue
        if i == ctrl_index:
            out.append({"t": "ctrl", "set": dict(merged_ctrl)})
        elif i == replay_index:
            out.append({"t": "replay", **merged_replay})
        elif i == lock_index:
            out_lock = {"t": "lock"}
            if lock_clear_flag:
                out_lock["clear"] = True
            if merged_lock_set:
                out_lock["set"] = dict(merged_lock_set)
            out.append(out_lock)
        elif i == render_index:
            out.append({"t": "render", "set": dict(merged_render_set)})
        else:
            out.append(cmd)
    return out
