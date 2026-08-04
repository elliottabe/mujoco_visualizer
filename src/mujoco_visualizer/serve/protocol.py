"""Client command schema: validate before anything touches the simulation.

Two jobs. Validation, so a malformed message from the browser produces an error reply
instead of a half-applied state change. And coalescing, so a mouse drag that emitted a
dozen camera messages while the loop was busy stepping applies once -- without it, a fast
drag over a slow link builds a backlog of stale positions the camera then walks through.

Otherwise pure data: the one exception is ``settings.load``, which is checked against
``list_available_settings()`` (a directory listing). That whitelist has to live in front of
the simulation rather than behind it, because the value reaches ``open()``/``json.load()`` and
validating a filename is exactly a validation job.
"""

import json
from typing import Dict, List

from mujoco_visualizer.render_settings import list_available_settings

COMMANDS = frozenset(
    {
        "ctrl",
        "ctrl_group",
        "mode",
        "sim",
        "speed",
        "camera",
        "render",
        "settings",
        "stream",
        "replay",
        "ping",
    }
)

_SIM_CMDS = frozenset({"play", "pause", "step", "reset"})
_MODES = frozenset({"absolute", "additive"})

# Types whose messages fully supersede an earlier one of the same type: only the last
# matters. ``ctrl`` and ``ctrl_group`` are merged instead (see _coalesce_*).
_LAST_WINS = frozenset({"mode", "speed", "camera", "render", "settings", "stream", "replay"})


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


def parse_command(raw) -> Dict:
    """Validate one client command. Accepts a JSON string or a dict.

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
        if len(out) == 1:
            raise CommandError("'camera' needs 'named' or at least one of az/el/dist/lookat")
        return out

    if kind == "render":
        values = cmd.get("set")
        if not isinstance(values, dict):
            raise CommandError("'render' requires a 'set' object")
        return {"t": "render", "set": dict(values)}

    if kind == "settings":
        name = cmd.get("load")
        if not isinstance(name, str) or not name:
            raise CommandError("'settings' requires a 'load' name")
        # Whitelisted by NAME against the bundled presets, never accepted as a path.
        # Visualizer._resolve_settings_path's first branch is `if Path(x).is_file()`, which is
        # correct for its own callers (a user naming a settings file on the command line) but
        # means an unvalidated wire value makes the server open() and json.load() any path a
        # client names -- and then echo the whitelisted keys back in the scene message. The
        # bind address defaults to 127.0.0.1 but --host widens it.
        available = list_available_settings()
        if name not in available:
            raise CommandError(
                "'settings.load' must be one of the available settings presets; "
                "{0!r} is not (available: {1})".format(name, ", ".join(available))
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
            if not isinstance(cmd["load"], str):
                raise CommandError("'replay.load' must be a path string")
            out["load"] = cmd["load"]
        if "frame" in cmd:
            out["frame"] = int(_num(cmd, "frame", lo=0, hi=1e9))
        if "play" in cmd:
            out["play"] = bool(cmd["play"])
        if len(out) == 1:
            raise CommandError("'replay' needs at least one of load/frame/play")
        return out

    return {"t": "ping"}


def coalesce(cmds: List[Dict]) -> List[Dict]:
    """Collapse redundant commands, keeping the order of the survivors.

    ``_LAST_WINS`` types keep only their final message. ``ctrl`` sets merge key-by-key with
    later values winning. ``ctrl_group`` keeps the last gain per group. ``sim`` messages are
    events (play/pause/step/reset) and are all preserved in order.
    """
    last_index: Dict[str, int] = {}
    merged_ctrl: Dict[str, float] = {}
    ctrl_index = None
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
        else:
            out.append(cmd)
    return out
