"""Stateful core of the streaming viewer: one model, one data, one persistent Renderer,
stepped through a swappable :class:`~mujoco_visualizer.serve.backends.PhysicsBackend`.

No threads and no networking live here -- :class:`SimLoop` supplies the thread and
``app.py`` the sockets -- so the whole control-and-stepping surface is testable in pytest
with no server running.

``Session`` does not call ``mj_step`` itself: it owns a *backend* (default
:class:`~mujoco_visualizer.serve.backends.CpuBackend`, sharing this Session's own ``MjData``)
and a :class:`~mujoco_visualizer.Visualizer`, and renders from ``self.data`` after syncing it
from the backend each step. That split keeps control composition/clamping/rendering here and
stepping/state in the backend, so a later device-resident backend (Warp/MJX) drops in without
touching this file.
"""

import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import mujoco
import numpy as np
import simplejpeg

from mujoco_visualizer import Visualizer, list_available_settings
from mujoco_visualizer.render_settings import PRESET_NAME_RE
from mujoco_visualizer.serve.backends import CpuBackend, PhysicsBackend, UnknownKeyframe
from mujoco_visualizer.serve.controls import actuator_group_map, build_control_tree
from mujoco_visualizer.serve.locks import build_joint_qpos_map
from mujoco_visualizer.visualizer import (
    _apply_forces_vis,
    actuator_names,
    apply_tendon_activation,
    build_actuator_tendon_map,
    build_ctrl_name_map,
    default_tendon_ctrl_full_scale,
)


class Diverged(RuntimeError):
    """Physics produced a non-finite state. Session state is rolled back to the last good
    step before this is raised, so the caller can still render and offer a reset."""


class CtrlWidthMismatch(ValueError):
    """:meth:`Session.set_qpos` raised this because the ``ctrl`` it was handed does not have
    exactly as many entries as this Session's replay ctrl map expects.

    A distinct subclass of ``ValueError`` -- not the plain one :meth:`set_qpos` raises for a
    non-finite qpos -- so a caller that needs to tell the two apart (``SimLoop._write_replay_
    qpos``, which must report a bad ctrl width as a non-pausing ``kind='command'`` error while
    still letting a bad qpos get the ``kind='replay'``, paused treatment it deserves) can catch
    exactly this one without also swallowing the other.

    Carries :attr:`expected_width` so a catcher can build a validly-shaped all-zero ctrl
    vector and hand it straight back through :meth:`Session.set_qpos`'s own ``ctrl``
    parameter -- the same seam a good ctrl vector takes -- rather than reaching for some other,
    separately-callable way to clear the stored visualisation ctrl. There is deliberately no
    such separate method: even though this ctrl is visualisation-only now (it does not feed
    ``mj_forward`` or the constraint solve -- see :meth:`Session.set_qpos`), a zeroing path
    reachable outside the one call that also writes qpos is still a footgun a future call site
    could trip over long after the reasoning why that matters has scrolled out of view. So the
    only zeroing path left is the same one a good ctrl vector takes, in the same method call
    that writes qpos.
    """

    def __init__(self, message: str, expected_width: int):
        super().__init__(message)
        self.expected_width = int(expected_width)


# mjtWarning splits into two classes that must NOT be conflated:
#
# - Divergence: the state itself is corrupt. MuJoCo's own check for "Nan, Inf or huge value"
#   fires here, and it repairs the offending DOF in place before returning -- so by the time
#   step() looks at qpos/qvel afterwards, the corruption can already be gone even though a
#   real blow-up happened. Comparing this counter before/after backend.step() is what
#   actually catches that, where an isfinite-only check cannot.
# - Capacity/quality: the model hit a fixed-size buffer or a numerically stiff-but-valid
#   config (e.g. the contact buffer filling up). These are expected to happen, are not a
#   corrupt state, and must stay visible-but-non-fatal -- pausing the viewer on a full
#   contact buffer would hide exactly the thing a user wants to watch develop.
# Wire name -> vis_state['camera'] key. The protocol (and viewer.js) speak the short
# az/el/dist that a drag handler naturally produces; Visualizer._cfg_to_mjvcamera reads the
# long names. This mapping is the only place the two meet -- deliberately here rather than by
# renaming either side, since the protocol is public to every connected browser and
# vis_state's key names are shared with the settings JSON files on disk.
_CAMERA_WIRE_KEYS = {
    "az": "azimuth",
    "el": "elevation",
    "dist": "distance",
    "lookat": "lookat",
}

_FATAL_WARNINGS = (
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)


def _carry_vis_state_across_swap(vis_state: Dict, model: mujoco.MjModel) -> Dict:
    """Drop ``geom_colors`` entries that don't exist on *model*, and re-apply ``forces`` onto
    *model*, in place; return *vis_state*.

    Category colours, lighting, floor, flags and camera are all model-agnostic and carry
    across a swap unchanged. ``geom_colors`` is the one exception: it is keyed by geom id, and
    ids are model-specific -- on a real overlay pair the second model has roughly twice the
    geoms of the first. Swapping from a model with more geoms to one with fewer would
    otherwise leave ids past the smaller model's ``ngeom`` pointing at geoms that no longer
    exist (wrong at best, an index error at worst). Going the other way every existing id is
    still a valid prefix, so nothing is dropped.

    Keys are coerced with ``int()`` rather than compared as-is. ``load_settings`` produces int
    keys, but ``apply_render`` walks dotted keys verbatim, so a wire command
    ``{"t":"render","set":{"geom_colors.5":"#f00"}}`` inserts the *string* ``"5"`` -- and
    ``"5" < model.ngeom`` raises TypeError, which surfaced as the ghost toggle failing rather
    than as anything to do with colours. A key that is not an int at all is dropped rather
    than raising: it cannot name a geom, so it can only be junk.

    ``forces`` is the opposite kind of trap: nothing here needs FIXING to stay valid on the
    new model (all five fields are plain floats, meaningful on any model) -- but nothing
    APPLIES them either. ``forces`` lives on ``MjModel.vis``, not on the per-render
    ``MjvOption``, so it is not "model-agnostic" the way lighting/floor/camera are: it lives on
    the model object itself, and ``rebind_model`` (called by ``swap_model`` just before this
    function) points at a brand-new model carrying its OWN MJCF vis values, not the ones
    ``vis_state`` still claims. Left alone, the new model would silently render with its own
    MJCF's force-arrow scaling regardless of what the user had dialled in -- structurally the
    same trap that a stale joint map shipped as (see ``loop.py``'s lock revalidation across a
    swap) -- so it is written back here, directly, rather than left to whatever the next
    render happens to do.
    """
    geom_colors = vis_state.get("geom_colors")
    if geom_colors:
        kept = {}
        for gid, hexcolor in geom_colors.items():
            try:
                index = int(gid)
            except (TypeError, ValueError):
                continue
            if 0 <= index < model.ngeom:
                kept[index] = hexcolor
        vis_state["geom_colors"] = kept
    forces = vis_state.get("forces")
    if forces:
        _apply_forces_vis(forces, model)
    return vis_state


class Session:
    """Owns the simulation and how it is drawn.

    Control composition has two modes. In ``"absolute"`` the per-actuator values set via
    :meth:`set_ctrl` are written to ``data.ctrl`` directly. In ``"additive"`` they are added
    on top of the attached controller's output -- so sliders act as perturbations, mirroring
    how ``scripts/vnc_explorer/sim.build_stim`` already perturbs a running model. Either way
    the result is clamped to each actuator's ``ctrlrange``, because clamping is what is
    wanted mid-drag; rejecting an out-of-range drag would just make the slider feel broken.
    The clamped result is handed to the backend via :meth:`PhysicsBackend.set_ctrl` --
    ``Session`` never writes ``data.ctrl`` itself, so this composition applies unchanged
    regardless of which backend is attached.
    """

    def __init__(
        self,
        *,
        xml_path: Optional[str] = None,
        model: Optional[mujoco.MjModel] = None,
        anatomy=None,
        settings: Optional[str] = None,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 75,
        backend: Optional[PhysicsBackend] = None,
        alt_model: Optional[mujoco.MjModel] = None,
        user_settings_dir: Optional[Path] = None,
        scene_modifiers: Optional[Sequence[Callable]] = None,
        actuator_color_schemes: Optional[Dict[str, Dict[str, Callable]]] = None,
        **viz_kwargs,
    ):
        # Where `save_settings_as` writes named presets, and where `load_settings` looks for
        # one beyond the bundled set -- NEVER the package's own `settings/` directory (see
        # `save_settings_as`). None (the default) just means this session has nowhere of its
        # own to save to yet; loading still works against the bundled presets.
        # .resolve() (not just Path(...)): Visualizer.save_settings only redirects a BARE
        # name into the package's bundled settings/ dir when the target path is relative
        # with parent '.' -- a user_settings_dir passed in as exactly '.' would collapse to
        # that same shape (`Path('.') / 'x.json'` normalises to `Path('x.json')`, parent
        # '.') and hit that redirect. Resolving to an absolute path here removes the
        # ambiguity once, rather than depending on every future caller happening to pass an
        # already-absolute directory.
        self.user_settings_dir = (
            Path(user_settings_dir).resolve() if user_settings_dir is not None else None
        )

        # Extra per-frame scene decoration (e.g. recorded force-sensor arrows drawn via
        # ``add_arrow_to_scene``): forwarded verbatim to ``Visualizer.render_with``'s existing
        # ``modify_scene_fns`` by :meth:`render` below, on every call -- the LIVE path only,
        # by construction, since this list lives on ``Session`` and ``ExportJob`` (serve/
        # export.py) builds its own independent ``Visualizer`` on its own thread and never
        # touches a ``Session`` at all. An export therefore does NOT currently honour anything
        # registered here; see :meth:`render`'s docstring for why this is flagged as a finding
        # rather than worked around in this seam.
        #
        # A plain mutable list, not a private attribute behind add/remove methods: callers
        # already compose ``modify_scene_fns`` as "a sequence of callables" everywhere else in
        # this package (``render_with``, ``render_video``, ``render_video_pan``), and a caller
        # here needs nothing beyond append/remove/clear/reassign, all of which a list already
        # gives for free. The constructor argument seeds it for a caller that knows its
        # modifiers up front; mutating the attribute after construction (``session.
        # scene_modifiers.append(fn)``) is equally supported and is how a caller that discovers
        # or toggles a modifier later (e.g. the force-arrows launcher) is expected to use it.
        self.scene_modifiers: List[Callable] = (
            list(scene_modifiers) if scene_modifiers is not None else []
        )

        self.viz = Visualizer(
            xml_path=xml_path,
            model=model,
            anatomy=anatomy,
            settings_json=settings,
            **viz_kwargs,
        )
        self.model = self.viz.model
        self.data = self.viz.data

        # Default backend shares this Session's own MjData, which is exactly what makes its
        # sync_to a no-op: the data it steps IS the data rendered below.
        self.backend: PhysicsBackend = (
            backend if backend is not None else CpuBackend(self.model, self.data)
        )

        self.width = int(width)
        self.height = int(height)
        self.jpeg_quality = int(jpeg_quality)
        self._renderer = self.viz.make_renderer(height=self.height, width=self.width)

        # A second prebuilt model the client can swap to (the reference-ghost pair). Kept
        # compiled from the start because compiling is slow and, more importantly, because
        # the swap has to happen on THIS thread -- the one holding the GL context -- and a
        # client command must not be the thing that triggers a first-time compile there.
        self._models = {"primary": self.model}
        if alt_model is not None:
            self._models["alt"] = alt_model
        self._active_model = "primary"

        # The order a replay ctrl vector is assumed to arrive in: the PRIMARY (policy) model's
        # own actuator order. This never changes across a swap -- the primary model itself is
        # never swapped away, only which model is ACTIVE -- so it is computed once here rather
        # than rebuilt alongside `_ctrl_map` below.
        self._primary_actuator_names = self._actuator_names(self._models["primary"])
        # {index into a primary-ordered replay ctrl vector -> index into data.ctrl on the
        # CURRENTLY ACTIVE model}, built by matching actuator NAMES (see _build_ctrl_map).
        # Rebuilt by swap_model whenever the active model changes -- see its own comment for
        # why a stale map here is exactly the silent-corruption failure mode this exists to
        # avoid.
        self._ctrl_map = self._build_ctrl_map()

        # Actuator colour schemes, keyed by the name that appears in
        # vis_state['tendons']['color_by']. Each value is
        # {"color": (name) -> hex|rgba, "group": (name) -> group_name}.
        #
        # A PARAMETER, not a palette shipped here: turning 'mu_T1_28a_left' into a colour is
        # specific to one model's actuator naming, and this class is model-agnostic for tendons
        # exactly as it is for lighting, floor and camera. An unregistered scheme name resolves
        # to {} and therefore to build_actuator_tendon_map's existing solid-red fallback, so a
        # settings file written against another model still renders (design D3).
        #
        # The "group" half is what a legend needs: grouping by resolved hex would label the
        # legend with colour codes instead of muscle-group names. Keeping both halves in one
        # entry means a scheme cannot supply a palette without the labels that explain it.
        self._actuator_color_schemes = dict(actuator_color_schemes or {})
        self._rebuild_tendon_state()

        self._tree = build_control_tree(self.model)
        self._group_of = actuator_group_map(self._tree)
        self._id_of = {
            a["name"]: a["id"] for g in self._tree["groups"] for a in g["actuators"]
        }

        self._offset = np.zeros(self.model.nu, dtype=np.float64)
        self._gain: Dict[str, float] = {}
        self._mode = "absolute"
        self._camera = None

        self._controller = None
        self._controller_out: Optional[np.ndarray] = None

        self._lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._hi = self.model.actuator_ctrlrange[:, 1].copy()
        self._limited = self.model.actuator_ctrllimited.astype(bool)

        # Per-frame warning deltas (see :meth:`new_warnings`) are measured against this.
        self._warn_baseline = np.zeros(int(mujoco.mjtWarning.mjNWARNING), dtype=np.int64)

        # A fresh MjData has geom_xpos all-zero and xquat all-zero, and mjv_updateScene draws
        # from exactly those precomputed arrays -- so without this the frames published before
        # the first Play (SimLoop publishes every tick but starts with _playing=False) draw the
        # whole model collapsed at the origin, on every backend.
        mujoco.mj_forward(self.model, self.data)
        self._snapshot()

    # -- controller -----------------------------------------------------------

    def attach_controller(self, controller) -> None:
        """Attach an object with ``rate_hz``, ``step(model, data)`` and ``readout()``.

        Attaching also flips to additive mode: with a controller driving ``ctrl``, absolute
        sliders would fight it on every step.
        """
        self._controller = controller
        self._controller_out = None
        self._mode = "additive"

    @property
    def controller_rate_hz(self) -> Optional[float]:
        return None if self._controller is None else float(self._controller.rate_hz)

    def advance_controller(self) -> None:
        """Call the controller once and cache its output. The loop calls this at
        ``rate_hz``, not once per physics step."""
        if self._controller is None:
            return
        out = self._controller.step(self.model, self.data)
        if out is not None:
            self._controller_out = np.asarray(out, dtype=np.float64)

    def readout(self) -> Dict:
        if self._controller is None:
            return {}
        return dict(self._controller.readout())

    # -- control ---------------------------------------------------------------

    def set_ctrl(self, values: Dict[str, float]) -> None:
        """Merge per-actuator values by name. Raises KeyError on an unknown name."""
        for name, value in values.items():
            if name not in self._id_of:
                raise KeyError(f"unknown actuator {name!r}")
            self._offset[self._id_of[name]] = float(value)

    def set_group_gain(self, group: str, gain: float) -> None:
        if group not in {g["id"] for g in self._tree["groups"]}:
            raise KeyError(f"unknown group {group!r}")
        self._gain[group] = float(gain)

    def set_ctrl_mode(self, mode: str) -> None:
        if mode not in ("absolute", "additive"):
            raise ValueError(f"mode must be 'absolute' or 'additive', got {mode!r}")
        self._mode = mode

    def _compose_ctrl(self) -> None:
        scaled = self._offset.copy()
        if self._gain:
            for i in range(self.model.nu):
                g = self._gain.get(self._group_of.get(i, ""), 1.0)
                if g != 1.0:
                    scaled[i] *= g
        if self._mode == "additive" and self._controller_out is not None:
            raw = self._controller_out + scaled
        else:
            raw = scaled
        np.clip(raw, self._lo, self._hi, out=raw, where=self._limited)
        self.backend.set_ctrl(raw)

    # -- stepping --------------------------------------------------------------

    def _snapshot(self) -> None:
        self._good = (self.data.qpos.copy(), self.data.qvel.copy(), float(self.data.time))

    def _restore(self) -> None:
        """Roll host AND backend state back to the last good step.

        Pushing it through :meth:`PhysicsBackend.set_state` is what makes the rollback real
        for a backend whose authoritative state lives off-host: rewriting only ``self.data``
        leaves a device-resident backend (``WarpBackend``) holding the diverged state, so
        every subsequent :meth:`step` re-diverges until :meth:`reset` -- and the "rolled back
        to the last good step" message would be a lie.

        Only qpos/qvel/time are restored; actuator activation (``data.act``, non-empty once
        the muscle conversion sets ``dyntype=MUSCLE``) is not snapshotted, so the divergence
        message below is deliberately explicit about what was rolled back.
        """
        qpos, qvel, t = self._good
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.time = t
        self.backend.set_state(qpos, qvel, t)
        mujoco.mj_forward(self.model, self.data)

    def _is_finite(self) -> bool:
        return bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())

    def _fatal_warning_count(self) -> int:
        """Sum of the divergence-class ``mjtWarning`` counters (see ``_FATAL_WARNINGS``).

        Cumulative for the life of ``self.data``, so what matters is the *delta* across one
        ``step()`` call, not the absolute value.
        """
        return sum(int(self.data.warning[int(w)].number) for w in _FATAL_WARNINGS)

    def step(self, n: int) -> None:
        """Advance physics *n* steps via the backend, then sync its state onto ``self.data``.

        Raises :class:`Diverged`, rolling back to the last good step, if physics diverged
        during those *n* steps -- checked on ``self.data`` *after* the sync, since that is
        the state actually rendered.

        Divergence is detected by an *increase* in the divergence-class warning counters
        (``_FATAL_WARNINGS``) across the call, not by an ``isfinite`` check alone: MuJoCo's
        own "Nan, Inf or huge value" check fires on exactly this condition and then silently
        repairs the offending DOF before returning, so a real blow-up can leave ``qpos``/
        ``qvel`` fully finite by the time this method looks at them. The ``isfinite`` check
        is kept only as a cheap backstop for whatever that warning mechanism doesn't cover.

        Capacity/quality warnings (contact buffer full, etc.) are deliberately excluded from
        this check -- they are expected, not corrupt, and must stay visible-but-non-fatal via
        :meth:`warnings`, not pause the viewer.
        """
        self._compose_ctrl()
        warn_before = self._fatal_warning_count()
        self.backend.step(int(n))
        self.backend.sync_to(self.data)
        # MANDATORY, and deliberately unconditional. mjv_updateScene renders from the
        # PRECOMPUTED xpos/xquat/geom_xpos, which only mj_forward/mj_step populate --
        # ``sync_to`` writes qpos/qvel/time and nothing else. So for any backend whose
        # authoritative state lives off-host (``WarpBackend``), skipping this pins the drawn
        # pose to whatever derived state the MjData last held, for the whole session, while
        # sim_time and rtf keep advancing convincingly. There is deliberately NO
        # "skip it for CpuBackend" branch: ~1 ms against a 40 ms tick does not justify one,
        # and a conditional is exactly how this class of bug comes back.
        mujoco.mj_forward(self.model, self.data)
        diverged = (self._fatal_warning_count() > warn_before) or not self._is_finite()
        if diverged:
            self._restore()
            raise Diverged(
                "physics diverged (fatal MuJoCo warning or non-finite qpos/qvel); "
                "host and backend rolled back to the last good step (qpos/qvel/time)"
            )
        self._snapshot()

    def warnings(self) -> Optional[str]:
        """Comma-joined names of MuJoCo warnings raised so far, or None.

        Surfaced per frame rather than treated as fatal: seeing "contact buffer full" as it
        develops is far more useful than only learning about it after a blow-up.
        """
        names = [
            mujoco.mjtWarning(i).name
            for i in range(mujoco.mjtWarning.mjNWARNING)
            if self.data.warning[i].number > 0
        ]
        return ", ".join(names) if names else None

    def new_warnings(self) -> Optional[str]:
        """Warnings raised SINCE THE LAST CALL, or None. This is what rides on ``frame_meta``.

        :meth:`warnings` reports MuJoCo's cumulative counters, which are only cleared by
        :meth:`reset` -- so once anything ever warns, a banner fed from it is pinned to a
        stale string forever, and a permanently-pinned banner then permanently masks whatever
        else wants to use that slot. A per-frame delta comes and goes with the condition: a
        contact buffer that is still filling re-increments every step and so stays visible,
        while one that stopped clears itself.

        STATEFUL: each call re-baselines, so it must be called exactly once per frame (from
        ``SimLoop._publish``). Use :meth:`warnings` for the cumulative view.
        """
        names = []
        for i in range(int(mujoco.mjtWarning.mjNWARNING)):
            count = int(self.data.warning[i].number)
            if count > self._warn_baseline[i]:
                names.append(mujoco.mjtWarning(i).name)
            self._warn_baseline[i] = count
        return ", ".join(names) if names else None

    @staticmethod
    def _actuator_names(model: mujoco.MjModel) -> list:
        """Every actuator name on *model*, in id order. Delegates to the module-level
        :func:`~mujoco_visualizer.visualizer.actuator_names` (task 15c fix round 1 collapsed
        three copies of this idiom -- this one, ``build_ctrl_name_map``'s own, and a private
        one in serve/export.py -- into that single shared implementation); kept as a
        ``Session`` method since callers throughout this class already call it as
        ``self._actuator_names(...)``."""
        return actuator_names(model)

    def _build_ctrl_map(self) -> np.ndarray:
        """``{index into a primary-ordered replay ctrl vector -> index into data.ctrl on the
        CURRENTLY ACTIVE model}``, built by matching actuator NAMES -- never by position.

        Delegates to the module-level :func:`~mujoco_visualizer.visualizer.build_ctrl_name_map`
        (extracted so ``ExportJob``, which has no ``Session`` to call into, can build the exact
        same map) -- this method keeps no separate implementation of its own.

        The reference-ghost pair doubles the actuator count (``nu`` 272 -> 544 on the real
        models), so a 272-wide replay ctrl vector cannot be written into a 544-wide
        ``data.ctrl`` positionally: assuming the policy's actuators occupy a fixed prefix of
        the doubled model is exactly the attachment-order assumption that produced a real
        data-corruption bug on this branch (a prefix heuristic silently mis-assigning ghost
        counterparts). ``locks.pair_with_suffix`` already solves the equivalent problem for
        joint names for the same reason; this does it for actuator ids.

        A primary name with no match on the active model (there is never one when the active
        model IS primary) maps to ``-1`` and is simply never written -- see :meth:`set_qpos`.
        The reference half of a doubled model is a kinematic overlay, not driven by anything,
        so its own actuators likewise never appear as a TARGET of this map and are left at
        whatever :meth:`set_qpos` zeroed them to.
        """
        return build_ctrl_name_map(self._primary_actuator_names, self.model)

    def _rebuild_tendon_state(self) -> None:
        """(Re)compute everything :meth:`_apply_tendon_activation_vis` needs from the CURRENTLY
        ACTIVE model: the actuator->tendon map/colours (via :meth:`_rebuild_tendon_colors`), a
        snapshot of the model's own tendon_rgba/tendon_width to restore to when the group is
        disabled, a fixed per-session FALLBACK reference scale to normalise activation by, and
        the ``_vis_ctrl`` store itself (see below).

        Called from ``__init__`` and again from :meth:`swap_model` ONLY -- never for a
        mid-session ``color_by`` change, which must go through :meth:`_rebuild_tendon_colors`
        alone. See that method's docstring for why re-snapshotting here mid-session would be
        destructive.

        Called from those two sites exactly like :meth:`_build_ctrl_map` right above each call
        site -- the reference-ghost swap roughly
        doubles ``ntendon``/``nu`` (260 -> 520 / 272 -> 544 on the real models), so a
        map/snapshot/store built against the OLD model would address the wrong tendons (or go
        out of range) on the new one. Unlike ``_ctrl_map``, nothing here is matched by NAME
        across the two models: this map is used only against whichever model is currently
        active, never to translate an id from one model to the other, so there is no
        primary/alt pairing to get wrong here.

        ``_tendon_default_ctrl_full_scale`` is fixed once here, not recomputed every frame, and
        is only ever a FALLBACK -- ``_apply_tendon_activation_vis`` prefers
        ``vis_state['tendons']['ctrl_full_scale']`` whenever that key is present (it always is,
        after ``Visualizer.__init__`` populates it with this exact value; this attribute is what
        a legacy settings file saved before that key existed would fall back to instead of
        raising). See :func:`default_tendon_ctrl_full_scale` for why this model-only ceiling is
        a poor normalisation reference on real data, and why the fix is an overridable knob
        rather than recomputing anything from ``data.ctrl`` here.
        """
        self._rebuild_tendon_colors()
        self._tendon_orig_rgba = self.model.tendon_rgba.copy()
        self._tendon_orig_width = self.model.tendon_width.copy()
        self._tendon_default_ctrl_full_scale = default_tendon_ctrl_full_scale(
            self.model, self._tendon_act_to_ten
        )
        # The visualisation-only ctrl vector :meth:`_apply_tendon_activation_vis` reads,
        # indexed like ``data.ctrl`` on the CURRENTLY ACTIVE model (one entry per actuator,
        # active-model order) -- see :meth:`set_qpos`, the only writer. Reset to all-zero here,
        # not merely resized, on every call: a swap changes ``model.nu`` (272 -> 544 on the
        # real models), so an old-sized array would be the wrong shape for
        # ``apply_tendon_activation`` below, and carrying over stale VALUES at whatever
        # addresses happen to still be in range would colour the new model's tendons from the
        # old model's last activation instead of leaving them inert until the next replay
        # frame writes a real one -- the same "stale-but-plausible visual" failure mode
        # :meth:`_apply_tendon_activation_vis` already guards against on disable.
        self._vis_ctrl = np.zeros(self.model.nu, dtype=np.float64)

    def _rebuild_tendon_colors(self) -> None:
        """(Re)compute the actuator->tendon map and its ``base_rgba`` for the CURRENT
        ``vis_state['tendons']['color_by']``, and record which scheme they were built for.

        Split out of :meth:`_rebuild_tendon_state` because a scheme change is a MID-SESSION
        rebuild, and the rest of that method must not run again then: it snapshots
        ``model.tendon_rgba`` into ``_tendon_orig_rgba``, which is correct at ``__init__``/
        ``swap_model`` (nothing has touched the tendons) and destructive once
        ``apply_tendon_activation`` has overwritten those arrays -- the snapshot would capture
        activation colours as the model's own, and
        :meth:`_apply_tendon_activation_vis` restores to it on every frame the group is off,
        permanently. Splitting by NAME rather than adding a ``snapshot=False`` argument makes
        the unsafe call unreachable rather than merely discouraged.

        Deliberately does NOT recompute ``_tendon_default_ctrl_full_scale`` or reset
        ``_vis_ctrl``: neither depends on the colour scheme, and zeroing ``_vis_ctrl`` here
        would blank the activation for one frame every time a dropdown changed.
        """
        scheme_name = self.viz.vis_state.get("tendons", {}).get("color_by", "function")
        scheme = self._actuator_color_schemes.get(scheme_name, {})
        self._tendon_act_to_ten, self._tendon_base_rgba = build_actuator_tendon_map(
            self.model, scheme.get("color")
        )
        self._tendon_color_scheme = scheme_name

    def _apply_tendon_activation_vis(self) -> None:
        """Drive ``vis_state['tendons']`` from :attr:`_vis_ctrl` for the frame about to be
        rendered -- never from ``data.ctrl``. ``data.ctrl`` holds whatever the constraint solve
        actually used, which for a replaying viewer is nothing (:meth:`set_qpos` no longer
        writes it); ``_vis_ctrl`` is the visualisation-only vector :meth:`set_qpos` maintains
        instead, and is what this method must colour/thicken tendons from regardless of
        whether physics is stepping or a recorded clip is scrubbing.

        Tolerates a PARTIAL ``vis_state['tendons']`` dict -- every field is read with
        ``.get(..., default)``, never indexed directly -- because ``render.set`` merges one
        wire key at a time (see ``Session.apply_render``) and a settings preset can likewise
        mention only some of this group's fields.

        Disabling does not merely stop updating the tendons: it actively restores
        ``model.tendon_rgba``/``model.tendon_width`` to this model's own values, every frame
        the group is off, not just the frame it was switched off on. A live loop has no
        natural "end of clip" the way ``render_video_pan`` does to restore once after its last
        frame -- leaving the LAST frame's activation frozen on screen the moment the toggle
        flips off would be a confident, wrong picture with nothing to signal it changed.

        ``ctrl_max`` is ``vis_state['tendons']['ctrl_full_scale']`` when present -- a caller
        holding the real rollout (the launcher) is expected to override it with a measured
        ``|ctrl|`` percentile -- falling back to ``self._tendon_default_ctrl_full_scale`` (the
        model-only ceiling computed in :meth:`_rebuild_tendon_state`) only for a settings file
        that predates this key.
        """
        tendons = self.viz.vis_state.get("tendons", {})
        if not tendons.get("enabled", False):
            self.model.tendon_rgba[:] = self._tendon_orig_rgba
            self.model.tendon_width[:] = self._tendon_orig_width
            return
        # A scheme change is picked up HERE, by comparison, rather than by hooking a write
        # path: apply_render is not the only writer -- load_settings lands a whole
        # vis_state['tendons'] dict, including color_by -- and one comparison against a cached
        # string covers both, plus swap_model. Costs one dict lookup and a string compare per
        # frame.
        if tendons.get("color_by", "function") != self._tendon_color_scheme:
            self._rebuild_tendon_colors()
        apply_tendon_activation(
            self.model,
            self._vis_ctrl,
            self._tendon_act_to_ten,
            self._tendon_base_rgba,
            tendon_width=tendons.get("max_width", 0.003),
            tendon_min_width=tendons.get("min_width", 0.0005),
            tendon_alpha_min=tendons.get("min_alpha", 0.05),
            tendon_baseline=tendons.get("baseline", 0.0),
            ctrl_max=tendons.get("ctrl_full_scale", self._tendon_default_ctrl_full_scale),
        )

    def set_qpos(self, qpos: Sequence[float], ctrl: Optional[Sequence[float]] = None) -> None:
        """Write state directly, no stepping. Used by replay scrubbing.

        Rejects non-finite input outright, before writing or snapshotting anything: a NaN/Inf
        qpos (reachable via a malformed replay-scrub or client message) would otherwise flow
        straight into :meth:`_snapshot`, permanently poisoning the rollback target that every
        later :meth:`step` restores to -- turning one bad frame into a session that raises
        :class:`Diverged` forever until :meth:`reset`.

        ``ctrl``, when given, is VISUALISATION-ONLY: it is mapped into :attr:`_vis_ctrl` (read
        by :meth:`_apply_tendon_activation_vis` to grow/shrink and colour muscle tendons) and
        is deliberately never written to ``data.ctrl``. A recorded rollout's ``ctrl`` is not
        trustworthy as a physics input during replay -- forces re-derived from state disagree
        with the rollout's own recorded sensors by roughly 60x at correlation ~0.3 -- so
        scattering it onto ``data.ctrl`` ahead of the ``mj_forward`` below would perturb the
        constraint solve, actuator forces, contact forces and sensor values for no benefit;
        this method used to do exactly that, which is precisely the behaviour removed here.
        The recorded/trustworthy forces for a replaying viewer come from the rollout's own
        recorded sensors, rendered by a sibling feature, not from ``mj_forward``.

        The mapping itself is unchanged: ``ctrl`` is scattered into :attr:`_vis_ctrl` through
        :attr:`_ctrl_map` (see :meth:`_build_ctrl_map`), i.e. by actuator NAME against whichever
        model is currently active, not by position -- so this is safe to call unchanged whether
        or not a reference-ghost overlay is active. :attr:`_vis_ctrl` is rebuilt from zero on
        every call (never updated in place): the reference half of a doubled model is a
        kinematic overlay that is never driven, so its actuators are deliberately left at zero
        rather than carrying over whatever they held before -- and a primary-ordered name with
        no match on the active model (``_ctrl_map`` entry ``-1``) is simply never written, same
        as before.

        Omitting ``ctrl`` (the default) leaves :attr:`_vis_ctrl` completely untouched -- exactly
        as omitting it used to leave ``data.ctrl`` untouched -- so every existing caller that
        only ever wrote qpos keeps behaving exactly as before. ``data.ctrl`` itself is never
        touched by this method at all now, whether or not ``ctrl`` is given: ``mj_forward``
        below reads ``data.ctrl`` to compute derived quantities (``actuator_force`` and
        friends) but does not write it, so it is simply left at whatever the physics backend
        or the interactive-slider path (:meth:`_compose_ctrl`, via :meth:`step`) last put there
        -- typically zero on a session that has never stepped or received a slider command.
        """
        arr = np.asarray(qpos, dtype=np.float64)
        if not np.isfinite(arr).all():
            raise ValueError("set_qpos: qpos contains non-finite values (NaN/Inf)")
        if ctrl is not None:
            ctrl_arr = np.asarray(ctrl, dtype=np.float64)
            if ctrl_arr.shape != (len(self._ctrl_map),):
                raise CtrlWidthMismatch(
                    f"set_qpos: ctrl has shape {ctrl_arr.shape}, expected "
                    f"({len(self._ctrl_map)},) to match this session's replay ctrl map",
                    expected_width=len(self._ctrl_map),
                )
            vis_ctrl = np.zeros(self.model.nu, dtype=np.float64)
            valid = self._ctrl_map >= 0
            vis_ctrl[self._ctrl_map[valid]] = ctrl_arr[valid]
            self._vis_ctrl = vis_ctrl
        self.data.qpos[:] = arr
        mujoco.mj_forward(self.model, self.data)
        self._snapshot()

    def reset(self) -> None:
        """Reset to the fly's rest pose, then sync that state from the backend.

        Looks up the ``'default_pose'`` keyframe BY NAME (never index 0 -- on the composed
        fly+floor model index 0 is an unrelated keyframe) and applies NO standing-height
        correction: ``default_pose`` is the experimentally determined muscle rest point with
        the fly *hanging*, deliberately not a standing pose, and it will not hold against a
        floor at ``ctrl=0``. Falls back to ``mj_resetData`` when the model has no such
        keyframe (e.g. the bare test fixtures here), rather than raising.
        """
        try:
            self.backend.reset_to_keyframe("default_pose")
        except UnknownKeyframe:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
        self.backend.sync_to(self.data)
        # Same reason as in step(): sync_to writes only qpos/qvel/time, and mjv_updateScene
        # needs the derived xpos/xquat/geom_xpos -- without this the first post-reset frame
        # keeps drawing the pre-reset pose on a device-resident backend.
        mujoco.mj_forward(self.model, self.data)
        for i in range(mujoco.mjtWarning.mjNWARNING):
            self.data.warning[i].number = 0
        self._warn_baseline[:] = 0
        self._controller_out = None
        self._snapshot()

    # -- rendering -------------------------------------------------------------

    def render(self) -> np.ndarray:
        """Render the current frame, including whatever :attr:`scene_modifiers` holds.

        Forwarded to :meth:`Visualizer.render_with`'s existing ``modify_scene_fns`` -- this is
        the ONLY seam that exists for extra per-frame scene geometry (e.g. recorded
        force-sensor arrows) on the live path. It does NOT reach an export: ``ExportJob``
        (serve/export.py) builds its own ``Visualizer`` from a ``vis_state`` snapshot on its
        own thread and calls ``render_with`` with no ``modify_scene_fns`` at all -- it has no
        parameter for one, and nothing here changes that. An exported video therefore
        currently lacks whatever :attr:`scene_modifiers` draws in the live preview; wiring
        ``ExportJob`` to accept and forward its own ``modify_scene_fns`` is a structurally
        separate change (a different class, a different thread, a different constructor) and
        is left as a follow-up rather than worked around here.
        """
        self._apply_tendon_activation_vis()
        return self.viz.render_with(
            self._renderer, camera=self._camera, modify_scene_fns=self.scene_modifiers
        )

    def encode(self, frame: np.ndarray) -> bytes:
        return simplejpeg.encode_jpeg(
            np.ascontiguousarray(frame), quality=self.jpeg_quality, colorspace="RGB"
        )

    def resize(self, width: int, height: int) -> None:
        """Rebuild the Renderer at a new resolution.

        Costs ~400 ms on a mesh-heavy model because every mesh re-uploads, so this is a
        deliberate operation the loop pauses around -- never per frame.
        """
        if (int(width), int(height)) == (self.width, self.height):
            return
        old = self._renderer
        self.width, self.height = int(width), int(height)
        self._renderer = self.viz.make_renderer(height=self.height, width=self.width)
        old.close()

    def set_camera(self, named: Optional[str] = None, **kw) -> None:
        """Point the camera. ``named`` selects a named camera/preset; keyword args
        (az/el/dist/lookat) update the free camera in ``vis_state``.

        The wire names are translated to the ``vis_state['camera']`` keys that
        ``Visualizer._cfg_to_mjvcamera`` actually reads (see :data:`_CAMERA_WIRE_KEYS`).
        Writing ``az``/``el``/``dist`` through verbatim looks like it works -- the keys land in
        the dict and come back out in the scene message -- but the renderer never reads them,
        so dragging the canvas is a silent no-op.
        """
        if named is not None:
            self._camera = named
            return
        self._camera = None
        cam = self.viz.vis_state.setdefault("camera", {})
        touched = False
        for key, value in kw.items():
            if value is None:
                continue
            cam[_CAMERA_WIRE_KEYS.get(key, key)] = value
            touched = True
        if touched:
            # A settings file can pin camera.mode to "named" (Earthy_V1 does, and it is
            # live.py's default), and Visualizer.get_camera short-circuits straight to the XML
            # camera whenever mode == 'named' -- silently discarding the user's drag. Arriving
            # free-camera parameters ARE the request to be on the free camera.
            cam["mode"] = "free"

    def apply_render(self, settings: Dict) -> None:
        """Merge flat render-setting keys into ``vis_state`` (e.g. ``floor.alpha``).

        ``geom_colors`` is the one sub-dict whose keys are not strings: every other producer
        (``load_settings``, the GUIs) keys it by INT geom id, and both consumers agree --
        ``Visualizer._apply_geom_colors`` tests ``if i in geom_overrides`` with an int, and
        ``_carry_vis_state_across_swap`` compares the key against ``model.ngeom``. A dotted
        wire key arrives as text, so writing it through verbatim inserted the string ``"5"``:
        the colour then never applied (the int lookup missed) and the next model swap raised
        ``TypeError: '<' not supported between 'str' and 'int'``, surfacing as the ghost
        toggle failing. Coerced here, at the one place wire keys enter, rather than papered
        over in each consumer.
        """
        for dotted, value in settings.items():
            node = self.viz.vis_state
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = self._descend(node, part, dotted)
            key = parts[-1]
            if parts[:-1] == ["geom_colors"]:
                try:
                    key = int(key)
                except ValueError:
                    raise ValueError(
                        f"'geom_colors' is keyed by geom id; {key!r} is not an integer"
                    ) from None
            if isinstance(node, list):
                node[self._list_index(node, key, dotted)] = value
            else:
                node[key] = value

    @staticmethod
    def _list_index(seq, key, dotted):
        """Resolve a dotted path segment against a list, with an actionable error.

        ``geom_groups``/``site_groups`` are fixed-length lists of booleans, so a wire key
        addresses them by index. Silently ignoring a bad index would leave a UI control that
        appears to do nothing; raising names the bound the client got wrong.
        """
        try:
            idx = int(key)
        except (TypeError, ValueError):
            raise ValueError(
                f"{dotted!r}: {key!r} is not an integer index into a list of {len(seq)}"
            ) from None
        if not 0 <= idx < len(seq):
            raise ValueError(
                f"{dotted!r}: index {idx} out of range for a list of {len(seq)}"
            )
        return idx

    @classmethod
    def _descend(cls, node, part, dotted):
        if isinstance(node, list):
            return node[cls._list_index(node, part, dotted)]
        return node.setdefault(part, {})

    @property
    def camera(self) -> Optional[str]:
        """The named camera/preset currently selected, or ``None`` for the free camera.

        Read-only, and deliberately public: an export job has to render with the camera the
        user pressed the button on, so whatever builds that job needs this value. Export
        factories read ``session._camera`` before this property existed, which made a private
        attribute part of an out-of-package contract. Setting still goes through
        :meth:`set_camera`, which is where the wire-name translation lives.
        """
        return self._camera

    @property
    def active_model_name(self) -> str:
        return self._active_model

    @property
    def primary_actuator_names(self) -> List[str]:
        """The order a replay ctrl vector is assumed to arrive in: the PRIMARY model's own.

        Public because :class:`~mujoco_visualizer.serve.export.ExportJob` *requires* a caller
        passing ``ctrl_frames`` to declare their ordering, and the caller's only correct source
        for it is this session. Its first real caller had to reach into the private attribute --
        a surface that forces its consumer to do that is not finished -- so this exists to make
        the supported thing the reachable one.

        A copy, not the list itself: a caller mutating it would silently desynchronise
        ``_ctrl_map``, which is built from it and rebuilt only on a model swap.
        """
        return list(self._primary_actuator_names)

    def vis_state_snapshot(self) -> Dict:
        """A deep copy of ``vis_state``, safe to hand to another thread.

        An export job renders with the look the user had when they pressed the button, while
        this thread keeps editing ``vis_state`` live. Sharing the dict would let a mid-export
        colour change land halfway through the video -- and could raise "dictionary changed
        size during iteration" in the job.
        """
        return copy.deepcopy(self.viz.vis_state)

    def swap_model(self, which: str) -> None:
        """Point this Session at one of its prebuilt models, rebuilding the renderer.

        Costs ~400-560 ms (every mesh re-uploads), exactly like :meth:`resize`, so it is a
        deliberate operation and never per-frame. Called only from the SimLoop thread, which
        is the thread that owns the GL context.

        The ghost pair cannot be a render flag: it is a second fly compiled into the model
        (nq 101 -> 202). Hiding it by geom group was measured to recover almost none of its
        cost (34.9 ms vs 37.1 ms) because the shadow pass still pays for hidden geometry --
        so an always-ghost model would cost 27 fps even with the ghost invisible.

        Failure safety mirrors :meth:`resize`: nothing owned by this Session (``model``,
        ``data``, ``backend``, ``_renderer``, ``_active_model``) is committed until rebind,
        backend construction, ``mj_forward``, the snapshot, AND the new renderer have all
        succeeded -- ``_renderer`` in particular is never set to ``None``, so a mid-swap
        exception leaves ``render()`` serving the last good frame instead of crashing on it.
        ``self.viz`` itself is mutated in place by ``rebind_model`` before that point (it has
        no transaction of its own), so a failure there is rolled back explicitly by rebinding
        it back to the old model before re-raising.
        """
        if which not in self._models:
            if which in ("primary", "alt"):
                raise ValueError(
                    f"cannot swap to {which!r}: no alt_model was supplied to this Session"
                )
            raise ValueError(
                f"unknown model {which!r}; have {sorted(self._models)}"
            )
        if which == self._active_model:
            return

        vis_state = copy.deepcopy(self.viz.vis_state)
        old_renderer = self._renderer
        old_model, old_data = self.model, self.data

        model = self._models[which]
        try:
            self.viz.rebind_model(model)
            new_model, new_data = self.viz.model, self.viz.data
            new_backend = CpuBackend(new_model, new_data)
            mujoco.mj_forward(new_model, new_data)
            good = (new_data.qpos.copy(), new_data.qvel.copy(), float(new_data.time))
            new_renderer = self.viz.make_renderer(height=self.height, width=self.width)
        except Exception:
            # rebind_model mutates self.viz.model/data (and their model-derived caches) in
            # place and may already have partially committed before raising -- e.g. it sets
            # self.viz.model before recomputing the caches that read it. Put the Visualizer
            # back on the OLD model: cheap and safe, because the colour-baking it repeats is
            # idempotent (geom_matid is already -1 from the first bake, so a second pass is a
            # no-op), and then restore its ACTUAL previous MjData -- rebind_model would
            # otherwise hand back a freshly zeroed one, silently discarding whatever
            # pose/velocity the old model was really holding.
            self.viz.rebind_model(old_model)
            self.viz.data = old_data
            raise

        # Nothing below here can fail, so this is where the swap actually becomes real.
        self.viz.vis_state = _carry_vis_state_across_swap(vis_state, new_model)
        self.model = new_model
        self.data = new_data
        self.backend = new_backend
        self._warn_baseline[:] = 0
        self._good = good
        self._renderer = new_renderer
        old_renderer.close()
        self._active_model = which
        # `_ctrl_map` is built by matching actuator NAMES against `self.model`, which just
        # changed -- a stale map would go on pointing at the OLD model's actuator ids, silently
        # writing a replay ctrl vector to the wrong slots on the new one (wrong at best, an
        # index error if the new model has fewer actuators). Structurally the same trap
        # `_carry_vis_state_across_swap`'s own docstring calls out for `forces`, and the one
        # `loop.py` guards against for its joint map on this same swap.
        self._ctrl_map = self._build_ctrl_map()
        self._rebuild_tendon_state()

    def load_settings(self, name: str) -> None:
        """Load a bundled OR user settings preset by name.

        Names only, never paths -- the same whitelist ``protocol.parse_command`` enforces, kept
        here too so the invariant does not depend on which entry point reached this method.
        ``Visualizer.load_settings`` deliberately still accepts paths for its own (local,
        non-networked) callers, which is why this guard lives in the serve layer.

        On a name collision between a bundled and a user preset, the user's own save wins:
        it is the more specific of the two, and a user who just saved a preset under a name
        that happens to match a bundled one clearly means to get their own version back, not
        silently keep loading the bundled default underneath it. ``list_available_settings``
        itself takes no side on this -- it lists both, distinguished by origin -- so the
        choice is made here, once, rather than left to whichever caller resolves the name.
        """
        available = list_available_settings(self.user_settings_dir)
        matches = [d for d in available if d["name"] == name]
        if not matches:
            raise ValueError(
                "unknown settings preset {0!r}; available: {1}".format(
                    name, ", ".join(sorted({d["name"] for d in available}))
                )
            )
        if any(d["origin"] == "user" for d in matches):
            self.viz.load_settings(str(self.user_settings_dir / f"{name}.json"))
        else:
            self.viz.load_settings(name)

    def save_settings_as(self, name: str) -> Path:
        """Save the current render settings as a NAMED preset in this session's
        ``user_settings_dir`` -- never in the package's own bundled ``settings/`` directory
        (that copy is read-only in some installs, is lost on reinstall, and is a tracked
        submodule directory a save must not dirty).

        *name* is checked against the same ``^[A-Za-z0-9_-]{1,64}$`` whitelist
        ``protocol.parse_command`` enforces on the wire, so this method is safe to call
        directly (from a test, a script, a future non-websocket caller) without depending on
        that guard having already run.

        Writes via a temp file in the same directory followed by an atomic replace, so a
        failure partway through (an unwritable directory, a full disk) raises and leaves NO
        partial preset file behind -- a half-written JSON silently accepted as a preset on a
        later load would be worse than the save just failing loudly.
        """
        if self.user_settings_dir is None:
            raise ValueError(
                "save_settings_as requires this Session to have been built with a "
                "user_settings_dir"
            )
        if not isinstance(name, str) or not PRESET_NAME_RE.match(name):
            raise ValueError(
                f"preset name must match {PRESET_NAME_RE.pattern!r}; got {name!r}"
            )
        self.user_settings_dir.mkdir(parents=True, exist_ok=True)
        dest = self.user_settings_dir / f"{name}.json"
        tmp = self.user_settings_dir / f".{name}.json.tmp"
        try:
            self.viz.save_settings(str(tmp))
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return dest

    # -- description -----------------------------------------------------------

    def scene_message(self) -> Dict:
        """Description the client builds its whole UI from -- a SNAPSHOT, not a live view.

        ``settings`` is a deep copy of ``vis_state`` rather than the dict itself. The loop
        thread mutates ``vis_state`` continuously (``apply_render``, ``load_settings``,
        ``set_camera``) while a Flask request thread may be part-way through serialising this
        message; handing out the live dict is how ``json.dumps``/``jsonify`` ends up raising
        "dictionary changed size during iteration" -- swallowed as a silently dropped
        WebSocket in ``_ws_loop``, or a 500 on ``/api/scene``.

        ``controls`` is the immutable tree built once in ``__init__`` and never written to
        again, so it is shared by reference deliberately: on the fly model it is 272 actuator
        dicts, and this method now runs once per published frame.

        Safe to call after :meth:`close`, which drops the backend -- a late ``/api/scene``
        should answer, not raise.
        """
        backend = getattr(self, "backend", None)
        return {
            "t": "scene",
            "nq": int(self.model.nq),
            "nv": int(self.model.nv),
            "nu": int(self.model.nu),
            "timestep": float(self.model.opt.timestep),
            "controls": self._tree,
            # Ordered lockable joints -- rebuilt from the CURRENT model every call (cheap: one
            # pass over njnt) rather than cached in __init__, so a swap_model() is reflected
            # without a separate invalidation path. Without this a client only ever learns
            # which joints are locked (frame_meta.locks), never which are lockable, so a lock
            # panel could not be built.
            "joints": [
                {"name": name, "width": width}
                for name, (_adr, width) in build_joint_qpos_map(self.model).items()
            ],
            "cameras": self.viz.list_cameras(),
            "presets": self.viz.list_presets(),
            "settings": copy.deepcopy(self.viz.vis_state),
            # Flat names, not the {"name", "origin"} dicts list_available_settings()
            # actually returns: this list goes straight to viewer.js as a dropdown's option
            # set (see static/viewer.js), which has always expected plain strings, and a
            # collision is not this wire message's problem to solve -- see load_settings for
            # where that gets decided. Kept as-is (additive change only) so this existing
            # field's contract does not shift under viewer.js.
            "settings_available": sorted(
                {d["name"] for d in list_available_settings(self.user_settings_dir)}
            ),
            # The {"name", "origin"} shape list_available_settings() actually returns,
            # untransformed -- so a later task's preset dropdown can tag each entry bundled
            # vs. user (design spec §6) without reaching back into this module. Additive:
            # "settings_available" above is untouched for existing/older clients.
            "settings_catalog": list_available_settings(self.user_settings_dir),
            "has_controller": self._controller is not None,
            "ctrl_mode": self._mode,
            "width": self.width,
            "height": self.height,
            "backend": None if backend is None else backend.label,
            "backend_warning": None if backend is None else backend.warning,
        }

    def close(self) -> None:
        """Release the Renderer and backend explicitly. EGL teardown raises from ``__del__``
        if left to the garbage collector, so lifetime is always explicit.

        Guarded the same way for both: a backend holding a real device context (a future
        ``WarpBackend``) must not be closed twice, even though it is harmless for
        ``CpuBackend``.
        """
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
        if getattr(self, "backend", None) is not None:
            self.backend.close()
            self.backend = None
