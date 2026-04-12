#!/usr/bin/env python3
"""gui.py -- Standalone dearpygui GUI for the generic Visualizer."""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure repo root is importable when running directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Default settings directory: visualizer/settings/ (next to this file)
_SETTINGS_DIR = Path(__file__).resolve().parent / 'settings'

import dearpygui.dearpygui as dpg

from mujoco_visualizer.visualizer import Visualizer, _hex_to_rgb, _rgb_to_hex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_to_dpg(hex_str: str) -> list[int]:
    """Convert '#rrggbb' to [r, g, b, 255] ints for dpg color widgets."""
    rgb = _hex_to_rgb(hex_str)
    return [int(c * 255) for c in rgb] + [255]


def _dpg_to_hex(rgba: list) -> str:
    """Convert dpg color [r, g, b, a] (0-255 ints) to '#rrggbb'."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]), int(rgba[1]), int(rgba[2]))


def _dpg_to_rgb_float(rgba: list) -> list[float]:
    """Convert dpg color [r, g, b, a] (0-255) to [r, g, b] (0-1)."""
    return [rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0]


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------

class DesktopGUI:
    """Interactive desktop GUI for the Visualizer using dearpygui."""

    PREVIEW_W = 640
    PREVIEW_H = 480

    def __init__(self, viz: Visualizer, qposes: Optional[np.ndarray] = None):
        self.viz = viz
        self._qposes = qposes
        self._frame_idx = 0
        self._playing = False
        self._fps = 30
        self._last_tick = 0.0
        self._suppress_callbacks = False
        self._BODY_CATEGORIES = viz.anatomy.category_names
        self._ALL_CAMERAS = viz.list_cameras()

        # Default qpos from model's initial configuration (qpos0 / keyframe)
        self._default_qpos = np.array(viz.model.qpos0, dtype=np.float64)

        # Widget tag storage
        self._tags: dict = {}

    def _current_qpos(self) -> np.ndarray:
        if self._qposes is not None:
            return self._qposes[self._frame_idx]
        return self._default_qpos

    # ── Rendering ─────────────────────────────────────────────────────────

    def _do_render(self):
        """Render current frame and push to texture."""
        frame = self.viz.render_frame(
            self._current_qpos(),
            height=self.PREVIEW_H,
            width=self.PREVIEW_W,
        )
        # Convert uint8 (H,W,3) to float32 (H,W,4) RGBA for dpg
        h, w = frame.shape[:2]
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, :3] = frame.astype(np.float32) / 255.0
        dpg.set_value(self._tags['texture'], rgba.ravel().tolist())

    # ── Build UI ──────────────────────────────────────────────────────────

    def _build(self):
        dpg.create_context()
        dpg.create_viewport(title="Visualizer", width=1100, height=600)

        # Create texture for preview
        with dpg.texture_registry():
            default_data = [0.0] * (self.PREVIEW_W * self.PREVIEW_H * 4)
            self._tags['texture'] = dpg.add_dynamic_texture(
                self.PREVIEW_W, self.PREVIEW_H,
                default_value=default_data,
            )

        with dpg.window(tag="primary", label="Visualizer"):
            with dpg.group(horizontal=True):
                # Left panel: controls
                with dpg.child_window(width=380, height=-1, tag="controls_panel"):
                    with dpg.tab_bar():
                        self._build_tab_colors()
                        self._build_tab_vis()
                        self._build_tab_lighting()
                        self._build_tab_camera()
                        self._build_tab_floor()
                        self._build_tab_settings()

                # Right panel: preview + playback
                with dpg.group():
                    dpg.add_image(self._tags['texture'])
                    self._build_playback_bar()

        dpg.set_primary_window("primary", True)

    # ── Tab: Colors ───────────────────────────────────────────────────────

    def _build_tab_colors(self):
        with dpg.tab(label="Colors"):
            dpg.add_text("Body Category Colors")
            dpg.add_separator()

            self._tags['cat_colors'] = {}
            for cat in self._BODY_CATEGORIES:
                hex_val = self.viz.vis_state['colors'].get(cat, '#888888')
                tag = dpg.add_color_edit(
                    default_value=_hex_to_dpg(hex_val),
                    label=cat,
                    no_alpha=True,
                    callback=self._on_cat_color,
                    user_data=cat,
                    width=180,
                )
                self._tags['cat_colors'][cat] = tag

            dpg.add_separator()
            dpg.add_text("Global Alpha")
            self._tags['alpha'] = dpg.add_slider_float(
                default_value=self.viz.vis_state['alpha'],
                min_value=0.0, max_value=1.0,
                callback=self._on_alpha,
                width=200,
            )

            dpg.add_separator()
            dpg.add_button(label="Reset Colors", callback=self._on_reset_colors)

    def _on_cat_color(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        cat = user_data
        hex_val = _dpg_to_hex(app_data)
        self.viz.vis_state['colors'][cat] = hex_val
        self._do_render()

    def _on_alpha(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['alpha'] = app_data
        self._do_render()

    def _on_reset_colors(self):
        self.viz.vis_state['colors'] = {
            cat: self.viz._cat_default_hex.get(cat, '#888888')
            for cat in self._BODY_CATEGORIES
        }
        self.viz.vis_state['geom_colors'] = {}
        self.viz.vis_state['alpha'] = 1.0
        self._sync_widgets_from_state()
        self._do_render()

    # ── Tab: Visualization ────────────────────────────────────────────────

    def _build_tab_vis(self):
        with dpg.tab(label="Vis"):
            dpg.add_text("Visibility Flags")
            dpg.add_separator()

            self._tags['vis_flags'] = {}
            for flag_name, default_val in self.viz.vis_state['vis_flags'].items():
                tag = dpg.add_checkbox(
                    label=flag_name.replace('_', ' ').title(),
                    default_value=default_val,
                    callback=self._on_vis_flag,
                    user_data=flag_name,
                )
                self._tags['vis_flags'][flag_name] = tag

            dpg.add_separator()
            dpg.add_text("Geom Groups")
            self._tags['geom_groups'] = []
            with dpg.group(horizontal=True):
                for i in range(6):
                    tag = dpg.add_checkbox(
                        label=str(i),
                        default_value=self.viz.vis_state['geom_groups'][i],
                        callback=self._on_geom_group,
                        user_data=i,
                    )
                    self._tags['geom_groups'].append(tag)

            dpg.add_text("Site Groups")
            self._tags['site_groups'] = []
            with dpg.group(horizontal=True):
                for i in range(6):
                    tag = dpg.add_checkbox(
                        label=str(i),
                        default_value=self.viz.vis_state['site_groups'][i],
                        callback=self._on_site_group,
                        user_data=i,
                    )
                    self._tags['site_groups'].append(tag)

    def _on_vis_flag(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['vis_flags'][user_data] = app_data
        self._do_render()

    def _on_geom_group(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['geom_groups'][user_data] = app_data
        self._do_render()

    def _on_site_group(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['site_groups'][user_data] = app_data
        self._do_render()

    # ── Tab: Lighting ─────────────────────────────────────────────────────

    def _build_tab_lighting(self):
        with dpg.tab(label="Lighting"):
            lights = self.viz.vis_state['lighting']['lights']
            self._tags['lights'] = []

            for li, ld in enumerate(lights):
                with dpg.collapsing_header(label=f"Light {li}", default_open=(li == 0)):
                    tags = {}
                    tags['active'] = dpg.add_checkbox(
                        label="Active", default_value=ld['active'],
                        callback=self._on_light_prop,
                        user_data=(li, 'active'),
                    )
                    tags['ambient'] = dpg.add_color_edit(
                        label="Ambient",
                        default_value=[int(c * 127.5) for c in ld['ambient']] + [255],
                        no_alpha=True, width=180,
                        callback=self._on_light_color,
                        user_data=(li, 'ambient'),
                    )
                    tags['diffuse'] = dpg.add_color_edit(
                        label="Diffuse",
                        default_value=[int(min(c, 1.0) * 255) for c in ld['diffuse']] + [255],
                        no_alpha=True, width=180,
                        callback=self._on_light_color,
                        user_data=(li, 'diffuse'),
                    )
                    tags['specular'] = dpg.add_color_edit(
                        label="Specular",
                        default_value=[int(min(c, 1.0) * 255) for c in ld['specular']] + [255],
                        no_alpha=True, width=180,
                        callback=self._on_light_color,
                        user_data=(li, 'specular'),
                    )
                    tags['dir_az'] = dpg.add_slider_float(
                        label="Azimuth", default_value=ld['dir_az'],
                        min_value=0.0, max_value=360.0, width=180,
                        callback=self._on_light_prop,
                        user_data=(li, 'dir_az'),
                    )
                    tags['dir_el'] = dpg.add_slider_float(
                        label="Elevation", default_value=ld['dir_el'],
                        min_value=-90.0, max_value=90.0, width=180,
                        callback=self._on_light_prop,
                        user_data=(li, 'dir_el'),
                    )
                    self._tags['lights'].append(tags)

            # Headlight
            hl = self.viz.vis_state['lighting']['headlight']
            with dpg.collapsing_header(label="Headlight", default_open=False):
                self._tags['hl'] = {}
                self._tags['hl']['active'] = dpg.add_checkbox(
                    label="Active", default_value=hl['active'],
                    callback=self._on_headlight_prop,
                    user_data='active',
                )
                for ch in ('ambient', 'diffuse', 'specular'):
                    self._tags['hl'][ch] = dpg.add_color_edit(
                        label=ch.title(),
                        default_value=[int(min(c, 1.0) * 255) for c in hl[ch]] + [255],
                        no_alpha=True, width=180,
                        callback=self._on_headlight_color,
                        user_data=ch,
                    )

            dpg.add_separator()
            # Presets
            lt = self.viz.vis_state['lighting']
            self._tags['dual_lighting'] = dpg.add_checkbox(
                label="Dual Lighting", default_value=lt.get('use_dual_lighting', False),
                callback=self._on_lighting_preset, user_data='use_dual_lighting',
            )
            self._tags['scale_lights'] = dpg.add_checkbox(
                label="Scale Lights", default_value=lt.get('use_scale_lights', False),
                callback=self._on_lighting_preset, user_data='use_scale_lights',
            )
            self._tags['scale_factor'] = dpg.add_slider_float(
                label="Scale Factor", default_value=lt.get('scale_lights_factor', 1.25),
                min_value=0.1, max_value=5.0, width=180,
                callback=self._on_scale_factor,
            )

    def _on_light_prop(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        li, prop = user_data
        self.viz.vis_state['lighting']['lights'][li][prop] = app_data
        self._do_render()

    def _on_light_color(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        li, channel = user_data
        # dpg color_edit returns [r,g,b,a] in 0-255
        rgb_float = _dpg_to_rgb_float(app_data)
        # For ambient, scale allows up to 2.0
        if channel == 'ambient':
            rgb_float = [c * 2.0 for c in rgb_float]
        self.viz.vis_state['lighting']['lights'][li][channel] = rgb_float
        self._do_render()

    def _on_headlight_prop(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['lighting']['headlight'][user_data] = app_data
        self._do_render()

    def _on_headlight_color(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        rgb_float = _dpg_to_rgb_float(app_data)
        self.viz.vis_state['lighting']['headlight'][user_data] = rgb_float
        self._do_render()

    def _on_lighting_preset(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['lighting'][user_data] = app_data
        self._do_render()

    def _on_scale_factor(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['lighting']['scale_lights_factor'] = app_data
        self._do_render()

    # ── Tab: Camera ───────────────────────────────────────────────────────

    def _build_tab_camera(self):
        with dpg.tab(label="Camera"):
            cam = self.viz.vis_state['camera']

            dpg.add_text("Camera Mode")
            self._tags['cam_mode'] = dpg.add_radio_button(
                items=["Named", "Free"],
                default_value="Named" if cam['mode'] == 'named' else "Free",
                horizontal=True,
                callback=self._on_cam_mode,
            )

            # Named panel
            with dpg.group(tag="named_panel"):
                self._tags['named_cam'] = dpg.add_combo(
                    items=self._ALL_CAMERAS,
                    default_value=cam.get('named', 'track1'),
                    label="Camera",
                    callback=self._on_named_cam,
                    width=200,
                )

            # Free panel
            with dpg.group(tag="free_panel"):
                self._tags['free_type'] = dpg.add_combo(
                    items=['free', 'fixed', 'track', 'trackcom'],
                    default_value=cam.get('free_type', 'free'),
                    label="Type",
                    callback=self._on_free_type,
                    width=200,
                )
                self._tags['cam_az'] = dpg.add_slider_float(
                    label="Azimuth", default_value=cam['azimuth'],
                    min_value=0, max_value=360, width=200,
                    callback=self._on_cam_slider, user_data='azimuth',
                )
                self._tags['cam_el'] = dpg.add_slider_float(
                    label="Elevation", default_value=cam['elevation'],
                    min_value=-90, max_value=90, width=200,
                    callback=self._on_cam_slider, user_data='elevation',
                )
                self._tags['cam_dist'] = dpg.add_slider_float(
                    label="Distance", default_value=cam['distance'],
                    min_value=0.01, max_value=5.0, width=200,
                    callback=self._on_cam_slider, user_data='distance',
                )
                dpg.add_text("Lookat (x, y, z)")
                with dpg.group(horizontal=True):
                    for i, axis in enumerate(['x', 'y', 'z']):
                        self._tags[f'lookat_{axis}'] = dpg.add_input_float(
                            default_value=cam['lookat'][i],
                            width=80, step=0.01,
                            callback=self._on_lookat, user_data=i,
                        )
                self._tags['trackbody'] = dpg.add_input_text(
                    label="Track Body", default_value=cam.get('trackbody', ''),
                    callback=self._on_trackbody, width=200,
                )
                self._tags['fixedcamid'] = dpg.add_input_text(
                    label="Fixed Cam", default_value=cam.get('fixedcamid', ''),
                    callback=self._on_fixedcam, width=200,
                )

            # Show/hide based on current mode
            if cam['mode'] == 'named':
                dpg.hide_item("free_panel")
            else:
                dpg.hide_item("named_panel")

    def _on_cam_mode(self, sender, app_data):
        if self._suppress_callbacks:
            return
        if app_data == "Named":
            self.viz.vis_state['camera']['mode'] = 'named'
            dpg.show_item("named_panel")
            dpg.hide_item("free_panel")
        else:
            self.viz.vis_state['camera']['mode'] = 'free'
            dpg.hide_item("named_panel")
            dpg.show_item("free_panel")
        self._do_render()

    def _on_named_cam(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['camera']['named'] = app_data
        self._do_render()

    def _on_free_type(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['camera']['free_type'] = app_data
        self._do_render()

    def _on_cam_slider(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['camera'][user_data] = app_data
        self._do_render()

    def _on_lookat(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['camera']['lookat'][user_data] = app_data
        self._do_render()

    def _on_trackbody(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['camera']['trackbody'] = app_data
        self._do_render()

    def _on_fixedcam(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['camera']['fixedcamid'] = app_data
        self._do_render()

    # ── Tab: Floor & Skybox ───────────────────────────────────────────────

    def _build_tab_floor(self):
        with dpg.tab(label="Floor"):
            fld = self.viz.vis_state['floor']
            sky = self.viz.vis_state['skybox']

            dpg.add_text("Floor")
            dpg.add_separator()

            self._tags['floor_color'] = dpg.add_color_edit(
                label="Floor Color",
                default_value=_hex_to_dpg(fld['color']),
                no_alpha=True, width=180,
                callback=self._on_floor_color,
            )
            self._tags['floor_alpha'] = dpg.add_slider_float(
                label="Alpha", default_value=fld['alpha'],
                min_value=0, max_value=1, width=200,
                callback=self._on_floor_prop, user_data='alpha',
            )
            self._tags['floor_texrep_x'] = dpg.add_slider_float(
                label="Tex Repeat X", default_value=fld['texrepeat_x'],
                min_value=0, max_value=20, width=200,
                callback=self._on_floor_prop, user_data='texrepeat_x',
            )
            self._tags['floor_texrep_y'] = dpg.add_slider_float(
                label="Tex Repeat Y", default_value=fld['texrepeat_y'],
                min_value=0, max_value=20, width=200,
                callback=self._on_floor_prop, user_data='texrepeat_y',
            )
            self._tags['floor_refl'] = dpg.add_slider_float(
                label="Reflectance", default_value=fld['reflectance'],
                min_value=0, max_value=1, width=200,
                callback=self._on_floor_prop, user_data='reflectance',
            )
            self._tags['floor_shine'] = dpg.add_slider_float(
                label="Shininess", default_value=fld['shininess'],
                min_value=0, max_value=1, width=200,
                callback=self._on_floor_prop, user_data='shininess',
            )
            self._tags['floor_emis'] = dpg.add_slider_float(
                label="Emission", default_value=fld['emission'],
                min_value=0, max_value=1, width=200,
                callback=self._on_floor_prop, user_data='emission',
            )

            dpg.add_separator()
            dpg.add_text("Skybox")
            dpg.add_separator()

            self._tags['sky_show'] = dpg.add_checkbox(
                label="Show Skybox", default_value=sky['show'],
                callback=self._on_sky_show,
            )
            self._tags['sky_top'] = dpg.add_color_edit(
                label="Sky Top",
                default_value=_hex_to_dpg(sky['sky_top']),
                no_alpha=True, width=180,
                callback=self._on_sky_color, user_data='sky_top',
            )
            self._tags['sky_bot'] = dpg.add_color_edit(
                label="Sky Bottom",
                default_value=_hex_to_dpg(sky['sky_bot']),
                no_alpha=True, width=180,
                callback=self._on_sky_color, user_data='sky_bot',
            )

    def _on_floor_color(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['floor']['color'] = _dpg_to_hex(app_data)
        self._do_render()

    def _on_floor_prop(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['floor'][user_data] = app_data
        self._do_render()

    def _on_sky_show(self, sender, app_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['skybox']['show'] = app_data
        self._do_render()

    def _on_sky_color(self, sender, app_data, user_data):
        if self._suppress_callbacks:
            return
        self.viz.vis_state['skybox'][user_data] = _dpg_to_hex(app_data)
        self._do_render()

    # ── Tab: Settings ─────────────────────────────────────────────────────

    def _build_tab_settings(self):
        with dpg.tab(label="Settings"):
            dpg.add_text("Save / Load Settings")
            dpg.add_separator()

            # List existing .json files in settings dir
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            existing = sorted(p.name for p in _SETTINGS_DIR.glob('*.json'))
            self._tags['settings_path'] = dpg.add_input_text(
                label="File", default_value=existing[0] if existing else "settings.json", width=200,
            )
            if existing:
                self._tags['settings_combo'] = dpg.add_combo(
                    items=existing,
                    default_value=existing[0],
                    label="Saved",
                    width=200,
                    callback=self._on_settings_combo,
                )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save", callback=self._on_save_settings)
                dpg.add_button(label="Load", callback=self._on_load_settings)
                dpg.add_button(label="Browse...", callback=self._on_browse_settings)

            self._tags['settings_status'] = dpg.add_text("", color=[128, 255, 128])

            dpg.add_separator()
            dpg.add_text("Camera Presets")
            dpg.add_separator()

            presets = list(self.viz.vis_state.get('camera_presets', {}).keys())
            self._tags['preset_name'] = dpg.add_input_text(
                label="Name", default_value="", width=200,
            )
            self._tags['preset_combo'] = dpg.add_combo(
                items=presets,
                default_value=presets[0] if presets else "",
                label="Preset",
                width=200,
            )
            with dpg.group(horizontal=True):
                dpg.add_button(label="Save Preset", callback=self._on_save_preset)
                dpg.add_button(label="Load Preset", callback=self._on_load_preset)
                dpg.add_button(label="Delete Preset", callback=self._on_delete_preset)

            self._tags['preset_status'] = dpg.add_text("", color=[128, 255, 128])

            # Rollout loading
            if self._qposes is None:
                dpg.add_separator()
                dpg.add_text("Load Rollout")
                dpg.add_separator()
                self._tags['h5_path'] = dpg.add_input_text(
                    label="HDF5 File", default_value="", width=200,
                )
                self._tags['rollout_idx'] = dpg.add_input_int(
                    label="Rollout Index", default_value=0, width=100,
                )
                dpg.add_button(label="Load Rollout", callback=self._on_load_rollout)

            # High-quality frame export
            dpg.add_separator()
            dpg.add_text("Export Frame")
            dpg.add_separator()
            self._tags['export_path'] = dpg.add_input_text(
                label="Output", default_value="frame.png", width=200,
            )
            with dpg.group(horizontal=True):
                self._tags['export_h'] = dpg.add_input_int(
                    label="H", default_value=2160, width=80,
                )
                self._tags['export_w'] = dpg.add_input_int(
                    label="W", default_value=3840, width=80,
                )
            dpg.add_button(label="Export HQ Frame", callback=self._on_export_frame)
            self._tags['export_status'] = dpg.add_text("", color=[128, 255, 128])

    def _resolve_settings_path(self, name: str) -> Path:
        """Resolve a settings filename to a full path in _SETTINGS_DIR."""
        p = Path(name)
        if p.is_absolute():
            return p
        # Ensure .json extension
        if p.suffix != '.json':
            p = p.with_suffix('.json')
        return _SETTINGS_DIR / p

    def _refresh_settings_combo(self):
        """Refresh the saved-settings dropdown after save/delete."""
        if 'settings_combo' not in self._tags:
            return
        existing = sorted(p.name for p in _SETTINGS_DIR.glob('*.json'))
        dpg.configure_item(self._tags['settings_combo'], items=existing)

    def _on_settings_combo(self, sender, app_data):
        """When a saved file is selected from the dropdown, put it in the text input."""
        dpg.set_value(self._tags['settings_path'], app_data)

    def _on_save_settings(self):
        name = dpg.get_value(self._tags['settings_path'])
        path = self._resolve_settings_path(name)
        try:
            _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
            self.viz.save_settings(str(path))
            self._refresh_settings_combo()
            dpg.set_value(self._tags['settings_status'], f"Saved: {path.name}")
        except Exception as e:
            dpg.set_value(self._tags['settings_status'], f"Error: {e}")

    def _on_load_settings(self):
        name = dpg.get_value(self._tags['settings_path'])
        path = self._resolve_settings_path(name)
        try:
            self.viz.load_settings(str(path))
            self._sync_widgets_from_state()
            self._do_render()
            dpg.set_value(self._tags['settings_status'], f"Loaded: {path.name}")
        except Exception as e:
            dpg.set_value(self._tags['settings_status'], f"Error: {e}")

    def _on_browse_settings(self):
        with dpg.file_dialog(
            label="Select Settings File",
            callback=self._on_browse_callback,
            width=500, height=400,
            default_path=str(_SETTINGS_DIR),
        ):
            dpg.add_file_extension(".json", color=[255, 255, 0])
            dpg.add_file_extension(".*")

    def _on_browse_callback(self, sender, app_data):
        if app_data and 'file_path_name' in app_data:
            picked = Path(app_data['file_path_name'])
            # Show just the filename if it's inside _SETTINGS_DIR
            try:
                rel = picked.relative_to(_SETTINGS_DIR)
                dpg.set_value(self._tags['settings_path'], str(rel))
            except ValueError:
                dpg.set_value(self._tags['settings_path'], str(picked))

    def _on_save_preset(self):
        name = dpg.get_value(self._tags['preset_name'])
        if not name:
            dpg.set_value(self._tags['preset_status'], "Enter a preset name")
            return
        self.viz.vis_state.setdefault('camera_presets', {})[name] = copy.deepcopy(
            self.viz.vis_state['camera']
        )
        self._refresh_preset_combo()
        dpg.set_value(self._tags['preset_status'], f"Saved preset '{name}'")

    def _on_load_preset(self):
        name = dpg.get_value(self._tags['preset_combo'])
        presets = self.viz.vis_state.get('camera_presets', {})
        if name not in presets:
            dpg.set_value(self._tags['preset_status'], f"Preset '{name}' not found")
            return
        self.viz.vis_state['camera'] = copy.deepcopy(presets[name])
        self._sync_camera_widgets()
        self._do_render()
        dpg.set_value(self._tags['preset_status'], f"Loaded preset '{name}'")

    def _on_delete_preset(self):
        name = dpg.get_value(self._tags['preset_combo'])
        presets = self.viz.vis_state.get('camera_presets', {})
        if name in presets:
            del presets[name]
            self._refresh_preset_combo()
            dpg.set_value(self._tags['preset_status'], f"Deleted preset '{name}'")

    def _refresh_preset_combo(self):
        presets = list(self.viz.vis_state.get('camera_presets', {}).keys())
        dpg.configure_item(self._tags['preset_combo'], items=presets)
        if presets:
            dpg.set_value(self._tags['preset_combo'], presets[0])

    def _on_load_rollout(self):
        path = dpg.get_value(self._tags['h5_path'])
        idx = dpg.get_value(self._tags['rollout_idx'])
        try:
            # Generic loader: .npy/.npz only. Use a domain-specific subclass
            # to support HDF5 / project-specific rollout formats.
            arr = np.load(path)
            self._qposes = arr['qpos'] if hasattr(arr, 'files') else arr
            self._frame_idx = 0
            # Update frame slider max
            if 'frame_slider' in self._tags:
                dpg.configure_item(
                    self._tags['frame_slider'],
                    max_value=len(self._qposes) - 1,
                )
                dpg.set_value(self._tags['frame_slider'], 0)
            if 'frame_label' in self._tags:
                dpg.set_value(
                    self._tags['frame_label'],
                    f"Frame 0 / {len(self._qposes)}",
                )
            self._do_render()
        except Exception as e:
            dpg.set_value(self._tags['settings_status'], f"Error: {e}")

    def _on_export_frame(self):
        path = dpg.get_value(self._tags['export_path'])
        h = dpg.get_value(self._tags['export_h'])
        w = dpg.get_value(self._tags['export_w'])
        try:
            self.viz.save_frame(self._current_qpos(), path, height=h, width=w)
            dpg.set_value(self._tags['export_status'], f"Exported {path} ({h}x{w})")
        except Exception as e:
            dpg.set_value(self._tags['export_status'], f"Error: {e}")

    # ── Playback Bar ──────────────────────────────────────────────────────

    def _build_playback_bar(self):
        max_frames = len(self._qposes) - 1 if self._qposes is not None else 0

        with dpg.group():
            with dpg.group(horizontal=True):
                dpg.add_button(label="<<", callback=self._on_frame_prev, width=30)
                self._tags['play_btn'] = dpg.add_button(
                    label="Play", callback=self._on_play_pause, width=50,
                )
                dpg.add_button(label=">>", callback=self._on_frame_next, width=30)
                self._tags['frame_label'] = dpg.add_text(
                    f"Frame 0 / {max_frames + 1}" if self._qposes is not None else "No rollout"
                )

            self._tags['frame_slider'] = dpg.add_slider_int(
                default_value=0, min_value=0, max_value=max(max_frames, 1),
                callback=self._on_frame_slider, width=-1,
            )
            with dpg.group(horizontal=True):
                dpg.add_text("FPS:")
                self._tags['fps_input'] = dpg.add_input_int(
                    default_value=self._fps, width=60,
                    callback=self._on_fps_change,
                )

    def _on_frame_slider(self, sender, app_data):
        if self._suppress_callbacks or self._qposes is None:
            return
        self._frame_idx = app_data
        dpg.set_value(
            self._tags['frame_label'],
            f"Frame {self._frame_idx} / {len(self._qposes)}",
        )
        self._do_render()

    def _on_play_pause(self):
        if self._qposes is None:
            return
        self._playing = not self._playing
        dpg.configure_item(
            self._tags['play_btn'],
            label="Pause" if self._playing else "Play",
        )
        if self._playing:
            self._last_tick = time.time()

    def _on_frame_prev(self):
        if self._qposes is None:
            return
        self._frame_idx = max(0, self._frame_idx - 1)
        dpg.set_value(self._tags['frame_slider'], self._frame_idx)
        dpg.set_value(
            self._tags['frame_label'],
            f"Frame {self._frame_idx} / {len(self._qposes)}",
        )
        self._do_render()

    def _on_frame_next(self):
        if self._qposes is None:
            return
        self._frame_idx = min(len(self._qposes) - 1, self._frame_idx + 1)
        dpg.set_value(self._tags['frame_slider'], self._frame_idx)
        dpg.set_value(
            self._tags['frame_label'],
            f"Frame {self._frame_idx} / {len(self._qposes)}",
        )
        self._do_render()

    def _on_fps_change(self, sender, app_data):
        self._fps = max(1, app_data)

    # ── Playback tick (called each frame) ─────────────────────────────────

    def _playback_tick(self):
        if not self._playing or self._qposes is None:
            return
        now = time.time()
        if now - self._last_tick >= 1.0 / self._fps:
            self._frame_idx = (self._frame_idx + 1) % len(self._qposes)
            dpg.set_value(self._tags['frame_slider'], self._frame_idx)
            dpg.set_value(
                self._tags['frame_label'],
                f"Frame {self._frame_idx} / {len(self._qposes)}",
            )
            self._do_render()
            self._last_tick = now

    # ── Sync widgets from vis_state ───────────────────────────────────────

    def _sync_widgets_from_state(self):
        """Push current vis_state values into all widgets."""
        self._suppress_callbacks = True
        try:
            # Colors
            for cat in self._BODY_CATEGORIES:
                hex_val = self.viz.vis_state['colors'].get(cat, '#888888')
                dpg.set_value(self._tags['cat_colors'][cat], _hex_to_dpg(hex_val))
            dpg.set_value(self._tags['alpha'], self.viz.vis_state['alpha'])

            # Vis flags
            for flag, tag in self._tags['vis_flags'].items():
                dpg.set_value(tag, self.viz.vis_state['vis_flags'].get(flag, False))

            # Geom/site groups
            for i, tag in enumerate(self._tags['geom_groups']):
                dpg.set_value(tag, self.viz.vis_state['geom_groups'][i])
            for i, tag in enumerate(self._tags['site_groups']):
                dpg.set_value(tag, self.viz.vis_state['site_groups'][i])

            # Lighting
            self._sync_lighting_widgets()

            # Camera
            self._sync_camera_widgets()

            # Floor
            fld = self.viz.vis_state['floor']
            dpg.set_value(self._tags['floor_color'], _hex_to_dpg(fld['color']))
            dpg.set_value(self._tags['floor_alpha'], fld['alpha'])
            dpg.set_value(self._tags['floor_texrep_x'], fld['texrepeat_x'])
            dpg.set_value(self._tags['floor_texrep_y'], fld['texrepeat_y'])
            dpg.set_value(self._tags['floor_refl'], fld['reflectance'])
            dpg.set_value(self._tags['floor_shine'], fld['shininess'])
            dpg.set_value(self._tags['floor_emis'], fld['emission'])

            # Skybox
            sky = self.viz.vis_state['skybox']
            dpg.set_value(self._tags['sky_show'], sky['show'])
            dpg.set_value(self._tags['sky_top'], _hex_to_dpg(sky['sky_top']))
            dpg.set_value(self._tags['sky_bot'], _hex_to_dpg(sky['sky_bot']))

            # Presets
            self._refresh_preset_combo()

        finally:
            self._suppress_callbacks = False

    def _sync_lighting_widgets(self):
        lights = self.viz.vis_state['lighting']['lights']
        for li, ld in enumerate(lights):
            if li >= len(self._tags['lights']):
                break
            tags = self._tags['lights'][li]
            dpg.set_value(tags['active'], ld['active'])
            dpg.set_value(tags['ambient'],
                          [int(min(c / 2.0, 1.0) * 255) for c in ld['ambient']] + [255])
            dpg.set_value(tags['diffuse'],
                          [int(min(c, 1.0) * 255) for c in ld['diffuse']] + [255])
            dpg.set_value(tags['specular'],
                          [int(min(c, 1.0) * 255) for c in ld['specular']] + [255])
            dpg.set_value(tags['dir_az'], ld['dir_az'])
            dpg.set_value(tags['dir_el'], ld['dir_el'])

        hl = self.viz.vis_state['lighting']['headlight']
        dpg.set_value(self._tags['hl']['active'], hl['active'])
        for ch in ('ambient', 'diffuse', 'specular'):
            dpg.set_value(self._tags['hl'][ch],
                          [int(min(c, 1.0) * 255) for c in hl[ch]] + [255])

        lt = self.viz.vis_state['lighting']
        dpg.set_value(self._tags['dual_lighting'], lt.get('use_dual_lighting', False))
        dpg.set_value(self._tags['scale_lights'], lt.get('use_scale_lights', False))
        dpg.set_value(self._tags['scale_factor'], lt.get('scale_lights_factor', 1.25))

    def _sync_camera_widgets(self):
        cam = self.viz.vis_state['camera']
        mode = cam.get('mode', 'named')
        dpg.set_value(self._tags['cam_mode'], "Named" if mode == 'named' else "Free")
        dpg.set_value(self._tags['named_cam'], cam.get('named', 'track1'))
        dpg.set_value(self._tags['free_type'], cam.get('free_type', 'free'))
        dpg.set_value(self._tags['cam_az'], cam.get('azimuth', 180.0))
        dpg.set_value(self._tags['cam_el'], cam.get('elevation', -30.0))
        dpg.set_value(self._tags['cam_dist'], cam.get('distance', 0.3))
        for i, axis in enumerate(['x', 'y', 'z']):
            dpg.set_value(self._tags[f'lookat_{axis}'], cam['lookat'][i])
        dpg.set_value(self._tags['trackbody'], cam.get('trackbody', ''))
        dpg.set_value(self._tags['fixedcamid'], cam.get('fixedcamid', ''))

        if mode == 'named':
            dpg.show_item("named_panel")
            dpg.hide_item("free_panel")
        else:
            dpg.hide_item("named_panel")
            dpg.show_item("free_panel")

    # ── Run ───────────────────────────────────────────────────────────────

    def run(self):
        """Launch the GUI."""
        self._build()
        dpg.setup_dearpygui()
        dpg.show_viewport()

        # Initial render
        self._do_render()

        # Main loop with playback tick
        while dpg.is_dearpygui_running():
            self._playback_tick()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='mujoco_visualizer interactive GUI')
    p.add_argument('--xml', required=True, help='Path to a MuJoCo XML file.')
    p.add_argument('--anatomy', help='Path to an anatomy YAML/JSON config.')
    p.add_argument('--settings', help='Settings JSON file or preset name.')
    p.add_argument('--floor-xml', help='Optional floor XML to attach into.')
    p.add_argument('--qpos', help='Optional .npy/.npz of qpos trajectory.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    viz = Visualizer(
        xml_path=args.xml,
        anatomy=args.anatomy,
        settings_json=args.settings,
        floor_xml=args.floor_xml,
    )
    qposes = None
    if args.qpos:
        arr = np.load(args.qpos)
        qposes = arr['qpos'] if hasattr(arr, 'files') else arr
    DesktopGUI(viz, qposes=qposes).run()


if __name__ == '__main__':
    main()
