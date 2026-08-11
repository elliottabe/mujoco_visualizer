"""mujoco_visualizer — generic offscreen MuJoCo visualizer.

Public API:
    Visualizer              — main rendering class
    AnatomyConfig           — dataclass describing categories / cameras / pose groups
    load_config             — load a YAML/JSON anatomy config
    filter_model_to_config_joints — structural model amputation utility
    render_settings helpers — load_settings, apply_settings, build_scene_option,
                              build_camera, apply_scene_flags, get_scene_modifiers,
                              setup_render, list_available_settings
"""
from mujoco_visualizer.visualizer import Visualizer, allocate_segment_frames
from mujoco_visualizer.config import AnatomyConfig, CategoryRule, PoseGroup, load_config
from mujoco_visualizer.categories import build_geom_categories
from mujoco_visualizer.model_utils import filter_model_to_config_joints
from mujoco_visualizer.render_settings import (
    apply_scene_flags,
    apply_settings,
    build_camera,
    build_scene_option,
    get_scene_modifiers,
    list_available_settings,
    load_settings,
    make_pan_cameras,
    setup_render,
)

__all__ = [
    "Visualizer",
    "AnatomyConfig",
    "CategoryRule",
    "PoseGroup",
    "load_config",
    "build_geom_categories",
    "filter_model_to_config_joints",
    "load_settings",
    "apply_settings",
    "build_scene_option",
    "apply_scene_flags",
    "get_scene_modifiers",
    "build_camera",
    "make_pan_cameras",
    "allocate_segment_frames",
    "setup_render",
    "list_available_settings",
]


def __getattr__(name):
    if name == "WidgetGUI":
        from mujoco_visualizer.widget_gui import WidgetGUI
        return WidgetGUI
    if name == "DesktopGUI":
        from mujoco_visualizer.gui import DesktopGUI
        return DesktopGUI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
