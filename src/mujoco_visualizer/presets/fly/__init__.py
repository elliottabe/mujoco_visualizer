"""Fruitfly preset: anatomy config + flight setup helpers.

These live here so users can do::

    from mujoco_visualizer.presets.fly import FLY_ANATOMY_PATH, apply_flight_setup

without depending on fly_neuromech. The ``fly_neuromech/visualizer``
wrapper builds its ``FlyVisualizer`` on top of these.
"""
from pathlib import Path

from .fly_model_utils import apply_flight_setup  # noqa: F401

FLY_ANATOMY_PATH = Path(__file__).parent / 'fly_anatomy.yaml'
FLY_V2_ANATOMY_PATH = Path(__file__).parent / 'fly_v2_anatomy.yaml'

__all__ = ['FLY_ANATOMY_PATH', 'FLY_V2_ANATOMY_PATH', 'apply_flight_setup']
