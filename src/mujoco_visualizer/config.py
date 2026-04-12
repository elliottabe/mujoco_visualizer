"""config.py — Anatomy / camera / pose configuration for the generic Visualizer.

A small dataclass-based schema describing how to bucket the geoms of an
arbitrary MuJoCo model into named categories (for per-category recoloring),
which named cameras to expose in GUIs, and how to group joints for an optional
"Pose" GUI tab. Loaded from YAML or JSON.

Schema (YAML)::

    cameras: [side, front, top]            # optional; falls back to model cameras
    root_body: thorax                      # optional; used for floor attachment
    categories:                            # ordered list
      - name: torso
        match:
          body_substring: [torso, abdomen]
      - name: left_arm
        match:
          body_substring: [arm, left]
          all: true                        # require ALL substrings to match
    pose_groups:                           # optional; drives a Pose tab
      - label: Left arm
        joint_substrings: [shoulder, elbow, wrist]
        side_filters: [left]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class CategoryRule:
    """A single category rule. Geoms whose body name (lowercased) contains the
    substrings in ``body_substring`` are assigned to this category. If ``all``
    is True, every substring must match; otherwise any single match is enough.

    ``geom_substring`` (optional) further restricts by geom name.
    """
    name: str
    body_substring: List[str] = field(default_factory=list)
    geom_substring: List[str] = field(default_factory=list)
    all: bool = False

    def matches(self, body_name: str, geom_name: str) -> bool:
        bn = (body_name or "").lower()
        gn = (geom_name or "").lower()
        if self.body_substring:
            hits = [s.lower() in bn for s in self.body_substring]
            if (all(hits) if self.all else any(hits)) is False:
                return False
        if self.geom_substring:
            hits = [s.lower() in gn for s in self.geom_substring]
            if (all(hits) if self.all else any(hits)) is False:
                return False
        return bool(self.body_substring or self.geom_substring)


@dataclass
class PoseGroup:
    """One group of joints exposed as a sub-tab in a Pose GUI."""
    label: str
    joint_substrings: List[str] = field(default_factory=list)
    side_filters: List[str] = field(default_factory=list)  # e.g. ["left", "right"]


@dataclass
class AnatomyConfig:
    cameras: List[str] = field(default_factory=list)
    root_body: Optional[str] = None
    categories: List[CategoryRule] = field(default_factory=list)
    pose_groups: List[PoseGroup] = field(default_factory=list)

    @property
    def category_names(self) -> List[str]:
        return [c.name for c in self.categories]

    @classmethod
    def from_dict(cls, d: dict) -> "AnatomyConfig":
        cats = []
        for c in d.get("categories", []):
            m = c.get("match", {}) or {}
            cats.append(CategoryRule(
                name=c["name"],
                body_substring=list(m.get("body_substring", []) or []),
                geom_substring=list(m.get("geom_substring", []) or []),
                all=bool(m.get("all", False)),
            ))
        groups = [
            PoseGroup(
                label=g["label"],
                joint_substrings=list(g.get("joint_substrings", []) or []),
                side_filters=list(g.get("side_filters", []) or []),
            )
            for g in d.get("pose_groups", []) or []
        ]
        return cls(
            cameras=list(d.get("cameras", []) or []),
            root_body=d.get("root_body"),
            categories=cats,
            pose_groups=groups,
        )


def load_config(path_or_dict: Union[str, Path, dict, None]) -> AnatomyConfig:
    """Load an AnatomyConfig from a YAML/JSON file path, dict, or ``None``.

    ``None`` returns an empty config (Visualizer will then auto-derive one
    category per top-level body and expose all cameras declared in the model).
    """
    if path_or_dict is None:
        return AnatomyConfig()
    if isinstance(path_or_dict, dict):
        return AnatomyConfig.from_dict(path_or_dict)
    p = Path(path_or_dict)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml
        d = yaml.safe_load(text) or {}
    else:
        d = json.loads(text)
    return AnatomyConfig.from_dict(d)
