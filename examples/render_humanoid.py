"""Render a single frame of any MuJoCo model with mujoco_visualizer."""
import argparse
import numpy as np
from PIL import Image
from mujoco_visualizer import Visualizer, load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xml", help="Path to a MuJoCo XML file (e.g. humanoid.xml).")
    p.add_argument("--anatomy", default=None, help="Optional anatomy YAML.")
    p.add_argument("--out", default="frame.png")
    args = p.parse_args()

    viz = Visualizer(args.xml, anatomy=load_config(args.anatomy))
    frame = viz.render_frame(viz.model.qpos0, height=480, width=640)
    Image.fromarray(frame).save(args.out)
    print(f"wrote {args.out}  ({frame.shape})")


if __name__ == "__main__":
    main()
