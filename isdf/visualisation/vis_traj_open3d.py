"""Visualize a trajectory (cam-to-world 4x4 matrices) with Open3D.

Supports iSDF-style trajectory files where each line is a flattened 4x4 (16 floats).

Example:
  python isdf/visualisation/vis_traj_open3d.py \
    --traj /home/hjx/dataset/iSDF/seqs/apt_667_nav/traj_cam.txt \
    --frame_stride 10 --frame_size 0.15

Tips:
- If your trajectory is large, increase --frame_stride.
- You can show only the polyline by setting --frame_stride 0.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "open3d is required. Install it in your environment (see environment.yml)."
    ) from exc


def load_traj_16f(path: str) -> np.ndarray:
    data = np.loadtxt(path)
    if data.ndim == 1:
        if data.size != 16:
            raise ValueError(f"Expected 16 floats, got {data.size} from {path}")
        data = data.reshape(1, 16)
    if data.shape[1] != 16:
        raise ValueError(f"Expected (N,16) floats, got {data.shape} from {path}")
    return data.reshape(-1, 4, 4).astype(np.float64)


def make_polyline(points: np.ndarray, color=(1.0, 0.2, 0.2)) -> o3d.geometry.LineSet:
    assert points.ndim == 2 and points.shape[1] == 3
    n = points.shape[0]
    lines = [[i, i + 1] for i in range(n - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    colors = np.tile(np.array(color, dtype=float)[None, :], (len(lines), 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def main() -> int:
    p = argparse.ArgumentParser(description="Open3D visualize 4x4 trajectory")
    p.add_argument("--traj", required=True, help="Path to traj file (N lines * 16 floats)")
    p.add_argument("--frame_stride", type=int, default=10, help="Draw one camera frame every N poses; 0 disables frames")
    p.add_argument("--frame_size", type=float, default=0.15, help="Camera frame axis size")
    p.add_argument("--show_world", action="store_true", help="Show world coordinate frame at origin")
    p.add_argument("--world_size", type=float, default=0.5, help="World frame axis size (when --show_world)")
    p.add_argument("--line_color", nargs=3, type=float, default=[1.0, 0.2, 0.2], help="Trajectory polyline RGB")

    args = p.parse_args()

    T_WC = load_traj_16f(args.traj)
    positions = T_WC[:, :3, 3]

    geoms: List[o3d.geometry.Geometry] = []

    geoms.append(make_polyline(positions, color=tuple(args.line_color)))

    if args.show_world:
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.world_size))

    stride = int(args.frame_stride)
    if stride > 0:
        for i in range(0, T_WC.shape[0], stride):
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(args.frame_size))
            frame.transform(T_WC[i])
            geoms.append(frame)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Trajectory",
        width=1280,
        height=720,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
