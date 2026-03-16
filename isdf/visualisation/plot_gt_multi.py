import argparse
import numpy as np
import trimesh
from isdf.geometry import transform as isdf_transform
from isdf.visualisation import draw3D
import os


def load_poses(path):
    # 支持 .npy (N,4,4) 或 文本文件 (N,12) 或 (N,16) 或 (N,3)
    if path.endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.loadtxt(path)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        return arr
    if arr.ndim == 2:
        if arr.shape[1] == 12:
            return arr.reshape(-1, 3, 4).astype(np.float64)
        if arr.shape[1] == 16:
            return arr.reshape(-1, 4, 4).astype(np.float64)
        if arr.shape[1] == 3:
            # 仅位置列表，构造齐次变换（单位旋转）
            T = np.tile(np.eye(4), (arr.shape[0], 1, 1)).astype(np.float64)
            T[:, :3, 3] = arr
            return T
    raise ValueError("无法解析位姿文件: 支持 .npy (N,4,4) 或 文本 (N,12/16/3)")


def to_T_WC(Ts):
    # 确保返回 (N,4,4) numpy
    if Ts.shape[1:] == (3, 4):
        T = np.tile(np.eye(4), (Ts.shape[0], 1, 1)).astype(np.float64)
        T[:, :3, :4] = Ts
        return T
    return Ts


def sample_with_interval(arr, interval=100):
    """以指定间隔采样数组"""
    return arr[::interval]


def split_into_segments(arr, num_segments=4):
    """将数组均匀分为指定数量的段"""
    n = len(arr)
    segment_size = n // num_segments
    segments = []
    for i in range(num_segments):
        start_idx = i * segment_size
        if i == num_segments - 1:
            # 最后一段包含所有剩余元素
            end_idx = n
        else:
            end_idx = (i + 1) * segment_size
        segments.append(arr[start_idx:end_idx])
    return segments


def main(mesh_path, traj_path, out_path, resolution=(1280, 720), topdown_distance=1.5, 
         num_segments=4, sample_interval=100):
    mesh = trimesh.load(mesh_path, process=False)
    Ts = load_poses(traj_path)
    Ts = to_T_WC(Ts)
    # 如果 poses 是 (N,3,4)，扩展为 (N,4,4)
    if Ts.shape[1:] == (3, 4):
        Tfull = np.tile(np.eye(4), (Ts.shape[0], 1, 1))
        Tfull[:, :3, :4] = Ts
        Ts = Tfull

    # 创建场景并添加几何
    scene = trimesh.Scene(mesh)
    scene.set_camera()
    scene.camera.fx = 577.870605
    scene.camera.fy = 577.870605
    scene.camera.resolution = resolution

    # 定义四种颜色 (RGBA)
    colors = [
        (1.0, 0.0, 0.0, 0.8),  # 红色
        (0.0, 1.0, 0.0, 0.8),  # 绿色
        (0.0, 0.0, 1.0, 0.8),  # 蓝色
        (1.0, 1.0, 0.0, 0.8),  # 黄色
    ]
    
    # 定义四种轨迹线颜色 (RGB)
    traj_colors = [
        (1.0, 0.0, 0.0),  # 红色
        (0.0, 1.0, 0.0),  # 绿色
        (0.0, 0.0, 1.0),  # 蓝色
        (1.0, 1.0, 0.0),  # 黄色
    ]

    # 将轨迹分为四段
    segments = split_into_segments(Ts, num_segments)
    
    print(f"总帧数: {len(Ts)}")
    for i, segment in enumerate(segments):
        print(f"段 {i+1}: {len(segment)} 帧")
    
    # 对每段进行采样并绘制
    for i, (segment, color, traj_color) in enumerate(zip(segments, colors, traj_colors)):
        # 采样
        sampled_segment = sample_with_interval(segment, sample_interval)
        print(f"段 {i+1} 采样后: {len(sampled_segment)} 帧")
        
        # 绘制相机
        draw3D.draw_cams(len(sampled_segment), sampled_segment, scene, 
                        color=color, latest_diff=False, cam_scale=0.8)
        
        # 绘制轨迹
        positions = segment[:, :3, 3]  # 使用原始密集轨迹点
        draw3D.draw_trajectory(positions, scene, color=traj_color)

    # 计算俯视相机位姿 (look_at)
    center = mesh.centroid
    bounds = mesh.bounds
    max_dim = np.max(bounds[1] - bounds[0])
    dist = max_dim * topdown_distance
    
    # 这里假设场景 y 为上方向（与仓库 trainer.to_topdown 使用一致）
    if ds_type == "replicaCAD":
        eye = center + np.array([0.0, dist, 0.0])
        up = np.array([1.0, 0.0, 0.0])
    elif ds_type == "scannet":
        eye = center + np.array([0.0, 0.0, dist])
        up = np.array([-1.0, 0.0, 0.0])
    
    R, t = isdf_transform.look_at(eye, center, up)
    T_cam = np.eye(4)
    T_cam[:3, :3] = R
    T_cam[:3, 3] = t

    # 渲染并保存图片
    im = draw3D.capture_scene_im(scene, T_cam, tm_pose=False)
    from PIL import Image
    Image.fromarray(im).save(out_path)
    print(f"Saved top-down image to {out_path}")


ds_type = "replicaCAD"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=int, default=1920)
    parser.add_argument("--h", type=int, default=1080)
    parser.add_argument("--dist", type=float, default=0, help="俯视相机到中心距离，默认基于场景尺寸")
    parser.add_argument("--num_segments", type=int, default=4, help="将轨迹分为几段")
    parser.add_argument("--interval", type=int, default=100, help="采样间隔（帧）")

    args = parser.parse_args()
    dataset_dir = "/home/hjx/dataset/iSDF"
    
    if ds_type == "replicaCAD":
        dataset = "apt_2"
        seq = "nav"
        args.dist=1.15
        args.mesh = os.path.join(dataset_dir, "gt_sdfs", dataset, "mesh.obj")
        args.traj = os.path.join(dataset_dir, "seqs", dataset + "_" + seq, "traj.txt")
        args.out = "/home/hjx/Pictures/ch4/gt/" + dataset + "_" + seq + "_gt.png"
    elif ds_type == "scannet":
        # ids=["04","05","09","10","30","31"]
        id = "10"
        dataset = "scene00" + id + "_00"
        args.mesh = os.path.join(dataset_dir, "gt_sdfs", dataset, "mesh.obj")
        args.traj = os.path.join(dataset_dir, "seqs", dataset, "traj.txt")
        args.dist=1.5
        args.out = "/home/hjx/Pictures/ch4/gt/" + dataset + "_gt.png"
        
    main(
        args.mesh,
        args.traj,
        args.out,
        resolution=(args.w, args.h),
        topdown_distance=args.dist,
        num_segments=args.num_segments,
        sample_interval=args.interval
    )