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


def main(mesh_path, traj_path, out_path, resolution=(1280, 720), topdown_distance=None,kf_indices=None):
    mesh = trimesh.load(mesh_path, process=False)
    Ts = load_poses(traj_path)
    Ts = to_T_WC(Ts)
    # 如果 poses 是 (N,3,4)，扩展为 (N,4,4)
    if Ts.shape[1:] == (3, 4):
        Tfull = np.tile(np.eye(4), (Ts.shape[0], 1, 1))
        Tfull[:, :3, :4] = Ts
        Ts = Tfull

    if kf_indices is not None:
        Ts = Ts[kf_indices]
    # 创建场景并添加几何
    scene = trimesh.Scene(mesh)
    scene.set_camera()
    scene.camera.fx = 577.870605
    scene.camera.fy = 577.870605
    scene.camera.resolution = resolution

    # draw cameras and trajectory
    # draw_cams expects batch_size, T_WC_batch_np, scene
    draw3D.draw_cams(len(Ts), Ts, scene, latest_diff=False)
    positions = Ts[:, :3, 3]
    draw3D.draw_trajectory(positions, scene, color=(1.0, 0.0, 0.0))

    # 计算俯视相机位姿 (look_at)
    center = mesh.centroid
    bounds = mesh.bounds
    max_dim = np.max(bounds[1] - bounds[0])
    dist = topdown_distance if topdown_distance is not None else max_dim * 1.5
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
    # im 是 numpy.ndarray (H,W,3)
    from PIL import Image

    Image.fromarray(im).save(out_path)
    print(f"Saved top-down image to {out_path}")

ds_type = "scannet"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mesh", required=True, help="GT mesh 文件路径 (obj/ply)")
    # parser.add_argument("--traj", required=True, help="相机位姿文件 (.npy 或 文本: 每行 12/16/3 列)")
    # parser.add_argument("--out", required=True, help="输出图片路径 (png)")
    parser.add_argument("--w", type=int, default=1920)
    parser.add_argument("--h", type=int, default=1080)
    parser.add_argument(
        "--dist", type=float, default=None, help="俯视相机到中心距离，默认基于场景尺寸"
    )

    args = parser.parse_args()
    dataset_dir = "/home/hjx/dataset/iSDF"
    
    if ds_type == "replicaCAD":
        dataset = "apt_3"
        seq = "mnp"
        args.mesh = os.path.join(dataset_dir, "gt_sdfs", dataset, "mesh.obj")
        args.traj = os.path.join(dataset_dir, "seqs", dataset + "_" + seq, "traj.txt")
        args.out = "/home/hjx/Pictures/ch2/gt/" + dataset + "_" + seq + ".png"
    elif ds_type == "scannet":
        id="10"
        dataset = "scene00"+id+"_00"
        args.mesh = os.path.join(dataset_dir, "gt_sdfs", dataset, "mesh.obj")
        args.traj = os.path.join(dataset_dir, "seqs", dataset ,"traj.txt")
        args.out = "/home/hjx/Pictures/ch2/gt/" + dataset + ".png"
        # instric_file = os.path.join(dataset_dir, "seqs", dataset, dataset + ".txt")
    
    main(
        args.mesh,
        args.traj,
        args.out,
        resolution=(args.w, args.h),
        topdown_distance=args.dist,
        kf_indices=None
    )
