import argparse
import numpy as np
import trimesh
from isdf.geometry import transform as isdf_transform
from isdf.visualisation import draw3D
import os


def main(mesh_path, out_path, resolution=(1280, 720), topdown_distance=1.5):
    mesh = trimesh.load(mesh_path, process=False)
   
    # 创建场景并添加几何
    scene = trimesh.Scene(mesh)
    scene.set_camera()
    scene.camera.fx = 577.870605
    scene.camera.fy = 577.870605
    scene.camera.resolution = resolution

    # 计算俯视相机位姿 (look_at)
    center = mesh.centroid
    bounds = mesh.bounds
    max_dim = np.max(bounds[1] - bounds[0])
    dist = max_dim * topdown_distance
    
    # 这里假设场景 y 为上方向（与仓库 trainer.to_topdown 使用一致）
    # if ds_type == "replicaCAD":
    #     eye = center + np.array([0.0, dist, 0.0])
    #     up = np.array([1.0, 0.0, 0.0])
    # elif ds_type == "scannet":
    eye = center + np.array([0.0, -dist, 0.0])
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--w", type=int, default=1920)
    parser.add_argument("--h", type=int, default=1080)
    parser.add_argument("--dist", type=float, default=1.5, help="俯视相机到中心距离，默认基于场景尺寸")
    parser.add_argument("--num_robots", type=int, default=2, help="机器人数量")

    args = parser.parse_args()
    ds_type = "scannet"
    
    #     dataset = dataset_dir.split("/")[-1]
    #     out_dir = "/home/hjx/Pictures/ch4/meshes/apt_3_nav/"
    #     os.makedirs(out_dir, exist_ok=True)
    #     # id = "10"
    dataset_dir = "/home/hjx/MACIM/results/03-25-26_16-34-58_mocim"
    meshes_dir = os.path.join(dataset_dir, "meshes")
    out_dir = "/home/hjx/MACIM/results/03-25-26_16-34-58_mocim"
    os.makedirs(out_dir, exist_ok=True)
    for robot_id in range(args.num_robots):
            mesh_file = f"robot_{robot_id}_mesh.ply"
            mesh_path = os.path.join(meshes_dir, mesh_file)
            
            if os.path.exists(mesh_path):

                out_path = os.path.join(out_dir, f"robot_{robot_id}.png")
                print(f"\n处理 {mesh_file}...")
                
                main(
                    mesh_path,
                    out_path,
                    resolution=(args.w, args.h),
                    topdown_distance=args.dist,
                )
            else:
                print(f"警告: 文件不存在 {mesh_path}")
    
  
  