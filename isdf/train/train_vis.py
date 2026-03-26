#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import os
from datetime import datetime
import argparse
import cv2

import open3d.visualization.gui as gui
from isdf.visualisation import isdf_window
from isdf.modules import trainer


# def export_mesh_and_sdf(isdf_trainer, mesh_path):
#     os.makedirs(os.path.dirname(mesh_path) or ".", exist_ok=True)

#     # Export final mesh snapshot.
#     isdf_trainer.write_mesh(mesh_path)

#     # Export SDF grid and affine transform (voxel index -> world).
#     sdf = isdf_trainer.get_sdf_grid().detach().cpu().numpy()
#     dim = int(isdf_trainer.grid_dim)

#     linear = isdf_trainer.bounds_transform_np[:3, :3] @ np.diag(
#         isdf_trainer.scene_scale_np
#     )
#     if dim > 1:
#         voxel_step = 2.0 / (dim - 1)
#     else:
#         voxel_step = 0.0
#     transform = np.eye(4, dtype=np.float64)
#     transform[:3, :3] = linear * voxel_step
#     transform[:3, 3] = (
#         isdf_trainer.bounds_transform_np[:3, 3]
#         - linear @ np.ones(3, dtype=np.float64)
#     )

#     out_dir = os.path.dirname(mesh_path) or "."
#     np.save(os.path.join(out_dir, "sdf.npy"), sdf)
#     np.savetxt(os.path.join(out_dir, "transform.txt"), transform)


def optim_iter(trainer, t):
    # get/add data---------------------------------------------------------
    new_kf = None
    end = False
    finish_optim = trainer.steps_since_frame == trainer.optim_frames
    if trainer.incremental and (finish_optim or t == 0):
        # After n steps with new frame, check whether to add it to kf set.
        if t == 0:
            add_new_frame = True
        else:
            add_new_frame = trainer.check_keyframe_latest()

        if add_new_frame:
            new_frame_id = trainer.get_latest_frame_id()
            size_dataset = len(trainer.scene_dataset)
            if new_frame_id >= size_dataset:
                end = True
                print("**************************************",
                      "End of sequence",
                      "**************************************")
            else:
                print("Total step time", trainer.tot_step_time)
                print("frame______________________", new_frame_id)

                frame_data = trainer.get_data([new_frame_id])
                trainer.add_frame(frame_data)

                if t == 0:
                    trainer.last_is_keyframe = True
                    trainer.optim_frames = 200

        if t == 0 or (isdf_trainer.last_is_keyframe and not add_new_frame):
            new_kf = isdf_trainer.frames.im_batch_np[-1]
            h = int(new_kf.shape[0] / 6)
            w = int(new_kf.shape[1] / 6)
            new_kf = cv2.resize(new_kf, (w, h))

    # optimisation step---------------------------------------------
    losses, step_time = isdf_trainer.step()
    status = [k + ': {:.6f}  '.format(losses[k]) for k in losses.keys()]
    status = "".join(status) + '-- Step time: {:.2f}  '.format(step_time)

    return status, new_kf, end


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser(description="iSDF.")
    parser.add_argument("--config", type=str, required=True, help="input json config")
    # parser.add_argument(
    #     "--export_mesh_path",
    #     type=str,
    #     default="",
    #     help="optional output mesh file path, e.g. /tmp/final.ply",
    # )
    parser.add_argument(
        "-ni",
        "--no_incremental",
        action="store_false",
        help="disable incremental SLAM option",
    )
    args, _ = parser.parse_known_args()  # ROS adds extra unrecongised args
    config_file = args.config
    incremental = args.no_incremental
    export_mesh_path = args.export_mesh_path.strip()

    if not export_mesh_path:
        now = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
        export_mesh_path = os.path.join("../../results/iSDF", f"vis_final_{now}.ply")

    # init trainer-------------------------------------------------------------
    isdf_trainer = trainer.Trainer(
        device,
        config_file,
        incremental=incremental,
    )

    # open3d vis window --------------------------------------------------------
    app = gui.Application.instance
    app.initialize()
    mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE))
    w = isdf_window.iSDFWindow(
        isdf_trainer,
        optim_iter,
        mono,
    )
    app.run()

    # export_mesh_and_sdf(isdf_trainer, export_mesh_path)
    # print("Exported mesh to", export_mesh_path)
    # print("Exported SDF to", os.path.join(os.path.dirname(export_mesh_path) or ".", "sdf.npy"))
    # print(
    #     "Exported transform to",
    #     os.path.join(os.path.dirname(export_mesh_path) or ".", "transform.txt"),
    # )
