# import argparse
# import torch
# from mmengine.config import Config
# from mmengine.runner import load_checkpoint
# from mmdet3d.registry import MODELS, VISUALIZERS
# from mmdet3d.structures import Det3DDataSample
# from mmengine.dataset import pseudo_collate

# from mmengine.registry import init_default_scope
# from mmdet3d.registry import DATASETS

# # Ensure registry is properly loaded
# init_default_scope('mmdet3d')


# def parse_args():
#     parser = argparse.ArgumentParser(description='MMDet3D demo with GTs')
#     parser.add_argument('--pcd', default="/home/dominik/Git_Repos/Public/mmdetection3d/data/nuscenes-driving/samples/middle_lidar/di_day_1706_1750163719000033000.pcd.bin", help='Point cloud file')
#     parser.add_argument('--model', default="/home/dominik/Git_Repos/Public/mmdetection3d/work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-driving-3d/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-driving-3d.py", help='Config file')
#     parser.add_argument('--weights', default="/home/dominik/Git_Repos/Public/mmdetection3d/work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-driving-3d/epoch_10.pth", help='Checkpoint file')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--show', action='store_true', help='Whether to show visualization')
#     parser.add_argument(
#         '--out-dir',
#         default='outputs',
#         type=str,
#         help='Output directory for results')
#     return parser.parse_args()

# def main():
#     args = parse_args()

#     # load config
#     cfg = Config.fromfile(args.model)
#     cfg.model.pop('train_cfg', None)

#     # build model
#     model = MODELS.build(cfg.model)
#     load_checkpoint(model, args.weights, map_location='cpu')
#     model = model.to(args.device).eval()

#     # build dataset (train set to get GT)
#     dataset = DATASETS.build(cfg.train_dataloader.dataset)

#     # try to find the sample with the same pcd file
#     data = None
#     for sample in dataset:
#         if 'points' in sample['inputs']:
#             data = sample
#             break

#     if data is None:
#         raise ValueError(f"Could not find {args.pcd} in dataset annotations!")

#     # convert to Det3DDataSample and run inference
#     data_sample: Det3DDataSample = data['data_samples']

#     batch = pseudo_collate([data]) 
#     with torch.no_grad():
#         pred = model.test_step(batch)[0]

#     # attach prediction to GT sample
#     data_sample.pred_instances_3d = pred.pred_instances_3d

#     # build visualizer
#     cfg.visualizer.save_dir = args.out_dir
#     visualizer = VISUALIZERS.build(cfg.visualizer)

#     visualizer.add_datasample(
#         name='demo',
#         data_input=data['inputs'],
#         data_sample=data_sample,
#         draw_gt=True,
#         draw_pred=True,
#         show=args.show,
#         o3d_save_path=args.out_dir + '/vis.png' if args.out_dir else None,
#         out_file='vis.png' if args.out_dir else None,
#         vis_task="lidar_det"
#     )

# if __name__ == '__main__':
#     main()


import torch
import numpy as np
import open3d as o3d
from mmdet3d.structures import Det3DDataSample
from mmdet3d.models import Base3DDetector

def data_to_datasample(data):
    """
    Convert a dataset sample to a Det3DDataSample
    """
    if 'data_samples' in data:
        return data['data_samples']
    else:
        raise ValueError("Dataset sample does not contain 'data_samples'")

def model_inference(model: Base3DDetector, data: dict) -> Det3DDataSample:
    """
    Run inference on a single data sample and return Det3DDataSample
    """
    model.eval()
    with torch.no_grad():
        # test_step expects a batch (list)
        result = model.test_step([data])
        return result[0]

def render_pointcloud(data_sample: Det3DDataSample, points: np.ndarray, out_file: str = "render.png",
                      pred_score_thr: float = 0.3, width: int = 800, height: int = 600):
    from open3d import geometry
    """
    Render point cloud + GT and predicted boxes headlessly.
    """
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3].detach().cpu().numpy().astype(np.float64))

    # Offscreen renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background([0, 0, 0, 1])
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.point_size = 1.0
    scene.add_geometry("pcd", pcd, mat)

    # Helper to add boxes
    def add_boxes(boxes, color):
        for i, box in enumerate(boxes):
            box = box.numpy()  # shape (9,) l,w,h,yaw,...
            center = box[0:3]  # x,y,z (bottom center)
            dims = box[3:6]    # l,w,h
            yaw = box[6]

            # OrientedBoundingBox expects geometric center, so shift z
            center_obb = center.copy()
            center_obb[2] += dims[2] / 2

            # Rotation matrix around Z-axis
            rot_mat = geometry.get_rotation_matrix_from_xyz([0, 0, yaw])

            obb = geometry.OrientedBoundingBox(center_obb, rot_mat, dims)
            lines = geometry.LineSet.create_from_oriented_bounding_box(obb)
            lines.paint_uniform_color(np.array(color))
            scene.add_geometry(f"box_{i}", lines, mat)

    # Add GT boxes (blue)
    if hasattr(data_sample, 'gt_instances_3d') and data_sample.gt_instances_3d is not None:
        add_boxes(data_sample.gt_instances_3d.bboxes_3d, color=[0, 0, 1])

    # Add predicted boxes (red)
    if hasattr(data_sample, 'pred_instances_3d') and data_sample.pred_instances_3d is not None:
        pred_boxes = data_sample.pred_instances_3d
        if hasattr(pred_boxes, 'scores_3d'):
            keep = pred_boxes.scores_3d > pred_score_thr
            pred_boxes = pred_boxes[keep]
        add_boxes(pred_boxes.tensor, color=[1, 0, 0])

    # Camera setup
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent().max()  # largest dimension of the cloud

    camera = renderer.scene.camera
    # Position the camera some distance away along +Z (or any direction)
    cam_distance = extent * 1.0
    camera_pos = center + np.array([cam_distance, cam_distance, cam_distance])
    camera.look_at(center, camera_pos, [0, 0, 1])  # up vector along Z
    camera.set_projection(60.0, width / height, 0.1, cam_distance * 10, 
                        o3d.cuda.pybind.visualization.rendering.Camera.FovType.Vertical)

    # Render to image
    img = renderer.render_to_image()
    o3d.io.write_image(out_file, img)
    print(f"Saved rendered point cloud to {out_file}")


import os
import argparse
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS, VISUALIZERS
from mmdet3d.structures import Det3DDataSample
from mmengine.dataset import pseudo_collate

from mmengine.registry import init_default_scope
from mmdet3d.registry import DATASETS
from mmengine.runner import Runner

import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['LOCAL_RANK'] = '0'

# Ensure registry is properly loaded
init_default_scope('mmdet3d')


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet3D demo with GTs')
    parser.add_argument('--model', default="work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-driving-3d/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-driving-3d.py", help='Config file')
    parser.add_argument('--weights', default="/home/dominik/Git_Repos/Public/mmdetection3d/work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-driving-3d/epoch_10.pth", help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show visualization')
    parser.add_argument(
        '--out-dir',
        default='outputs',
        type=str,
        help='Output directory for results')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs('vis_test_outputs', exist_ok=True)

    # load config
    cfg = Config.fromfile(args.model)
    cfg.model.pop('train_cfg', None)

    # build model
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.weights, map_location='cpu')
    model = model.to(args.device).eval()

    # build dataset
    dataset = DATASETS.build(cfg.val_dataloader.dataset)

    from mmdet3d.visualization import Det3DLocalVisualizer
    vis = Det3DLocalVisualizer()

    # iterate samples
    for sample in dataset:
        if 'points' in sample['inputs']:

            # convert to Det3DDataSample and run inference
            data_sample: Det3DDataSample = sample['data_samples']
            filename = os.path.basename(data_sample.lidar_path).split('.')[0]

            batch = pseudo_collate([sample]) 
            with torch.no_grad():
                pred = model.test_step(batch)[0]

            # attach prediction to GT sample
            data_sample.pred_instances_3d = pred.pred_instances_3d

            # render image
            render_pointcloud(data_sample, sample['inputs']['points'], out_file=f"vis_test_outputs/{filename}_demo.png")

            # vis.add_datasample(
            #     name='sample_bev',
            #     data_input={'points': sample['inputs']['points']},  # LiDAR points, shape (N, 3/4)
            #     data_sample=data_sample,        # Det3DDataSample with gt_instances_3d
            #     draw_gt=True,
            #     draw_pred=False,
            #     vis_task='lidar_det',           # BEV / LiDAR task
            #     out_file='bev_output.png'       # Save to file
            # )


if __name__ == '__main__':
    # Load LiDAR binary file (float32)
    main()
