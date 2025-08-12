import mmcv
import numpy as np
from mmengine import load

from mmdet3d.visualization import Det3DLocalVisualizer

info_file = load('/home/dominik/Git_Repos/Public/mmdetection3d/data/nuscenes-driving/nuscenes_infos_train.pkl')
points = np.fromfile('/home/dominik/Git_Repos/Public/mmdetection3d/data/nuscenes-driving/samples/middle_lidar/1750883585799973120.pcd.bin', dtype=np.float64)
points = points.reshape(-1, 4)[:, :3]
lidar2img = np.array(info_file['data_list'][0]['images']['back_left_camera']['lidar2img'], dtype=np.float32)

visualizer = Det3DLocalVisualizer()
img = mmcv.imread('demo/data/kitti/000008.png')
img = mmcv.imconvert(img, 'bgr', 'rgb')
visualizer.set_image(img)
visualizer.draw_points_on_image(points, lidar2img)
visualizer.show()



import torch
import numpy as np

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes

points = np.fromfile('/nvme_data1/DrivIng_nuScenes/samples/middle_lidar/di_night_2506_1750883585799973120.pcd.bin', dtype=np.float64)
points = points.reshape(-1, 5)
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
bboxes_3d = LiDARInstance3DBoxes(
    torch.tensor([[8.7314, -1.8559, -1.5997, 4.2000, 3.4800, 1.8900,
                   -1.5808]]))
# Draw 3D bboxes
visualizer.draw_bboxes_3d(bboxes_3d)
visualizer.show()
