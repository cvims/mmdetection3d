from mmdet3d.visualization import Det3DLocalVisualizer

# Suppose you have a data_sample and your points
vis = Visualizer3D(dataset_meta=dataset.metainfo, show=False)

# You can add the LiDAR points
vis.add_datasample(
    name='sample_bev',
    data_input={'points': points},   # (N, 3) or (N, 4)
    data_sample=data_sample,
    draw_gt=True,
    draw_pred=True,
    vis_task='lidar_det',            # important: use lidar task
    out_file='bev_output.png'
)
