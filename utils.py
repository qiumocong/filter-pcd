import numpy as np
import open3d as o3d
from datetime import datetime
import os

SAVE_COUNT = 0


def process_point_cloud(pcd,
                        min_depth=None,
                        max_depth=None,
                        voxel_size=None,
                        nb_neighbors=20,
                        std_ratio=2.0):
    if min_depth is None or max_depth is None:
        pcd_cropped = pcd
    else:
        limit = 1000.0  # 足够大的数，覆盖视场角
        min_bound = np.array([-limit, -limit, min_depth])
        max_bound = np.array([limit, limit, max_depth])

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_cropped = pcd.crop(bbox)

    # 检查裁剪后是否还有点，如果没有直接返回空
    if not pcd_cropped.has_points():
        return pcd_cropped

    # 使用 Voxel Grid 下采样，大幅减少点数，提升后续处理速度
    if voxel_size is None:
        pcd_down = pcd_cropped
    else:
        pcd_down = pcd_cropped.voxel_down_sample(voxel_size=voxel_size)

    # 使用统计离群值移除 (Statistical Outlier Removal) 去除噪点
    if pcd_down.has_points():
        pcd_filtered, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return pcd_filtered
    else:
        return pcd_down


def save_point_cloud(pcd, folder_path):
    """保存点云到文件"""
    global SAVE_COUNT
    if len(pcd.points) == 0:
        print("点云为空，忽略保存！")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(folder_path, f'pcd_{timestamp}_{SAVE_COUNT}.pcd')
    print(f"正在保存点云到 {filename} ...", end=" ")
    o3d.io.write_point_cloud(filename, pcd, write_ascii=False)
    print("成功！")
    SAVE_COUNT += 1