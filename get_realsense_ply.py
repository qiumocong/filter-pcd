import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import time
import os
from datetime import datetime
import keyboard

SAVE_COUNT = 0


def main(process: bool = False, print_fps: bool = False):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)

    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window("Tests")
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat([0, 0, 1])
        ctr.set_zoom(0.5)

        pcd = o3d.geometry.PointCloud()
        first_frame = True
        pc = rs.pointcloud()

        while True:
            dt0 = datetime.now()

            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()

            if not color or not depth:
                continue

            pc.map_to(color)
            points = pc.calculate(depth)

            vtx_raw = np.asanyarray(points.get_vertices())

            vtx = vtx_raw.view(np.float32).reshape(-1, 3)
            vtx = vtx.astype(np.float64)

            pcd.points = o3d.utility.Vector3dVector(vtx)

            if process:
                pcd = process_point_cloud(pcd, 0, 3, 0.05, 20, 1.0)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0])

            # if first_frame:
            #     vis.add_geometry(pcd)
            #     vis.add_geometry(mesh_frame)
            #     first_frame = False
            # else:
            #     vis.update_geometry(pcd)
            if not first_frame:
                vis.clear_geometries()
            vis.add_geometry(mesh_frame)
            vis.add_geometry(pcd)
            first_frame = False

            vis.poll_events()
            vis.update_renderer()

            if keyboard.is_pressed('s'):
                save_point_cloud(pcd)

            process_time = datetime.now() - dt0
            if process_time.total_seconds() > 0:
                if print_fps:
                    print("FPS = {0:.2f}".format(1 / process_time.total_seconds()))

    finally:
        pipeline.stop()


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


def save_point_cloud(pcd):
    """保存点云到文件"""
    global SAVE_COUNT
    if len(pcd.points) == 0:
        print("点云为空，忽略保存！")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"./saved_pcd/pcd_{timestamp}_{SAVE_COUNT}.pcd"
    print(f"正在保存点云到 {filename} ...", end=" ")
    o3d.io.write_point_cloud(filename, pcd, write_ascii=False)
    print("成功！")
    SAVE_COUNT += 1


if __name__ == "__main__":
    main(True, print_fps=False)
