#!/usr/bin/env python3
# 在 WSL 下运行。依赖: pyrealsense2, numpy, open3d, opencv-python, rospy, sensor_msgs
# 用法示例：
#   source /opt/ros/noetic/setup.bash
#   roscore  # 另开一个终端运行
#   python3 point_cloud_bag_pub.py --bag /mnt/c/Users/Noko/path/to/your.bag --rate 30 --frame map \
#     --use-official-estimation false --min-depth -3 --max-depth -0.2 --voxel-size 0.05 --nb-neighbors 6 --std-ratio 1.0 --stride 2

import argparse
import os
import numpy as np
import cv2
import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import pyrealsense2 as rs
import open3d as o3d

# 如果你已有 process_point_cloud.py 中的这些函数，可改为:
#   from process_point_cloud import frames_to_pointcloud, process_point_cloud
# 这里内联实现，避免额外依赖模块路径。
def frames_to_pointcloud(depth_frame_np, color_frame_np, depth_intrinsics_o3d):
    o3d_color = o3d.geometry.Image(color_frame_np)
    o3d_depth = o3d.geometry.Image(depth_frame_np)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth,
        depth_scale=1000.0,   # 深度值缩放（z16 的毫米值转米：1000.0）
        depth_trunc=4.0,      # 截断距离（米），按需调整
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, depth_intrinsics_o3d)
    return pcd

def process_point_cloud(pcd,
                        min_depth=None,
                        max_depth=None,
                        voxel_size=None,
                        nb_neighbors=20,
                        std_ratio=2.0):
    if min_depth is None or max_depth is None:
        pcd_cropped = pcd
    else:
        limit = 1000.0
        min_bound = np.array([-limit, -limit, min_depth])
        max_bound = np.array([limit,  limit,  max_depth])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_cropped = pcd.crop(bbox)

    if not pcd_cropped.has_points():
        return pcd_cropped

    pcd_down = pcd_cropped if voxel_size is None else pcd_cropped.voxel_down_sample(voxel_size=voxel_size)
    if pcd_down.has_points():
        pcd_filtered, _ = pcd_down.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return pcd_filtered
    return pcd_down

def o3d_intrinsics_from_rs_stream(depth_stream):
    intr = depth_stream.as_video_stream_profile().get_intrinsics()
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )
    return o3d_intr

def create_pc2_msg(xyz: np.ndarray, frame_id: str):
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    return pc2.create_cloud_xyz32(header, xyz.tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="输入 .bag 文件路径（如 /mnt/c/Users/Noko/Videos/capture.bag）")
    parser.add_argument("--rate", type=float, default=30.0, help="发布频率（Hz）")
    parser.add_argument("--frame", type=str, default="map", help="PointCloud2 的 frame_id（RViz Fixed Frame 要一致）")
    parser.add_argument("--use-official-estimation", type=str, default="false",
                        help="true/false：使用 pyrealsense2 的官方点云估计（快）还是 Open3D 估计（灵活）")
    parser.add_argument("--stride", type=int, default=2, help="点云下采样步长（发布前的稀疏化，2 表示取一半）")

    # 后处理参数（只在 use_official_estimation=false 时生效）
    parser.add_argument("--min-depth", type=float, default=None, help="裁剪最小深度 z（米），例如 -3")
    parser.add_argument("--max-depth", type=float, default=None, help="裁剪最大深度 z（米），例如 -0.2")
    parser.add_argument("--voxel-size", type=float, default=0.05, help="体素下采样大小（米），如 0.05")
    parser.add_argument("--nb-neighbors", type=int, default=6, help="统计滤波的近邻数")
    parser.add_argument("--std-ratio", type=float, default=1.0, help="统计滤波的标准差系数")

    args = parser.parse_args()
    use_official = str(args.use_official_estimation).lower() in ("1", "true", "yes", "y")

    if not os.path.exists(args.bag):
        raise FileNotFoundError(f"bag 不存在: {args.bag}")

    rospy.init_node("point_cloud_from_bag")
    pub = rospy.Publisher("/my_point_cloud", PointCloud2, queue_size=2)
    rate = rospy.Rate(args.rate)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(args.bag, repeat_playback=False)
    profile = pipeline.start(config)

    playback = profile.get_device().as_playback()
    try:
        playback.set_real_time(False)  # 非实时播放，避免处理跟不上时“卡住”
    except Exception:
        pass

    align = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    # 构建 Open3D 相机内参（仅在自定义估计时需要）
    depth_stream = profile.get_stream(rs.stream.depth)
    o3d_intr = o3d_intrinsics_from_rs_stream(depth_stream)

    rospy.loginfo(f"开始从 {args.bag} 播放，发布到 /my_point_cloud，估计方式: {'官方' if use_official else 'Open3D'}")
    frames_published = 0

    try:
        while not rospy.is_shutdown():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
            except Exception:
                rospy.loginfo("播放结束或取帧超时，退出。")
                break

            aligned = align.process(frames)
            depth = aligned.get_depth_frame()
            color = aligned.get_color_frame()
            if not depth or not color:
                continue

            if use_official:
                # 官方点云估计
                pc.map_to(color)
                points = pc.calculate(depth)
                vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                xyz = vtx.astype(np.float32)
            else:
                # Open3D 点云估计（RGBD → 点云）
                depth_np = np.asanyarray(depth.get_data())
                color_np = np.asanyarray(color.get_data())
                # 保证为 RGB 排列（多数情况下 color 流是 rgb8，直接使用）
                # 如果你的彩色流是 BGR，请启用下面这一行：
                # color_np = cv2.cvtColor(color_np, cv2.COLOR_BGR2RGB)

                pcd = frames_to_pointcloud(depth_np, color_np, o3d_intr)

                # 坐标系调整，与 RViz 常用坐标一致（可按你实际需求调整）
                # 原代码中使用 [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]
                # 这里先按原逻辑翻转 Y/Z
                pcd.transform([[1, 0, 0, 0],
                               [0,-1, 0, 0],
                               [0, 0,-1, 0],
                               [0, 0, 0, 1]])

                # 后处理（裁剪、体素下采样、统计滤波）
                pcd = process_point_cloud(
                    pcd,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    voxel_size=args.voxel_size,
                    nb_neighbors=args.nb_neighbors,
                    std_ratio=args.std_ratio
                )

                xyz = np.asarray(pcd.points, dtype=np.float32)

            # 移除零点（可选）
            if xyz.size == 0:
                continue
            mask = ~(np.isclose(xyz[:,0], 0.0) & np.isclose(xyz[:,1], 0.0) & np.isclose(xyz[:,2], 0.0))
            xyz = xyz[mask]

            # 额外下采样（发布前稀疏化，减少 RViz 压力）
            if args.stride > 1:
                xyz = xyz[::args.stride]

            msg = create_pc2_msg(xyz, frame_id=args.frame)
            pub.publish(msg)
            frames_published += 1
            rate.sleep()

        rospy.loginfo(f"发布完成，共发布帧数: {frames_published}")
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()