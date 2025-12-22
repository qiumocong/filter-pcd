import pyrealsense2 as rs
import numpy as np
import open3d as o3d


def process_point_cloud(pcd,
                        min_depth=0.1,
                        max_depth=3.0,
                        voxel_size=0.02,
                        nb_neighbors=20,
                        std_ratio=2.0):
    """
    对点云进行实时预处理：深度截取 -> 降采样 -> 滤波

    参数:
    min_depth: float 最小深度 (米)
    max_depth: float 最大深度 (米)
    voxel_size: float 降采样体素大小 (米)
    nb_neighbors: int 统计滤波参考的邻域点数
    std_ratio: float 统计滤波的标准差倍数 (越小过滤越严格)

    返回:
    pcd_filtered: 处理后的点云
    """

    # --- 1. 深度区域截取 (Depth Cropping) ---
    # 假设 Z 轴为深度方向（常见于 RGBD 相机）。
    # 使用 AxisAlignedBoundingBox 进行裁剪比遍历 numpy 数组更快。

    # 定义裁剪边界：XY平面设为很大的范围（保留），Z轴设为设定的深度范围
    limit = 1000.0  # 足够大的数，覆盖视场角
    min_bound = np.array([-limit, -limit, min_depth])
    max_bound = np.array([limit, limit, max_depth])

    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    pcd_cropped = pcd.crop(bbox)

    # 检查裁剪后是否还有点，如果没有直接返回空
    if not pcd_cropped.has_points():
        return pcd_cropped

    # --- 2. 点云降采样 (Downsampling) ---
    # 使用 Voxel Grid 下采样，大幅减少点数，提升后续处理速度
    pcd_down = pcd_cropped.voxel_down_sample(voxel_size=voxel_size)

    # --- 3. 点云滤波 (Filtering) ---
    # 使用统计离群值移除 (Statistical Outlier Removal) 去除噪点
    # 半径滤波 (Radius Outlier) 也是一种选择，但统计滤波对去除传感器产生的离散噪点效果较好
    if pcd_down.has_points():
        pcd_filtered, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return pcd_filtered
    else:
        return pcd_down


class RealSensePointCloud:
    def __init__(self, process:bool):
        """初始化 RealSense 相机"""
        # 创建 pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 配置深度流和彩色流
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 启动 pipeline
        print("正在启动 RealSense D435i...")
        self.pipe_profile = self.pipeline.start(self.config)

        # 创建对齐对象（将深度对齐到彩色）
        self.align = rs.align(rs.stream.color)

        # 创建点云对象
        self.pc = rs.pointcloud()

        # Open3D 可视化器
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("RealSense Point Cloud", 1280, 720)

        # 用于存储点云的变量
        self.pcd = o3d.geometry.PointCloud()
        self.first_frame = True

        self.process = process

        print("✓ RealSense D435i 已启动")

    def get_frames(self):
        """获取对齐后的深度和彩色帧"""
        frames = self.pipeline.wait_for_frames()

        # 对齐深度帧到彩色帧
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        return depth_frame, color_frame

    def frames_to_pointcloud(self, depth_frame, color_frame):
        color_raw = o3d.io.read_image(color_frame)
        depth_raw = o3d.io.read_image(depth_frame)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw)
        depth_stream = self.pipe_profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        fx, fy, cx, cy = depth_intrinsics

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width=color_frame.shape[1], height=color_frame.shape[0],
                fx=fx, fy=fy,
                cx=cx, cy=cy
            )

        )
        if self.process:
            self.pcd = process_point_cloud(
                pcd=pcd,
                min_depth=-0.2,  # 截取 1米 到 4米 之间
                max_depth=-1,
                voxel_size=0.05,  # 5cm 体素降采样
                nb_neighbors=20,  # 滤波参数
                std_ratio=2.0
            )
        else:
            self.pcd = pcd
        return self.pcd

    def run(self):
        """运行实时点云可视化"""
        print("开始实时点云可视化...")
        print("按 Q 键退出\n")

        try:
            while True:
                # 获取帧
                depth_frame, color_frame = self.get_frames()
                if depth_frame is None:
                    continue

                # 转换为点云
                pcd = self.frames_to_pointcloud(depth_frame, color_frame)

                # 更新可视化
                if self.first_frame:
                    self.vis.add_geometry(pcd)
                    self.first_frame = False
                else:
                    self.vis.update_geometry(pcd)

                self.vis.poll_events()
                self.vis.update_renderer()

                # 检查窗口是否关闭
                if not self.vis.poll_events():
                    break

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.stop()

    def capture_single_frame(self, filename="captured_pointcloud.pcd"):
        """捕获单帧点云并保存"""
        print("正在捕获点云...")

        # 等待几帧让相机稳定
        for _ in range(30):
            self.pipeline.wait_for_frames()

        # 获取帧
        depth_frame, color_frame = self.get_frames()

        if depth_frame:
            pcd = self.frames_to_pointcloud(depth_frame, color_frame)

            # 保存点云
            o3d.io.write_point_cloud(filename, pcd)
            print(f"✓ 点云已保存到: {filename}")

            # 可视化
            o3d.visualization.draw_geometries([pcd],
                                              window_name="Captured Point Cloud",
                                              width=1280, height=720)
            return pcd

        return None

    def stop(self):
        """停止相机"""
        print("正在关闭 RealSense...")
        self.vis.destroy_window()
        self.pipeline.stop()
        print("✓ 已关闭")


# 使用示例
if __name__ == "__main__":
    rs_pc = RealSensePointCloud()

    # 方式1: 实时可视化
    rs_pc.run()

    # 方式2: 捕获单帧
    rs_pc.capture_single_frame("my_pointcloud.pcd")
    # rs_pc.stop()
