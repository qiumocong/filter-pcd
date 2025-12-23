import open3d as o3d
import numpy as np
import time


def process_point_cloud(pcd,
                        min_depth=None,
                        max_depth=None,
                        voxel_size=None,
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

    # 定义裁剪边界：XY平面设为很大的范围（保留），Z轴设为设定的深度范围
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
    # 半径滤波 (Radius Outlier) 也是一种选择，但统计滤波对去除传感器产生的离散噪点效果较好
    if pcd_down.has_points():
        pcd_filtered, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return pcd_filtered
    else:
        return pcd_down


def run(pcd):
    print("开始实时点云可视化...")
    print("按 Q 键退出\n")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    first_frame = True

    try:
        while True:
            source_pcd = pcd
            # 转换为点云
            processed_pcd = process_point_cloud(
                pcd=source_pcd,
                min_depth=-1,  # 截取 0.2米 到 1米 之间
                max_depth=-0.2,
                voxel_size=0.01,  # 5cm 体素降采样
                nb_neighbors=20,  # 滤波参数
                std_ratio=0.5
            )

            # 更新可视化
            if first_frame:
                vis.add_geometry(processed_pcd)
                first_frame = False
            else:
                vis.update_geometry(processed_pcd)

            vis.poll_events()
            vis.update_renderer()
            print(time.time())

            # 检查窗口是否关闭
            if not vis.poll_events():
                break
    except KeyboardInterrupt:
        print("\n用户中断")


if __name__ == '__main__':
    source_pcd = o3d.io.read_point_cloud(r"./1.ply")
    # source_pcd = o3d.io.read_point_cloud(r"./point_clouds/table_scene_lms400.pcd")

    # print("生成测试点云...")
    # pts = np.random.rand(100000, 3) * 5.0  # 0-5米范围内的随机点
    # # 添加一些人为的离群噪点（很远的点）
    # noise = np.random.rand(500, 3) * 10.0
    # pts = np.vstack((pts, noise))
    #
    # source_pcd = o3d.geometry.PointCloud()
    # source_pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"原始点数: {len(source_pcd.points)}")
    run(source_pcd)

    start_time = time.time()

    # processed_pcd = process_point_cloud(
    #     pcd=source_pcd,
    #     min_depth=-100,  # 截取 0.2米 到 1米 之间
    #     max_depth=100,
    #     voxel_size=0.025,  # 5cm 体素降采样
    #     nb_neighbors=6,  # 滤波参数
    #     std_ratio=0.8
    # )
    processed_pcd = process_point_cloud(
        pcd=source_pcd,
        min_depth=None,  # 截取 0.2米 到 1米 之间
        max_depth=100,
        voxel_size=0.01,  # 5cm 体素降采样
        nb_neighbors=20,  # 滤波参数
        std_ratio=0.5
    )

    end_time = time.time()

    print(f"处理耗时: {(end_time - start_time) * 1000:.2f} ms")
    print(f"处理后点数: {len(processed_pcd.points)}")

    # 3. 可视化对比 (原始点云为灰色，处理后为红色)
    source_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    processed_pcd.paint_uniform_color([1, 0, 0])

    # 为了看清效果，将原始点云稍微平移
    source_pcd.translate((-4, 0, 0))

    o3d.visualization.draw_geometries([processed_pcd, source_pcd], window_name="处理结果")

