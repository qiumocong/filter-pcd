import pyrealsense2 as rs

import keyboard
import cv2
from utils import *

SAVE_COUNT = 0


def main(process: bool = False, use_offical_estimation: bool = False, print_fps: bool = False, visualize: bool = False):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_stream = profile.get_stream(rs.stream.depth)
    intr = depth_stream.as_video_stream_profile().get_intrinsics()
    o3d_intr = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy
    )

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

            # frames = pipeline.wait_for_frames()
            # color = frames.get_color_frame()
            # depth = frames.get_depth_frame()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth = aligned_frames.get_depth_frame()
            color = aligned_frames.get_color_frame()

            if not color or not depth:
                continue

            if use_offical_estimation:
                pc.map_to(color)
                points = pc.calculate(depth)
                vtx_raw = np.asanyarray(points.get_vertices())
                vtx = vtx_raw.view(np.float32).reshape(-1, 3)
                vtx = vtx.astype(np.float64)
                pcd.points = o3d.utility.Vector3dVector(vtx)
            else:
                depth_data = np.asanyarray(depth.get_data())
                color_data = np.asanyarray(color.get_data())
                color_data_rgb = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
                estimated_pcd = frames_to_pointcloud(depth_data, color_data_rgb, o3d_intr)
                pcd.points = estimated_pcd.points
                pcd.colors = estimated_pcd.colors

            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.5, origin=[0, 0, 0])
            if not visualize:
                return pcd

            if process:
                processed_pcd = process_point_cloud(pcd, -3, -0.2, 0.05, 6, 1.0)
                pcd.points = processed_pcd.points
                if not visualize:
                    return pcd
                if first_frame:
                    vis.add_geometry(pcd)
                    vis.add_geometry(mesh_frame)
                    first_frame = False
                else:
                    vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            if keyboard.is_pressed('s'):
                save_point_cloud(pcd, './saved_pcd')

            process_time = datetime.now() - dt0
            if process_time.total_seconds() > 0:
                if print_fps:
                    print("FPS = {0:.2f}".format(1 / process_time.total_seconds()))

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main(process=True, use_offical_estimation=False, print_fps=True, visualize=True)
