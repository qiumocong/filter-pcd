from datetime import datetime
import pyrealsense2 as rs
import numpy as np
import open3d as o3d

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

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0])

        if first_frame:
            vis.add_geometry(pcd)
            vis.add_geometry(mesh_frame)
            first_frame = False
        else:
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        process_time = datetime.now() - dt0
        if process_time.total_seconds() > 0:
            print("FPS = {0:.2f}".format(1 / process_time.total_seconds()))

finally:
    pipeline.stop()