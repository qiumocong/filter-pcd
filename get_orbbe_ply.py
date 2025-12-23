from utils import *
from pyorbbecsdk import *


def main(process, get_rgb):
    pipeline = Pipeline()
    config = Config()

    # Configure depth stream
    depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    if depth_profile_list is None:
        print("No proper depth profile, cannot generate point cloud")
        return
    depth_profile = depth_profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    has_color_sensor = False
    try:
        # Configure color stream if available
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if profile_list is not None:
            color_profile = profile_list.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            has_color_sensor = True
    except OBError as e:
        print(e)

    pipeline.enable_frame_sync()
    pipeline.start(config)

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    point_cloud_filter = PointCloudFilter()

    vis = o3d.visualization.Visualizer()
    vis.create_window('Orbbe PointCloud', width=1024, height=768)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=[0, 0, 0])
    first_frame = True

    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue

        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue

        color_frame = frames.get_color_frame()
        if has_color_sensor and color_frame is None:
            continue

        frame = align_filter.process(frames)

        point_format = OBFormat.RGB_POINT if has_color_sensor and get_rgb and color_frame is not None else OBFormat.POINT
        point_cloud_filter.set_create_point_format(point_format)

        point_cloud_frame = point_cloud_filter.process(frame)
        if point_cloud_frame is None:
            continue

        xyzrgb = point_cloud_filter.calculate(point_cloud_frame)
        xyz_data = np.array(xyzrgb[:, :3]) / 1000
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_data)
        if get_rgb:
            rgb_data = np.array(xyzrgb[:, 3:6]) / 255
            pcd.colors = o3d.utility.Vector3dVector(rgb_data)

        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        if process:
            pcd = process_point_cloud(pcd, -3, -0.2, 0.05, 6, 1.0)
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(mesh_frame)
        else:
            if first_frame:
                vis.add_geometry(pcd)
                vis.add_geometry(mesh_frame)
                first_frame = False
            else:
                vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    print("stop pipeline")
    pipeline.stop()


if __name__ == "__main__":
    main(False, False)
