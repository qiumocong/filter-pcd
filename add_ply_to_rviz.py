#!/usr/bin/env python3

import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import get_realsense_ply as rs_cam
import numpy as np
import math


def main():
    rospy.init_node('point_cloud_publisher_node')
    pub = rospy.Publisher('/my_point_cloud', PointCloud2, queue_size=10)
    # 设置发布频率 (5Hz)
    rate = rospy.Rate(5)
    rospy.loginfo("开始发布点云数据...")
    while not rospy.is_shutdown():
        # 实际使用时，这里替换为你的真实点云数据源
        pcd = rs_cam.main(process=True, use_offical_estimation=False)
        if pcd is not None:
            points = np.asarray(pcd.points).tolist()
        else:
            points = [[0, 0, 0]]
        # 4. 创建 PointCloud2 消息头
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "map"  # 重要：RViz 中 Fixed Frame 需要设置为这个值

        # 5. 生成 PointCloud2 消息
        # create_cloud_xyz32 用于仅包含 XYZ 的简单点云
        pc_msg = pc2.create_cloud_xyz32(header, points)

        # 6. 发布消息
        pub.publish(pc_msg)

        rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass