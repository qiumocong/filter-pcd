#!/usr/bin/env python3

import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import math


def main():
    # 1. 初始化 ROS 节点
    rospy.init_node('point_cloud_publisher_node')

    # 2. 创建发布者 (Topic 名称: "/my_point_cloud")
    pub = rospy.Publisher('/my_point_cloud', PointCloud2, queue_size=10)

    # 设置发布频率 (10Hz)
    rate = rospy.Rate(10)

    rospy.loginfo("开始发布点云数据...")

    while not rospy.is_shutdown():
        # 3. 模拟实时数据生成 (例如生成一个旋转的正弦波点云)
        # 实际使用时，这里替换为你的真实点云数据源
        points = []
        t = rospy.get_time()
        for i in range(1000):
            x = float(i) / 100.0
            y = math.sin(x + t)  # 让波形动起来
            z = math.cos(x + t)

            # (x, y, z) 坐标
            points.append([x, y, z])

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