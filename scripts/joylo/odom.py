#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist, Point, Quaternion, Vector3


class OdometryPublisher:
    def __init__(self):
        rospy.init_node('odometry_publisher', anonymous=True)
        self.odom_pub = rospy.Publisher('/camera/odom/sample', Odometry, queue_size=10)
        self.rate = rospy.Rate(15)  # 10Hz

    def publish_odometry(self):
        odom_msg = Odometry()

        # 设置header（时间戳和坐标系）
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # 填充位姿数据 (position和orientation)
        odom_msg.pose.pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),  # 示例坐标
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # 无旋转
        )

        # 填充速度数据 (linear和angular)
        odom_msg.twist.twist = Twist(
            linear=Vector3(x=0.0, y=0.0, z=0.0),  # 示例线速度
            angular=Vector3(x=0.0, y=0.0, z=0.0)  # 示例角速度
        )

        self.odom_pub.publish(odom_msg)
        rospy.loginfo("Published odometry message")

    def run(self):
        while not rospy.is_shutdown():
            self.publish_odometry()
            self.rate.sleep()


if __name__ == '__main__':
    try:
        publisher = OdometryPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass