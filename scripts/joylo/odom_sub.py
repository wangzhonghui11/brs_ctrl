#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry


class OdometrySubscriber:
    def __init__(self):
        rospy.init_node('odometry_subscriber', anonymous=True)

        # 初始化数据存储结构（与你的回调函数一致）
        self._odom_data = {
            "position": np.zeros((1, 3)),
            "orientation": np.zeros((1, 4)),
            "linear_velocity": np.zeros((1, 3)),
            "angular_velocity": np.zeros((1, 3)),
            "stamp": np.zeros(1)
        }

        # 订阅Odometry话题（替换为实际话题名）
        rospy.Subscriber("/camera/odom/sample", Odometry, self._update_odom_callback)
        rospy.loginfo("Subscribed to /odom")

    def _update_odom_callback(self, odom_msg: Odometry):
        """ 处理Odometry消息的回调函数 """
        # 时间戳转换（秒）
        stamp = odom_msg.header.stamp.to_sec()

        # 提取位姿和速度数据
        position = np.array([
            odom_msg.pose.pose.position.x,
            odom_msg.pose.pose.position.y,
            odom_msg.pose.pose.position.z
        ])

        orientation = np.array([
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        ])  # 注意：PyBullet使用[x,y,z,w]顺序

        linear_velocity = np.array([
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.linear.z
        ])

        angular_velocity = np.array([
            odom_msg.twist.twist.angular.x,
            odom_msg.twist.twist.angular.y,
            odom_msg.twist.twist.angular.z
        ])

        # 更新数据存储
        self._odom_data["position"] = position[np.newaxis, :]
        self._odom_data["orientation"] = orientation[np.newaxis, :]
        self._odom_data["linear_velocity"] = linear_velocity[np.newaxis, :]
        self._odom_data["angular_velocity"] = angular_velocity[np.newaxis, :]
        self._odom_data["stamp"] = np.array([stamp])

        # 打印示例（实际应用中可替换为其他处理逻辑）
        rospy.loginfo_throttle(5.0,
                               f"Position: {position}, Orientation: {orientation}\n"
                               f"Linear Vel: {linear_velocity}, Angular Vel: {angular_velocity}"
                               ,)
        print("stamp:",stamp)
    def run(self):
        rospy.spin()  # 保持节点运行


if __name__ == '__main__':
    try:
        subscriber = OdometrySubscriber()
        subscriber.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Subscriber node terminated.")