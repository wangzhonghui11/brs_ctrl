import time

import numpy as np
import rospy
from brs_ctrl.robot_interface import R1Interface
from brs_ctrl.utils import PointCloudVisualizer


if __name__ == "__main__":
    robot = R1Interface(enable_pointcloud=True)
    time.sleep(3)

    pcd_viz = PointCloudVisualizer()

    while not rospy.is_shutdown():
        latest_pcd = robot.last_pointcloud
        pcd_xyz, pcd_rgb = latest_pcd["xyz"], latest_pcd["rgb"]

        if isinstance(pcd_rgb, np.ndarray):
            pcd_viz(latest_pcd["xyz"], latest_pcd["rgb"])
