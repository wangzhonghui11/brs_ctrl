from brs_ctrl.kinematics import R1Kinematics
import numpy as np
kinematics = R1Kinematics()

# 获取变换矩阵格式的位姿
poses = kinematics.get_link_poses_in_base_link(
    curr_left_arm_joint=np.array([1.56, 2.94, -2.54, 0, 0, 0]),
    curr_right_arm_joint=np.array([-1.56, 2.94, -2.54, 0, 0, 0]),
    curr_torso_joint=np.array([0.1, 0.2, 0.3, 0.4]),
    return_matrix=True
)

# 访问特定部位的位姿
left_camera_pose = poses["left_wrist_camera"]  # 4×4变换矩阵
head_pose = poses["head_camera"]

# 获取里程计坐标系变换
odom_to_base = kinematics.T_odom2base