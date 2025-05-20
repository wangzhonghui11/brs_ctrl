import time
import rospy
import numpy as np
from tqdm import tqdm
import os
from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1JoyConInterface
from brs_ctrl.robot_interface import R1Interface
from brs_ctrl.robot_interface.grippers import GalaxeaR1Gripper


neutral_left_arm_qs = np.array([1.56, 2.94, -2.54, 0, 0, 0])
neutral_right_arm_qs = np.array([-1.56, 2.94, -2.54, 0, 0, 0])

def get_motor_port():
    ports_to_try = ["/dev/ttyUSB0", "/dev/ttyUSB1"]
    for port in ports_to_try:
        if os.path.exists(port):
            return port
    raise FileNotFoundError("No available motor port found (tried: %s)" % ports_to_try)

robot = R1Interface(
    left_gripper=GalaxeaR1Gripper(left_or_right="left", gripper_close_stroke=1),
    right_gripper=GalaxeaR1Gripper(left_or_right="right", gripper_close_stroke=1),
)
joycon = R1JoyConInterface(
    ros_publish_functional_buttons=True,
    init_ros_node=False,
    gripper_toggle_mode=True,
)

joylo_arms = JoyLoArmPositionController(
    left_motor_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    right_motor_ids=[8, 9, 10, 11, 12, 13, 14, 15],
    motors_port=get_motor_port(),
    left_arm_joint_signs=[-1, -1, 1, 1, 1, 1],
    right_arm_joint_signs=[-1, -1, -1, 1, 1, 1],
    left_slave_motor_ids=[1, 3],
    left_master_motor_ids=[0, 2],
    right_slave_motor_ids=[9, 11],
    right_master_motor_ids=[8, 10],
    left_arm_joint_reset_positions=neutral_left_arm_qs,
    right_arm_joint_reset_positions=neutral_right_arm_qs,
    multithread_read_joints=True,
)
joylo = JoyLoController(joycon=joycon, joylo_arms=joylo_arms)
last_joint=[0,0,0,0]
link2base = robot._kin_model.get_link_poses_in_base_link(
    curr_left_arm_joint=robot.last_joint_position["left_arm"],
    curr_right_arm_joint=robot.last_joint_position["right_arm"],
    curr_torso_joint=np.asarray(last_joint)
)  # (4, 4)
odom_to_base = robot._kin_model.T_odom2base
print(odom_to_base)
print(link2base)