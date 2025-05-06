
#https://mirrors.aliyun.com/pypi/simple
import time
import rospy
import numpy as np
from sympy import false
from tqdm import tqdm

from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1JoyConInterface
from brs_ctrl.robot_interface import R1Interface
from brs_ctrl.robot_interface.grippers import GalaxeaR1Gripper

if __name__ == "__main__":
    robot = R1Interface(
        left_gripper=GalaxeaR1Gripper(left_or_right="left", gripper_close_stroke=1),
        right_gripper=GalaxeaR1Gripper(left_or_right="right", gripper_close_stroke=1),
    )
    joycon = R1JoyConInterface(
        ros_publish_functional_buttons=True,
        init_ros_node=False,
        gripper_toggle_mode=False,
    )
    pbar = tqdm()
    try:
        while not rospy.is_shutdown():
            curr_torso_qs = robot.last_joint_position["torso"]
            joycon_action = joycon.act(curr_torso_qs)
            robot_torso_cmd = np.zeros((4,))
            robot_torso_cmd[:] = joycon_action["torso_cmd"][:]
            robot.control(
                arm_cmd={
                    "left": None,
                    "right": None,
                },
                gripper_cmd={
                    "left": joycon_action["gripper_cmd"]["left"],
                    "right": joycon_action["gripper_cmd"]["right"],
                },
                torso_cmd=robot_torso_cmd,
                base_cmd=joycon_action["mobile_base_cmd"],
            )

            pbar.update(1)

    except KeyboardInterrupt:
        joycon.close()
        pbar.close()
        robot.close()

