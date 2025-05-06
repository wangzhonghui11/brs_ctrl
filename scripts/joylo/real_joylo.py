import time
import rospy
import numpy as np
from tqdm import tqdm

from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1JoyConInterface
from brs_ctrl.robot_interface import R1Interface
from brs_ctrl.robot_interface.grippers import GalaxeaR1Gripper


neutral_left_arm_qs = np.array([1.56, 2.94, -2.54, 0, 0, 0])
neutral_right_arm_qs = np.array([-1.56, 2.94, -2.54, 0, 0, 0])


if __name__ == "__main__":
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
        motors_port="/dev/tty_joylo",
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

    time.sleep(3)

    alpha = 0.95
    left_joylo_q = None
    right_joylo_q = None

    pbar = tqdm()
    try:
        while not rospy.is_shutdown():
            joylo_arms_q = joylo.q
            left_joylo_q = (
                joylo_arms_q["left"]
                if left_joylo_q is None
                else (1 - alpha) * left_joylo_q + alpha * joylo_arms_q["left"]
            )
            right_joylo_q = (
                joylo_arms_q["right"]
                if right_joylo_q is None
                else (1 - alpha) * right_joylo_q + alpha * joylo_arms_q["right"]
            )
            curr_torso_qs = robot.last_joint_position["torso"]
            joycon_action = joycon.act(curr_torso_qs)
            robot_torso_cmd = np.zeros((4,))
            robot_torso_cmd[:] = joycon_action["torso_cmd"][:]

            robot.control(
                arm_cmd={
                    "left": left_joylo_q,
                    "right": right_joylo_q,
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
        joylo.close()
        pbar.close()
        robot.close()
