import numpy as np
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.robot_interface import R1Interface


neutral_left_arm_qs = np.array([1.56, 2.94, -2.54, 0, 0, 0])
neutral_right_arm_qs = np.array([-1.56, 2.94, -2.54, 0, 0, 0])


if __name__ == "__main__":
    robot = R1Interface()
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
        multithread_read_joints=False,
    )

    control_started = False

    try:
        while True:
            left_robot_arm_q = robot.last_joint_position["left_arm"]
            right_robot_arm_q = robot.last_joint_position["right_arm"]
            joylo_arms.set_new_goal(
                {
                    "left": left_robot_arm_q,
                    "right": right_robot_arm_q,
                }
            )
            if not control_started:
                joylo_arms.start_control()
                control_started = True
    except KeyboardInterrupt:
        joylo_arms.close()
        robot.close()
