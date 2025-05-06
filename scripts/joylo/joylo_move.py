import numpy as np
from brs_ctrl.joylo.joylo_arms import JoyLoArmImpedanceController


neutral_left_arm_qs = np.array([1.56, 2.94, -2.54, 0, 0, 0])
neutral_right_arm_qs = np.array([-1.56, 2.94, -2.54, 0, 0, 0])


if __name__ == "__main__":
    joylo_arms = JoyLoArmImpedanceController(
        left_motor_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        right_motor_ids=[8, 9, 10, 11, 12, 13, 14, 15],
        motors_port="/dev/tty_joylo",
        left_arm_joint_signs=[-1, -1, 1, 1, 1, 1],
        right_arm_joint_signs=[-1, -1, -1, 1, 1, 1],
        left_slave_motor_ids=[1, 3],
        left_master_motor_ids=[0, 2],
        right_slave_motor_ids=[9, 11],
        right_master_motor_ids=[8, 10],
        left_arm_Kp=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        left_arm_Kd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        right_arm_Kp=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        right_arm_Kd=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        left_arm_joint_reset_positions=neutral_left_arm_qs,
        right_arm_joint_reset_positions=neutral_right_arm_qs,
    )

    control_started = False

    try:
        while True:
            joylo_arms.set_new_goal(
                {
                    "left": np.array([1.56, 2.94, -2.79, 0, 0, 0]),
                    "right": np.array([-1.56, 2.94, -2.79, 0, 0, 0]),
                }
            )
            if not control_started:
                joylo_arms.start_control()
                control_started = True
    except KeyboardInterrupt:
        joylo_arms.close()
