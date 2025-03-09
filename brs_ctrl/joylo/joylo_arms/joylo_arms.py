from typing import Sequence, Union, Optional, Dict

import numpy as np

from brs_ctrl.joylo.joylo_arms.dxl.position import DXLPositionController
from brs_ctrl.joylo.joylo_arms.dxl.joint_impedance import DXLJointImpedanceController


class JoyLoArmPositionController:
    LEFT_JOINT_LIMIT_LOW = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    LEFT_JOINT_LIMIT_HIGH = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])

    RIGHT_JOINT_LIMIT_LOW = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    RIGHT_JOINT_LIMIT_HIGH = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])

    def __init__(
        self,
        *,
        left_motor_ids: Sequence[int],
        right_motor_ids: Sequence[int],
        motors_port: str,
        baudrate: int = 3000000,
        left_arm_joint_reset_positions: Optional[np.ndarray] = None,
        right_arm_joint_reset_positions: Optional[np.ndarray] = None,
        left_arm_joint_signs: Sequence[int],
        right_arm_joint_signs: Sequence[int],
        left_slave_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        left_master_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        right_slave_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        right_master_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        multithread_read_joints: bool = False,
    ):
        if left_arm_joint_reset_positions is None:
            left_arm_joint_reset_positions = np.array(
                [1.3691489361702127, 3.094255319148936, -2.8148936170212764, 0, 0, 0]
            )
        assert left_arm_joint_reset_positions.shape == self.LEFT_JOINT_LIMIT_LOW.shape
        assert np.all(
            self.LEFT_JOINT_LIMIT_LOW <= left_arm_joint_reset_positions
        ) and np.all(left_arm_joint_reset_positions <= self.LEFT_JOINT_LIMIT_HIGH)
        assert len(left_arm_joint_signs) == len(left_arm_joint_reset_positions)

        if right_arm_joint_reset_positions is None:
            right_arm_joint_reset_positions = np.array(
                [-1.3953191489361702, 3.07468085106383, -2.793191489361702, 0, 0, 0]
            )
        assert right_arm_joint_reset_positions.shape == self.RIGHT_JOINT_LIMIT_LOW.shape
        assert np.all(
            self.RIGHT_JOINT_LIMIT_LOW <= right_arm_joint_reset_positions
        ) and np.all(right_arm_joint_reset_positions <= self.RIGHT_JOINT_LIMIT_HIGH)
        assert len(right_arm_joint_signs) == len(right_arm_joint_reset_positions)

        assert set(left_motor_ids).isdisjoint(
            right_motor_ids
        ), "Motor ids should be different."

        motor_idx = 0
        left_motors_idxs, right_motors_idxs = [], []
        motors_ids = []
        for motor_id in left_motor_ids:
            motors_ids.append(motor_id)
            if motor_id not in left_slave_motor_ids:
                left_motors_idxs.append(motor_idx)
                motor_idx += 1
        for motor_id in right_motor_ids:
            motors_ids.append(motor_id)
            if motor_id not in right_slave_motor_ids:
                right_motors_idxs.append(motor_idx)
                motor_idx += 1
        self._motors_idxs = {
            "left": np.array(left_motors_idxs, dtype=np.int32),
            "right": np.array(right_motors_idxs, dtype=np.int32),
        }
        slave_motor_ids = left_slave_motor_ids + right_slave_motor_ids
        master_motor_ids = left_master_motor_ids + right_master_motor_ids
        self._dxl_controller = DXLPositionController(
            ids=motors_ids,
            port=motors_port,
            baudrate=baudrate,
            slave_motor_ids=slave_motor_ids,
            master_motor_ids=master_motor_ids,
            multithread_read_joints=multithread_read_joints,
            gripper_ids=None,
        )
        if multithread_read_joints:
            self._dxl_controller.start_read_thread()

        # initial calibration
        dxl_init_positions = self._dxl_controller.curr_positions_and_velocities[0]

        dxl_left_arm_init_positions = dxl_init_positions[self._motors_idxs["left"]]
        dxl_right_arm_init_positions = dxl_init_positions[self._motors_idxs["right"]]

        assert np.all(dxl_left_arm_init_positions > -40960) and np.all(
            dxl_left_arm_init_positions < 40960
        ), (
            "Some joints in the left arm rotate more than 10 turns in one direction. "
            "Consider rebooting or turning back."
        )
        assert np.all(dxl_right_arm_init_positions > -40960) and np.all(
            dxl_right_arm_init_positions < 40960
        ), (
            "Some joints in the right arm rotate more than 10 turns in one direction. "
            "Consider rebooting or turning back."
        )
        left_arm_joint_signs = np.array(left_arm_joint_signs)
        right_arm_joint_signs = np.array(right_arm_joint_signs)
        self._left_dxl_to_robot_joint_fn = (
            lambda x: (x - dxl_left_arm_init_positions)
            * left_arm_joint_signs
            * np.pi
            / 2048
            + left_arm_joint_reset_positions
        )
        self._left_robot_joint_to_dxl_fn = lambda x: np.array(
            (x - left_arm_joint_reset_positions) * 2048 / np.pi / left_arm_joint_signs
            + dxl_left_arm_init_positions,
            dtype=np.int32,
        )
        self._right_dxl_to_robot_joint_fn = (
            lambda x: (x - dxl_right_arm_init_positions)
            * right_arm_joint_signs
            * np.pi
            / 2048
            + right_arm_joint_reset_positions
        )
        self._right_robot_joint_to_dxl_fn = lambda x: np.array(
            (x - right_arm_joint_reset_positions) * 2048 / np.pi / right_arm_joint_signs
            + dxl_right_arm_init_positions,
            dtype=np.int32,
        )

    def set_new_goal(self, goal_dict: Dict[str, np.ndarray]):
        assert set(goal_dict.keys()) == {
            "left",
            "right",
        }, "Invalid keys in goal_dict. Expect {'left', 'right'}."
        left_arm_goal, right_arm_goal = goal_dict["left"], goal_dict["right"]
        assert left_arm_goal.shape == self.LEFT_JOINT_LIMIT_LOW.shape
        assert np.all(self.LEFT_JOINT_LIMIT_LOW <= left_arm_goal) and np.all(
            left_arm_goal <= self.LEFT_JOINT_LIMIT_HIGH
        )
        assert right_arm_goal.shape == self.RIGHT_JOINT_LIMIT_LOW.shape
        assert np.all(self.RIGHT_JOINT_LIMIT_LOW <= right_arm_goal) and np.all(
            right_arm_goal <= self.RIGHT_JOINT_LIMIT_HIGH
        )

        left_arm_goal_motor_positions = self._left_robot_joint_to_dxl_fn(left_arm_goal)
        right_arm_goal_motor_positions = self._right_robot_joint_to_dxl_fn(
            right_arm_goal
        )
        motor_goal_positions = np.concatenate(
            [
                left_arm_goal_motor_positions,
                right_arm_goal_motor_positions,
            ],
            axis=0,
        )

        self._dxl_controller.set_new_goal(motor_goal_positions)

    def start_control(self):
        left_arm_raw_joint_low = self._left_robot_joint_to_dxl_fn(
            self.LEFT_JOINT_LIMIT_LOW
        )
        left_arm_raw_joint_high = self._left_robot_joint_to_dxl_fn(
            self.LEFT_JOINT_LIMIT_HIGH
        )
        left_arm_joint_low = np.minimum(left_arm_raw_joint_low, left_arm_raw_joint_high)
        left_arm_joint_high = np.maximum(
            left_arm_raw_joint_low, left_arm_raw_joint_high
        )
        right_arm_raw_joint_low = self._right_robot_joint_to_dxl_fn(
            self.RIGHT_JOINT_LIMIT_LOW
        )
        right_arm_raw_joint_high = self._right_robot_joint_to_dxl_fn(
            self.RIGHT_JOINT_LIMIT_HIGH
        )
        right_arm_joint_low = np.minimum(
            right_arm_raw_joint_low, right_arm_raw_joint_high
        )
        right_arm_joint_high = np.maximum(
            right_arm_raw_joint_low, right_arm_raw_joint_high
        )

        dxl_position_limit_low = np.concatenate(
            [
                left_arm_joint_low,
                right_arm_joint_low,
            ],
            axis=0,
        )
        dxl_position_limit_high = np.concatenate(
            [
                left_arm_joint_high,
                right_arm_joint_high,
            ],
            axis=0,
        )

        self._dxl_controller.start_control(
            position_limit_low=dxl_position_limit_low,
            position_limit_high=dxl_position_limit_high,
        )

    def close(self):
        self._dxl_controller.close()

    def __del__(self):
        self.close()

    @property
    def q(self):
        motors_positions = self._dxl_controller.curr_positions_and_velocities[0]
        left_arm_q = motors_positions[self._motors_idxs["left"]]
        right_arm_q = motors_positions[self._motors_idxs["right"]]
        left_arm_q = np.clip(
            self._left_dxl_to_robot_joint_fn(left_arm_q),
            a_min=self.LEFT_JOINT_LIMIT_LOW,
            a_max=self.LEFT_JOINT_LIMIT_HIGH,
        )
        right_arm_q = np.clip(
            self._right_dxl_to_robot_joint_fn(right_arm_q),
            a_min=self.RIGHT_JOINT_LIMIT_LOW,
            a_max=self.RIGHT_JOINT_LIMIT_HIGH,
        )
        return {
            "left": left_arm_q,
            "right": right_arm_q,
        }


class JoyLoArmImpedanceController:
    LEFT_JOINT_LIMIT_LOW = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    LEFT_JOINT_LIMIT_HIGH = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])

    RIGHT_JOINT_LIMIT_LOW = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    RIGHT_JOINT_LIMIT_HIGH = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])

    def __init__(
        self,
        *,
        left_motor_ids: Sequence[int],
        right_motor_ids: Sequence[int],
        motors_port: str,
        baudrate: int = 3000000,
        left_arm_joint_reset_positions: Optional[np.ndarray] = None,
        right_arm_joint_reset_positions: Optional[np.ndarray] = None,
        left_arm_joint_signs: Sequence[int],
        right_arm_joint_signs: Sequence[int],
        left_slave_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        left_master_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        right_slave_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        right_master_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        left_arm_Kp: Union[float, Sequence[float]],
        left_arm_Kd: Union[float, Sequence[float]],
        right_arm_Kp: Union[float, Sequence[float]],
        right_arm_Kd: Union[float, Sequence[float]],
    ):
        if left_arm_joint_reset_positions is None:
            left_arm_joint_reset_positions = np.array(
                [1.3691489361702127, 3.094255319148936, -2.8148936170212764, 0, 0, 0]
            )
        assert left_arm_joint_reset_positions.shape == self.LEFT_JOINT_LIMIT_LOW.shape
        assert np.all(
            self.LEFT_JOINT_LIMIT_LOW <= left_arm_joint_reset_positions
        ) and np.all(left_arm_joint_reset_positions <= self.LEFT_JOINT_LIMIT_HIGH)
        assert len(left_arm_joint_signs) == len(left_arm_joint_reset_positions)

        if right_arm_joint_reset_positions is None:
            right_arm_joint_reset_positions = np.array(
                [-1.3953191489361702, 3.07468085106383, -2.793191489361702, 0, 0, 0]
            )
        assert right_arm_joint_reset_positions.shape == self.RIGHT_JOINT_LIMIT_LOW.shape
        assert np.all(
            self.RIGHT_JOINT_LIMIT_LOW <= right_arm_joint_reset_positions
        ) and np.all(right_arm_joint_reset_positions <= self.RIGHT_JOINT_LIMIT_HIGH)
        assert len(right_arm_joint_signs) == len(right_arm_joint_reset_positions)

        assert set(left_motor_ids).isdisjoint(
            right_motor_ids
        ), "Motor ids should be different."

        motor_idx = 0
        left_motors_idxs, right_motors_idxs = [], []
        motors_ids = []
        for motor_id in left_motor_ids:
            motors_ids.append(motor_id)
            if motor_id not in left_slave_motor_ids:
                left_motors_idxs.append(motor_idx)
                motor_idx += 1
        for motor_id in right_motor_ids:
            motors_ids.append(motor_id)
            if motor_id not in right_slave_motor_ids:
                right_motors_idxs.append(motor_idx)
                motor_idx += 1
        self._motors_idxs = {
            "left": np.array(left_motors_idxs, dtype=np.int32),
            "right": np.array(right_motors_idxs, dtype=np.int32),
        }
        slave_motor_ids = left_slave_motor_ids + right_slave_motor_ids
        master_motor_ids = left_master_motor_ids + right_master_motor_ids

        if isinstance(left_arm_Kp, float):
            left_arm_Kp = [left_arm_Kp] * 6
        if isinstance(left_arm_Kd, float):
            left_arm_Kd = [left_arm_Kd] * 6
        if isinstance(right_arm_Kp, float):
            right_arm_Kp = [right_arm_Kp] * 6
        if isinstance(right_arm_Kd, float):
            right_arm_Kd = [right_arm_Kd] * 6
        Kp = np.concatenate([left_arm_Kp, right_arm_Kp])
        Kd = np.concatenate([left_arm_Kd, right_arm_Kd])

        self._dxl_controller = DXLJointImpedanceController(
            ids=motors_ids,
            port=motors_port,
            baudrate=baudrate,
            slave_motor_ids=slave_motor_ids,
            master_motor_ids=master_motor_ids,
            Kp=Kp,
            Kd=Kd,
        )

        # initial calibration
        dxl_init_positions = self._dxl_controller.curr_positions_and_velocities[0]

        dxl_left_arm_init_positions = dxl_init_positions[self._motors_idxs["left"]]
        dxl_right_arm_init_positions = dxl_init_positions[self._motors_idxs["right"]]

        assert np.all(dxl_left_arm_init_positions > -40960) and np.all(
            dxl_left_arm_init_positions < 40960
        ), (
            "Some joints in the left arm rotate more than 10 turns in one direction. "
            "Consider rebooting or turning back."
        )
        assert np.all(dxl_right_arm_init_positions > -40960) and np.all(
            dxl_right_arm_init_positions < 40960
        ), (
            "Some joints in the right arm rotate more than 10 turns in one direction. "
            "Consider rebooting or turning back."
        )
        left_arm_joint_signs = np.array(left_arm_joint_signs)
        right_arm_joint_signs = np.array(right_arm_joint_signs)
        self._left_dxl_to_robot_joint_fn = (
            lambda x: (x - dxl_left_arm_init_positions)
            * left_arm_joint_signs
            * np.pi
            / 2048
            + left_arm_joint_reset_positions
        )
        self._left_robot_joint_to_dxl_fn = lambda x: np.array(
            (x - left_arm_joint_reset_positions) * 2048 / np.pi / left_arm_joint_signs
            + dxl_left_arm_init_positions,
            dtype=np.int32,
        )
        self._right_dxl_to_robot_joint_fn = (
            lambda x: (x - dxl_right_arm_init_positions)
            * right_arm_joint_signs
            * np.pi
            / 2048
            + right_arm_joint_reset_positions
        )
        self._right_robot_joint_to_dxl_fn = lambda x: np.array(
            (x - right_arm_joint_reset_positions) * 2048 / np.pi / right_arm_joint_signs
            + dxl_right_arm_init_positions,
            dtype=np.int32,
        )

    def set_new_goal(self, goal_dict: Dict[str, np.ndarray]):
        assert set(goal_dict.keys()) == {
            "left",
            "right",
        }, "Invalid keys in goal_dict. Expect {'left', 'right'}."
        left_arm_goal, right_arm_goal = goal_dict["left"], goal_dict["right"]
        assert left_arm_goal.shape == self.LEFT_JOINT_LIMIT_LOW.shape
        assert np.all(self.LEFT_JOINT_LIMIT_LOW <= left_arm_goal) and np.all(
            left_arm_goal <= self.LEFT_JOINT_LIMIT_HIGH
        )
        assert right_arm_goal.shape == self.RIGHT_JOINT_LIMIT_LOW.shape
        assert np.all(self.RIGHT_JOINT_LIMIT_LOW <= right_arm_goal) and np.all(
            right_arm_goal <= self.RIGHT_JOINT_LIMIT_HIGH
        )

        left_arm_goal_motor_positions = self._left_robot_joint_to_dxl_fn(left_arm_goal)
        right_arm_goal_motor_positions = self._right_robot_joint_to_dxl_fn(
            right_arm_goal
        )
        motor_goal_positions = np.concatenate(
            [
                left_arm_goal_motor_positions,
                right_arm_goal_motor_positions,
            ],
            axis=0,
        )

        self._dxl_controller.set_new_goal(motor_goal_positions)

    def start_control(self):
        left_arm_raw_joint_low = self._left_robot_joint_to_dxl_fn(
            self.LEFT_JOINT_LIMIT_LOW
        )
        left_arm_raw_joint_high = self._left_robot_joint_to_dxl_fn(
            self.LEFT_JOINT_LIMIT_HIGH
        )
        left_arm_joint_low = np.minimum(left_arm_raw_joint_low, left_arm_raw_joint_high)
        left_arm_joint_high = np.maximum(
            left_arm_raw_joint_low, left_arm_raw_joint_high
        )
        right_arm_raw_joint_low = self._right_robot_joint_to_dxl_fn(
            self.RIGHT_JOINT_LIMIT_LOW
        )
        right_arm_raw_joint_high = self._right_robot_joint_to_dxl_fn(
            self.RIGHT_JOINT_LIMIT_HIGH
        )
        right_arm_joint_low = np.minimum(
            right_arm_raw_joint_low, right_arm_raw_joint_high
        )
        right_arm_joint_high = np.maximum(
            right_arm_raw_joint_low, right_arm_raw_joint_high
        )

        dxl_position_limit_low = np.concatenate(
            [
                left_arm_joint_low,
                right_arm_joint_low,
            ],
            axis=0,
        )
        dxl_position_limit_high = np.concatenate(
            [
                left_arm_joint_high,
                right_arm_joint_high,
            ],
            axis=0,
        )

        self._dxl_controller.start_control(
            position_limit_low=dxl_position_limit_low,
            position_limit_high=dxl_position_limit_high,
        )

    def close(self):
        self._dxl_controller.close()

    def __del__(self):
        self.close()

    @property
    def q(self):
        motors_positions = self._dxl_controller.curr_positions_and_velocities[0]
        left_arm_q = motors_positions[self._motors_idxs["left"]]
        right_arm_q = motors_positions[self._motors_idxs["right"]]
        left_arm_q = np.clip(
            self._left_dxl_to_robot_joint_fn(left_arm_q),
            a_min=self.LEFT_JOINT_LIMIT_LOW,
            a_max=self.LEFT_JOINT_LIMIT_HIGH,
        )
        right_arm_q = np.clip(
            self._right_dxl_to_robot_joint_fn(right_arm_q),
            a_min=self.RIGHT_JOINT_LIMIT_LOW,
            a_max=self.RIGHT_JOINT_LIMIT_HIGH,
        )
        return {
            "left": left_arm_q,
            "right": right_arm_q,
        }
