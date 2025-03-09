from functools import partial
from typing import Sequence, Union, Optional
from threading import Event, Thread

import numpy as np
from brs_ctrl.joylo.joylo_arms.dxl.position_control import DXLPositionControlDriver


class DXLPositionController:
    def __init__(
        self,
        ids: Union[int, Sequence[int]],
        *,
        gripper_ids: Optional[Union[int, Sequence[int]]] = None,
        port: str,
        baudrate: int = 3000000,
        slave_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        master_motor_ids: Optional[Union[int, Sequence[int]]] = None,
        multithread_read_joints: bool = False,
    ):
        if slave_motor_ids is not None:
            assert (
                master_motor_ids is not None
            ), "master_motor_ids must be provided when slave_motor_ids is provided"
            if isinstance(slave_motor_ids, int):
                slave_motor_ids = [slave_motor_ids]
            if isinstance(master_motor_ids, int):
                master_motor_ids = [master_motor_ids]
            assert len(slave_motor_ids) == len(
                master_motor_ids
            ), "slave_motor_ids and master_motor_ids must have the same length"
            assert set(slave_motor_ids).issubset(
                ids
            ), "slave_motor_ids must be a subset of ids"
            assert set(master_motor_ids).issubset(
                ids
            ), "master_motor_ids must be a subset of ids"
            assert (
                len(set(slave_motor_ids).intersection(master_motor_ids)) == 0
            ), "slave_motor_ids and master_motor_ids must be disjoint"
            self._effective_dofs = len(ids) - len(slave_motor_ids)
            self._slave_motor_idxs = np.array(
                [idx for idx, i in enumerate(ids) if i in slave_motor_ids],
                dtype=np.int32,
            )
            self._master_motor_idxs = np.array(
                [idx for idx, i in enumerate(ids) if i in master_motor_ids],
                dtype=np.int32,
            )
        else:
            self._effective_dofs = len(ids)
            self._slave_motor_idxs = np.array([], dtype=np.int32)
            self._master_motor_idxs = np.array([], dtype=np.int32)
        self._reading_motor_idxs = np.array(
            [idx for idx in range(len(ids)) if idx not in self._slave_motor_idxs],
            dtype=np.int32,
        )

        self._driver = DXLPositionControlDriver(
            ids=ids,
            gripper_ids=gripper_ids,
            port=port,
            baudrate=baudrate,
            operating_mode="position",
            multithread_read_joints=multithread_read_joints,
        )
        self._driver.set_normal_motors(ids)

        # dual-motor calibration
        if slave_motor_ids is not None:
            self._driver.set_reverse_motors(slave_motor_ids)
            # enabling torque in position control mode lets present position be in one turn
            self._driver.set_torque_mode(True)
            self._driver.set_torque_mode(False)
            # clear homing offset for all motors
            self._driver.set_homing_offset(ids, 0)
            all_positions = self._driver.get_joints(block=True)[0]
            master_position = all_positions[self._master_motor_idxs]
            slave_position = all_positions[self._slave_motor_idxs]
            homing_offset = slave_position - master_position
            self._driver.set_homing_offset(slave_motor_ids, homing_offset)
        curr_positions, curr_velocities, _, _ = self._driver.get_joints(block=True)
        self._curr_positions = curr_positions[self._reading_motor_idxs]
        self._curr_velocities = curr_velocities[self._reading_motor_idxs]

        self._stop_thread = None
        self._control_thread = None

        self._goal = None

    def _control(self, position_limit_high, position_limit_low):
        assert self._goal is not None, "Goal not set"
        while not self._stop_thread.is_set():
            curr_positions, curr_velocities = self._driver.get_joints()[:2]
            curr_positions = curr_positions[self._reading_motor_idxs]
            curr_velocities = curr_velocities[self._reading_motor_idxs]
            assert np.all(curr_positions <= position_limit_high) and np.all(
                curr_positions >= position_limit_low
            ), f"Current position out of limit. {curr_positions} not in [{position_limit_low}, {position_limit_high}]"
            self._curr_positions = curr_positions
            self._curr_velocities = curr_velocities

            ctrl_cmd = np.zeros((len(self._driver.motor_ids),), dtype=np.int32)
            ctrl_cmd[self._reading_motor_idxs] = self._goal.copy()
            ctrl_cmd[self._slave_motor_idxs] = ctrl_cmd[self._master_motor_idxs]

            self._driver.set_joints(ctrl_cmd)

    def set_new_goal(self, goal: np.ndarray):
        assert len(goal) == self._effective_dofs
        assert goal.dtype == np.int32, "Goal must be DXL raw position values."
        self._goal = goal

    def start_control(self, position_limit_high=None, position_limit_low=None):
        if position_limit_high is None:
            position_limit_high = np.array([2147483647] * self._effective_dofs)
        if position_limit_low is None:
            position_limit_low = np.array([-2147483648] * self._effective_dofs)
        assert (
            len(position_limit_high) == len(position_limit_low) == self._effective_dofs
        )
        self._driver.set_torque_mode(True)
        self._stop_thread = Event()
        self._control_thread = Thread(
            target=partial(self._control, position_limit_high, position_limit_low)
        )
        self._control_thread.daemon = True
        self._control_thread.start()

    def close(self):
        if self._stop_thread is not None:
            self._stop_thread.set()
        if self._control_thread is not None:
            self._control_thread.join()
        self._driver.set_torque_mode(False)
        self._driver.close()

    def __del__(self):
        self.close()

    @property
    def curr_positions_and_velocities(self):
        curr_positions, curr_velocities = self._driver.get_joints()[:2]
        return (
            curr_positions[self._reading_motor_idxs],
            curr_velocities[self._reading_motor_idxs],
        )

    @property
    def curr_gripper_positions_and_velocities(self):
        return self._driver.get_joints()[2:]

    def start_read_thread(self):
        self._driver.start_read_thread()
