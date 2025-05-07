from typing import Protocol, Sequence, Union, Literal, Tuple, Optional
import time
from threading import Lock, Event, Thread

import numpy as np
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite

from brs_ctrl.joylo.joylo_arms.dxl.constants import (
    operating_modes,
    PresentPosition,
    PresentVelocity,
    Torque,
    Comm,
    OperatingMode,
    DriveMode,
    HomingOffset,
)


class DXLBaseDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the current joint angles in radians.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                A tuple of joint angles, joint velocities, gripper angles, and gripper velocities (if grippers exist).
        """
        ...

    def close(self):
        """Close the driver."""


class DXLBaseDriver(DXLBaseDriverProtocol):
    read_hz: int = 500

    def __init__(
        self,
        ids: Union[int, Sequence[int]],
        port: str,
        baudrate: int = 3000000,
        multithread_read_joints: bool = False,
        *,
        operating_mode: Literal[
            "current",
            "velocity",
            "position",
            "extended_position",
            "current_based_position",
            "pwm",
        ],
        gripper_ids: Optional[Union[int, Sequence[int]]] = None,
    ):
        if isinstance(ids, int):
            ids = [ids]
        assert all(isinstance(i, int) for i in ids)
        if gripper_ids is not None:
            if isinstance(gripper_ids, int):
                gripper_ids = [gripper_ids]
            assert all(isinstance(i, int) for i in gripper_ids)
            assert set(gripper_ids).isdisjoint(
                set(ids)
            ), f"gripper_ids must be disjoint from ids; got {gripper_ids} and {ids}"

        self._ids = ids
        self._gripper_ids = gripper_ids
        self._joint_positions = None
        self._joint_velocities = None
        self._gripper_positions = None
        self._gripper_velocities = None
        self._lock = Lock()

        # Initialize the port handler, packet handler, and group sync read/write
        self._port_handler = PortHandler(port)
        self._packet_handler = PacketHandler(2.0)
        self._pos_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            PresentPosition.ADDR.value,
            PresentPosition.LEN.value,
        )
        self._vel_read = GroupSyncRead(
            self._port_handler,
            self._packet_handler,
            PresentVelocity.ADDR.value,
            PresentVelocity.LEN.value,
        )
        assert operating_mode in operating_modes
        self._group_sync_write = self._create_group_sync_write(operating_mode)

        # Open the port and set the baudrate
        if not self._port_handler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._port_handler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        # change the operating mode
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler,
                    dxl_id,
                    OperatingMode.ADDR.value,
                    operating_modes[operating_mode].value,
                )
                if dxl_comm_result != Comm.SUCCESS.value:
                    raise ValueError(
                        self._packet_handler.getTxRxResult(dxl_comm_result)
                    )
                elif dxl_error != 0:
                    raise ValueError(self._packet_handler.getRxPacketError(dxl_error))

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._pos_read.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
            if not self._vel_read.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )
        if self._gripper_ids is not None:
            for dxl_id in self._gripper_ids:
                if not self._pos_read.addParam(dxl_id):
                    raise RuntimeError(
                        f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                    )
                if not self._vel_read.addParam(dxl_id):
                    raise RuntimeError(
                        f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                    )

        # Disable torque for each Dynamixel servo
        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

        self._multithread_read_joints = multithread_read_joints

    def start_read_thread(self):
        if not self._multithread_read_joints:
            return
        self._stop_thread = Event()
        self._start_reading_thread()

    @property
    def motor_ids(self):
        return self._ids

    @property
    def gripper_ids(self):
        return self._gripper_ids

    def set_normal_motors(self, r_ids: Sequence[int]):
        assert set(r_ids).issubset(set(self._ids))

        with self._lock:
            for dxl_id in r_ids:
                dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler,
                    dxl_id,
                    DriveMode.ADDR.value,
                    DriveMode.NORMAL.value,
                )
                if dxl_comm_result != Comm.SUCCESS.value:
                    raise ValueError(
                        self._packet_handler.getTxRxResult(dxl_comm_result)
                    )
                elif dxl_error != 0:
                    raise ValueError(self._packet_handler.getRxPacketError(dxl_error))

    def set_reverse_motors(self, r_ids: Sequence[int]):
        assert set(r_ids).issubset(set(self._ids))

        with self._lock:
            for dxl_id in r_ids:
                dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler,
                    dxl_id,
                    DriveMode.ADDR.value,
                    DriveMode.REVERSE.value,
                )
                if dxl_comm_result != Comm.SUCCESS.value:
                    raise ValueError(
                        self._packet_handler.getTxRxResult(dxl_comm_result)
                    )
                elif dxl_error != 0:
                    raise ValueError(self._packet_handler.getRxPacketError(dxl_error))

    def _create_group_sync_write(self, operating_mode: str) -> GroupSyncWrite:
        """Create a GroupSyncWrite object for the given operating mode."""
        ...

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        with self._lock:
            self._set_joints(joint_angles)

    def _set_joints(self, joint_angles: Sequence[float]):
        """set joints according to different controller type"""
        raise NotImplementedError

    def set_torque_mode(self, enable: bool):
        torque_value = Torque.ENABLE.value if enable else Torque.DISABLE.value
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packet_handler.write1ByteTxRx(
                    self._port_handler, dxl_id, Torque.ADDR.value, torque_value
                )
                if dxl_comm_result != Comm.SUCCESS.value or dxl_error != 0:
                    error_msg = (
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )
                    if dxl_comm_result != 0:
                        error_msg += self._packet_handler.getTxRxResult(dxl_comm_result)
                    elif dxl_error != 0:
                        error_msg += self._packet_handler.getRxPacketError(dxl_error)
                    raise RuntimeError(error_msg)

        self._torque_enabled = enable

    def set_homing_offset(
        self, dxl_ids: Union[int, Sequence[int]], values: Union[int, Sequence[int]]
    ):
        if isinstance(dxl_ids, int):
            dxl_ids = [dxl_ids]
        if isinstance(values, int):
            values = [values] * len(dxl_ids)
        assert len(dxl_ids) == len(
            values
        ), f"dxl_ids and values must have the same length, got {len(dxl_ids)} and {len(values)}"
        with self._lock:
            for dxl_id, value in zip(dxl_ids, values):
                dxl_comm_result, dxl_error = self._packet_handler.write4ByteTxRx(
                    self._port_handler, dxl_id, HomingOffset.ADDR.value, value
                )
                if dxl_comm_result != Comm.SUCCESS.value or dxl_error != 0:
                    error_msg = (
                        f"Failed to set homing offset for Dynamixel with ID {dxl_id}"
                    )
                    if dxl_comm_result != 0:
                        error_msg += self._packet_handler.getTxRxResult(dxl_comm_result)
                    elif dxl_error != 0:
                        error_msg += self._packet_handler.getRxPacketError(dxl_error)
                    raise RuntimeError(error_msg)

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joints)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joints(self):
        # Continuously read joint angles and update the joint_angles array
        while not self._stop_thread.is_set():
            with self._lock:
                self._pos_read.txRxPacket()
                self._vel_read.txRxPacket()

                _joint_angles = np.zeros(len(self._ids), dtype=int)
                read_error_occur = False
                for i, dxl_id in enumerate(self._ids):
                    if self._pos_read.isAvailable(
                        dxl_id, PresentPosition.ADDR.value, PresentPosition.LEN.value
                    ):
                        angle = self._pos_read.getData(
                            dxl_id,
                            PresentPosition.ADDR.value,
                            PresentPosition.LEN.value,
                        )
                        angle = np.int32(np.uint32(angle))
                        _joint_angles[i] = angle
                    else:
                        read_error_occur = True
                        break
                if not read_error_occur:
                    self._joint_positions = _joint_angles
                _joint_velocities = np.zeros(len(self._ids), dtype=int)
                read_error_occur = False
                for i, dxl_id in enumerate(self._ids):
                    if self._vel_read.isAvailable(
                        dxl_id, PresentVelocity.ADDR.value, PresentVelocity.LEN.value
                    ):
                        velocity = self._vel_read.getData(
                            dxl_id,
                            PresentVelocity.ADDR.value,
                            PresentVelocity.LEN.value,
                        )
                        velocity = np.int32(np.uint32(velocity))
                        _joint_velocities[i] = velocity
                    else:
                        read_error_occur = True
                        break
                if not read_error_occur:
                    self._joint_velocities = _joint_velocities

                if self._gripper_ids is not None:
                    _gripper_angles = np.zeros(
                        len(
                            self._gripper_ids,
                        ),
                        dtype=int,
                    )
                    read_error_occur = False
                    for i, dxl_id in enumerate(self._gripper_ids):
                        if self._pos_read.isAvailable(
                            dxl_id,
                            PresentPosition.ADDR.value,
                            PresentPosition.LEN.value,
                        ):
                            angle = self._pos_read.getData(
                                dxl_id,
                                PresentPosition.ADDR.value,
                                PresentPosition.LEN.value,
                            )
                            angle = np.int32(np.uint32(angle))
                            _gripper_angles[i] = angle
                        else:
                            read_error_occur = True
                            break
                    if not read_error_occur:
                        self._gripper_positions = _gripper_angles

                    _gripper_velocities = np.zeros(len(self._gripper_ids), dtype=int)
                    read_error_occur = False
                    for i, dxl_id in enumerate(self._gripper_ids):
                        if self._vel_read.isAvailable(
                            dxl_id,
                            PresentVelocity.ADDR.value,
                            PresentVelocity.LEN.value,
                        ):
                            velocity = self._vel_read.getData(
                                dxl_id,
                                PresentVelocity.ADDR.value,
                                PresentVelocity.LEN.value,
                            )
                            velocity = np.int32(np.uint32(velocity))
                            _gripper_velocities[i] = velocity
                        else:
                            read_error_occur = True
                            break
                    if not read_error_occur:
                        self._gripper_velocities = _gripper_velocities

    def get_joints(
        self, block=False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if (
            self._multithread_read_joints
            and getattr(self, "_reading_thread", None) is None
            and not block
        ):
            self.start_read_thread()
        if self._multithread_read_joints and not block:
            while self._joint_positions is None:
                time.sleep(0.1)

        else:

            with self._lock:
                _joint_angles = np.zeros(len(self._ids), dtype=int)
                dxl_comm_result = self._pos_read.txRxPacket()
                if dxl_comm_result != Comm.SUCCESS.value:
                    print(self._packet_handler.getTxRxResult(dxl_comm_result))
                read_error_occur = False
                for i, dxl_id in enumerate(self._ids):
                    if self._pos_read.isAvailable(
                        dxl_id, PresentPosition.ADDR.value, PresentPosition.LEN.value
                    ):
                        angle = self._pos_read.getData(
                            dxl_id,
                            PresentPosition.ADDR.value,
                            PresentPosition.LEN.value,
                        )
                        angle = np.int32(np.uint32(angle))
                        _joint_angles[i] = angle
                    else:
                        read_error_occur = True
                        break
                if not read_error_occur:
                    self._joint_positions = _joint_angles
                _joint_velocities = np.zeros(len(self._ids), dtype=int)
                dxl_comm_result = self._vel_read.txRxPacket()
                if dxl_comm_result != Comm.SUCCESS.value:
                    print(self._packet_handler.getTxRxResult(dxl_comm_result))
                read_error_occur = False
                for i, dxl_id in enumerate(self._ids):
                    if self._vel_read.isAvailable(
                        dxl_id, PresentVelocity.ADDR.value, PresentVelocity.LEN.value
                    ):
                        velocity = self._vel_read.getData(
                            dxl_id,
                            PresentVelocity.ADDR.value,
                            PresentVelocity.LEN.value,
                        )
                        velocity = np.int32(np.uint32(velocity))
                        _joint_velocities[i] = velocity
                    else:
                        read_error_occur = True
                        break
                if not read_error_occur:
                    self._joint_velocities = _joint_velocities

                if self._gripper_ids is not None:
                    _gripper_angles = np.zeros(
                        len(
                            self._gripper_ids,
                        ),
                        dtype=int,
                    )
                    read_error_occur = False
                    for i, dxl_id in enumerate(self._gripper_ids):
                        if self._pos_read.isAvailable(
                            dxl_id,
                            PresentPosition.ADDR.value,
                            PresentPosition.LEN.value,
                        ):
                            angle = self._pos_read.getData(
                                dxl_id,
                                PresentPosition.ADDR.value,
                                PresentPosition.LEN.value,
                            )
                            angle = np.int32(np.uint32(angle))
                            _gripper_angles[i] = angle
                        else:
                            read_error_occur = True
                            break
                    if not read_error_occur:
                        self._gripper_positions = _gripper_angles

                    _gripper_velocities = np.zeros(len(self._gripper_ids), dtype=int)
                    read_error_occur = False
                    for i, dxl_id in enumerate(self._gripper_ids):
                        if self._vel_read.isAvailable(
                            dxl_id,
                            PresentVelocity.ADDR.value,
                            PresentVelocity.LEN.value,
                        ):
                            velocity = self._vel_read.getData(
                                dxl_id,
                                PresentVelocity.ADDR.value,
                                PresentVelocity.LEN.value,
                            )
                            velocity = np.int32(np.uint32(velocity))
                            _gripper_velocities[i] = velocity
                        else:
                            read_error_occur = True
                            break
                    if not read_error_occur:
                        self._gripper_velocities = _gripper_velocities
        rtn = (
            (
                self._joint_positions.copy(),
                self._joint_velocities.copy(),
                self._gripper_positions.copy(),
                self._gripper_velocities.copy(),
            )
            if self._gripper_ids is not None
            else (
                self._joint_positions.copy(),
                self._joint_velocities.copy(),
                None,
                None,
            )
        )
        return rtn

    def close(self):
        self.set_homing_offset(self._ids, 0)
        if self._multithread_read_joints:
            self._stop_thread.set()
            self._reading_thread.join()
        self._pos_read.clearParam()
        self._vel_read.clearParam()
        self._port_handler.closePort()
