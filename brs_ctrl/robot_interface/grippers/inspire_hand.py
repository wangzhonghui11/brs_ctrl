import time
import math
from typing import Any, Union
from threading import Lock, Event, Thread

import serial
import numpy as np
from sensor_msgs.msg import JointState

from brs_ctrl.robot_interface.grippers.base import BaseGripper


class _InspireHand:
    def __init__(self, hand_id, port, baudrate=115200):
        self.hand_id = hand_id
        self.ser = serial.Serial(port, baudrate)
        self.ser.isOpen()
        self.error = [0] * 6
        self.status = [2] * 6
        self.force_limits = [500] * 6
        self.speed_limits = [1000] * 6
        self.actpos = [0] * 6
        self.actangle = [0] * 6
        self.actforce = [0] * 6

    def checknum(self, data, leng):
        result = 0
        for i in range(2, leng):
            result += data[i]
        return result & 0xFF

    def data2bytes(self, data):
        rdata = [0xFF] * 2
        if data != -1:
            rdata[0] = data & 0xFF
            rdata[1] = (data >> 8) & (0xFF)
        return rdata

    def num2str(self, num):
        str = hex(num)
        str = str[2:4]
        if len(str) == 1:
            str = "0" + str
        return bytes.fromhex(str)

    def prepare_data_frame(self, data_num):
        data_bytes = [0] * (data_num + 5)
        data_bytes[0] = 0xEB
        data_bytes[1] = 0x90
        data_bytes[2] = self.hand_id
        data_bytes[3] = data_num
        return data_bytes

    def update_hand_state(self):
        self.get_error()
        self.get_status()
        self.get_actpos()
        self.get_actangle()
        self.get_actforce()

    def send_data(self, data):
        """
        Send data to the hand and return the response
        """
        putdata = b""
        for i in range(len(data)):
            putdata = putdata + self.num2str(data[i])
        self.ser.write(putdata)
        # return message
        if data[4] == 0x11:  # get
            return_message_len = data[7] + 8
        elif data[4] == 0x12:  # set
            return_message_len = 9
        return self.ser.read(return_message_len)

    def parse_returned_message(self, dataframe, dtype="byte"):
        data_num = dataframe[3]
        if dtype == "byte":
            data = [0] * (data_num - 3)
            for i in range(data_num - 3):
                data[i] = dataframe[i + 7]
        elif dtype == "short":
            data = [0] * ((data_num - 3) // 2)
            for i in range((data_num - 3) // 2):
                data[i] = int.from_bytes(
                    dataframe[i * 2 + 7 : i * 2 + 9], byteorder="little", signed=True
                )
        else:
            raise ValueError("dtype should be byte or short")
        return data

    def reset_param(self):
        """Reset parameters to factory setting"""
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x12
        data_bytes[5] = 0xEE
        data_bytes[6] = 0x03
        data_bytes[7] = 0x01
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        return self.send_data(data_bytes)

    def clear_error(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x12
        data_bytes[5] = 0xEC
        data_bytes[6] = 0x03
        data_bytes[7] = 0x01
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        return self.send_data(data_bytes)

    def calibrate_force(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x12
        data_bytes[5] = 0xF1
        data_bytes[6] = 0x03
        data_bytes[7] = 0x01
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        self.send_data(data_bytes)
        print("Calibrating force...")
        time.sleep(20)

    def get_error(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x11
        data_bytes[5] = 0x46
        data_bytes[6] = 0x06
        data_bytes[7] = 0x06
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        message = self.send_data(data_bytes)

        self.error = self.parse_returned_message(message)

    def get_status(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x11
        data_bytes[5] = 0x4C
        data_bytes[6] = 0x06
        data_bytes[7] = 0x06
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        message = self.send_data(data_bytes)

        self.status = self.parse_returned_message(message)

    def get_actpos(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x11
        data_bytes[5] = 0xFE
        data_bytes[6] = 0x05
        data_bytes[7] = 0x0C
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        message = self.send_data(data_bytes)

        self.actpos = self.parse_returned_message(message, dtype="short")

    def get_actangle(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x11
        data_bytes[5] = 0x0A
        data_bytes[6] = 0x06
        data_bytes[7] = 0x0C
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        message = self.send_data(data_bytes)

        self.actangle = self.parse_returned_message(message, dtype="short")

    def get_actforce(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x11
        data_bytes[5] = 0x2E
        data_bytes[6] = 0x06
        data_bytes[7] = 0x0C
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        message = self.send_data(data_bytes)

        self.actforce = self.parse_returned_message(message, dtype="short")

    def set_pos(self, pos):
        """
        Per the manual, this is not recommended. Use setangle instead.
        """
        for i in range(len(pos)):
            if abs(self.actforce[i]) >= self.force_limits[i]:
                pos[i] = -1
                # alternatively, we can add a small backward value to the position
                pos[i] = self.actpos[i] + int(math.copysign(20, self.actforce[i]))
            elif pos[i] < -1 or pos[i] > 2000:
                print(f"pos {i} out of range: -1-2000")
                return

        datanum = 0x0F
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x12
        data_bytes[5] = 0xC2
        data_bytes[6] = 0x05
        data_bytes[7] = self.data2bytes(pos[0])[0]
        data_bytes[8] = self.data2bytes(pos[0])[1]
        data_bytes[9] = self.data2bytes(pos[1])[0]
        data_bytes[10] = self.data2bytes(pos[1])[1]
        data_bytes[11] = self.data2bytes(pos[2])[0]
        data_bytes[12] = self.data2bytes(pos[2])[1]
        data_bytes[13] = self.data2bytes(pos[3])[0]
        data_bytes[14] = self.data2bytes(pos[3])[1]
        data_bytes[15] = self.data2bytes(pos[4])[0]
        data_bytes[16] = self.data2bytes(pos[4])[1]
        data_bytes[17] = self.data2bytes(pos[5])[0]
        data_bytes[18] = self.data2bytes(pos[5])[1]
        data_bytes[19] = self.checknum(data_bytes, datanum + 4)
        return self.send_data(data_bytes)

    def set_angle(self, angle):
        for i in range(len(angle)):
            if abs(self.actforce[i]) >= self.force_limits[i]:
                angle[i] = -1
                # alternatively, we can add a small backward value to the angle
                angle[i] = self.actangle[i] + int(math.copysign(10, self.actforce[i]))
            elif angle[i] < -1 or angle[i] > 1000:
                print(f"angle {i} out of range: -1-1000")
                return

        datanum = 0x0F
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x12
        data_bytes[5] = 0xCE
        data_bytes[6] = 0x05
        data_bytes[7] = self.data2bytes(angle[0])[0]
        data_bytes[8] = self.data2bytes(angle[0])[1]
        data_bytes[9] = self.data2bytes(angle[1])[0]
        data_bytes[10] = self.data2bytes(angle[1])[1]
        data_bytes[11] = self.data2bytes(angle[2])[0]
        data_bytes[12] = self.data2bytes(angle[2])[1]
        data_bytes[13] = self.data2bytes(angle[3])[0]
        data_bytes[14] = self.data2bytes(angle[3])[1]
        data_bytes[15] = self.data2bytes(angle[4])[0]
        data_bytes[16] = self.data2bytes(angle[4])[1]
        data_bytes[17] = self.data2bytes(angle[5])[0]
        data_bytes[18] = self.data2bytes(angle[5])[1]
        data_bytes[19] = self.checknum(data_bytes, datanum + 4)
        return self.send_data(data_bytes)

    def set_force(self, force=None):
        if force is None:
            force = self.force_limits
        else:
            self.force_limits = force
        for i, f in enumerate(force):
            if f < 0 or f > 1000:
                print(f"force {i} out of range: 0-1000")
                return

        datanum = 0x0F
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x12
        data_bytes[5] = 0xDA
        data_bytes[6] = 0x05
        data_bytes[7] = self.data2bytes(force[0])[0]
        data_bytes[8] = self.data2bytes(force[0])[1]
        data_bytes[9] = self.data2bytes(force[1])[0]
        data_bytes[10] = self.data2bytes(force[1])[1]
        data_bytes[11] = self.data2bytes(force[2])[0]
        data_bytes[12] = self.data2bytes(force[2])[1]
        data_bytes[13] = self.data2bytes(force[3])[0]
        data_bytes[14] = self.data2bytes(force[3])[1]
        data_bytes[15] = self.data2bytes(force[4])[0]
        data_bytes[16] = self.data2bytes(force[4])[1]
        data_bytes[17] = self.data2bytes(force[5])[0]
        data_bytes[18] = self.data2bytes(force[5])[1]
        data_bytes[19] = self.checknum(data_bytes, datanum + 4)
        return self.send_data(data_bytes)

    def set_speed(self, speed=None):
        if speed is None:
            speed = self.speed_limits
        else:
            self.speed_limits = speed
        for i, s in enumerate(speed):
            if s < 0 or s > 1000:
                print(f"speed {i} out of range: 0-1000")
                return

        datanum = 0x0F
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x12
        data_bytes[5] = 0xF2
        data_bytes[6] = 0x05
        data_bytes[7] = self.data2bytes(speed[0])[0]
        data_bytes[8] = self.data2bytes(speed[0])[1]
        data_bytes[9] = self.data2bytes(speed[1])[0]
        data_bytes[10] = self.data2bytes(speed[1])[1]
        data_bytes[11] = self.data2bytes(speed[2])[0]
        data_bytes[12] = self.data2bytes(speed[2])[1]
        data_bytes[13] = self.data2bytes(speed[3])[0]
        data_bytes[14] = self.data2bytes(speed[3])[1]
        data_bytes[15] = self.data2bytes(speed[4])[0]
        data_bytes[16] = self.data2bytes(speed[4])[1]
        data_bytes[17] = self.data2bytes(speed[5])[0]
        data_bytes[18] = self.data2bytes(speed[5])[1]
        data_bytes[19] = self.checknum(data_bytes, datanum + 4)
        return self.send_data(data_bytes)

    def get_setforce(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x11
        data_bytes[5] = 0xDA
        data_bytes[6] = 0x05
        data_bytes[7] = 0x0C
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        message = self.send_data(data_bytes)

        setforce = self.parse_returned_message(message, dtype="short")
        return setforce

    def get_setspeed(self):
        datanum = 0x04
        data_bytes = self.prepare_data_frame(datanum)
        data_bytes[4] = 0x11
        data_bytes[5] = 0xF2
        data_bytes[6] = 0x05
        data_bytes[7] = 0x0C
        data_bytes[8] = self.checknum(data_bytes, datanum + 4)
        message = self.send_data(data_bytes)

        setspeed = self.parse_returned_message(message, dtype="short")
        return setspeed


class InspireHand(BaseGripper):
    def __init__(
        self,
        hand_id: int,
        port: str,
        baudrate: int = 115200,
        close_pose: np.ndarray = np.array([500, 500, 500, 500, 500, 80]),
        open_pose: np.ndarray = np.array([1000, 1000, 1000, 1000, 1000, 1000]),
    ):
        self._hand = _InspireHand(hand_id, port, baudrate)
        self._close_pose = close_pose
        self._open_pose = open_pose

        self._lock = Lock()
        self._target_pose = None

    def init_hook(self):
        self._hand.set_force()
        self._hand.set_speed()

        self._stop_thread = Event()
        self._control_thread = Thread(target=self._control_loop)
        self._control_thread.daemon = True
        self._control_thread.start()

    def act(self, action: Union[float, np.ndarray]):
        if isinstance(action, np.ndarray):
            cmd = action * 1000
        else:
            cmd = self._close_pose + (1 - action) * (self._open_pose - self._close_pose)
        self._target_pose = cmd.astype(np.int64)

    def get_state(self, data: JointState) -> Any:
        return {
            "hand_pose": np.array([self._hand.actangle]),
            "hand_effort": np.array([self._hand.actforce]),
        }

    def _control_loop(self):
        while not self._stop_thread.is_set():
            with self._lock:
                if self._target_pose is not None:
                    self._hand.update_hand_state()
                    self._hand.set_angle(self._target_pose)

    def close(self):
        self._stop_thread.set()
        self._control_thread.join()
        self._hand.ser.close()
