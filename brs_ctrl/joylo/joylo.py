from typing import Union

import numpy as np

from brs_ctrl.joylo.joycon import R1JoyConInterface
from brs_ctrl.joylo.joylo_arms import (
    JoyLoArmPositionController,
    JoyLoArmImpedanceController,
)


class JoyLoController:
    def __init__(
        self,
        *,
        joycon: R1JoyConInterface,
        joylo_arms: Union[JoyLoArmPositionController, JoyLoArmImpedanceController],
    ):
        self._joycon = joycon
        self._joylo_arms = joylo_arms

    def act(self, curr_torso_q: np.ndarray):
        joycon_action = self._joycon.act(curr_torso_q)
        arm_q = self.q
        joycon_action["arm_cmd"] = arm_q
        return joycon_action

    @property
    def q(self):
        return self._joylo_arms.q

    def set_new_arm_goal(self, *args, **kwargs):
        self._joylo_arms.set_new_goal(*args, **kwargs)

    def start_arm_control(self):
        self._joylo_arms.start_control()

    def close(self):
        self._joycon.close()
        self._joylo_arms.close()
