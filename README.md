# BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities
<div align="center">

[Yunfan Jiang](https://yunfanj.com/),
[Ruohan Zhang](https://ai.stanford.edu/~zharu/),
[Josiah Wong](https://jdw.ong/),
[Chen Wang](https://www.chenwangjeremy.net/),
[Yanjie Ze](https://yanjieze.com/),
[Hang Yin](https://hang-yin.github.io/),
[Cem Gokmen](https://www.cemgokmen.com/),
[Shuran Song](https://shurans.github.io/),
[Jiajun Wu](https://jiajunwu.com/),
[Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li)

<img src="media/SUSig-red.png" width=200>

[[Website]](https://behavior-robot-suite.github.io/)
[[arXiv]]()
[[PDF]](https://behavior-robot-suite.github.io/assets/pdf/brs_paper.pdf)
[[Doc]](https://behavior-robot-suite.github.io/docs/)
[[Algorithm Code]](https://github.com/behavior-robot-suite/brs-algo)
[[Assembly Guide]](https://behavior-robot-suite.github.io/docs/sections/joylo/step_by_step_assembly_guidance.html)


[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://github.com/behavior-robot-suite/brs-ctrl)
[<img src="https://img.shields.io/badge/Doc-Passing-green.svg"/>](https://behavior-robot-suite.github.io/docs/)
[![GitHub license](https://img.shields.io/github/license/behavior-robot-suite/brs-ctrl)](https://github.com/behavior-robot-suite/brs-ctrl/blob/main/LICENSE)

![](media/pull.gif)
______________________________________________________________________
</div>

We introduce the **BEHAVIOR Robot Suite** (BRS), a comprehensive framework for learning whole-body manipulation to tackle diverse real-world household tasks. BRS addresses both hardware and learning challenges through two key innovations: **JoyLo** and [WB-VIMA](https://github.com/behavior-robot-suite/brs-algo).

JoyLo provides a general, cost-effective approach to whole-body teleoperation by integrating multifunctional joystick controllers mounted on the ends of two 3D-printed arms.  The mounting arms serve as scaled-down kinematic twins of the robot’s arms, enabling precise bilateral teleoperation. JoyLo also inherits key advantages of puppeteering devices, including intuitive operation, reduced singularities, and enhanced stability. By grasping the JoyCon controllers attached to the kinematic-twin arms, users can operate the arms, grippers, torso, and mobile base in unison. This significantly accelerates data collection by allowing users to perform bimanual coordination tasks, navigate safely and accurately, and guide the end-effectors to effectively reach various locations in 3D space.

![](media/joylo.gif)

## Getting Started

> [!TIP]
> 🚀 Check out the [doc](https://behavior-robot-suite.github.io/docs/sections/brs_ctrl/overview.html) for detailed installation and usage instructions!

To control the Galaxea R1 robot, simply do

```Python
from brs_ctrl.robot_interface import R1Interface
from brs_ctrl.robot_interface.grippers import GalaxeaR1Gripper

# Initialize the robot interface with grippers
robot = R1Interface(
    left_gripper=GalaxeaR1Gripper(
        left_or_right="left",
        gripper_close_stroke=0.0,
        gripper_open_stroke=100.0,
    ),
    right_gripper=GalaxeaR1Gripper(
        left_or_right="right",
        gripper_close_stroke=0.0,
        gripper_open_stroke=100.0,
    ),
)

# Control the robot
robot.control(
    arm_cmd={
        "left": left_arm_action,  # np (6,)
        "right": right_arm_action,  # np (6,)
    },
    gripper_cmd={
        "left": left_gripper_action,  # float between 0.0 and 1.0
        "right": right_gripper_action,  # float between 0.0 and 1.0
    },
    torso_cmd=torso_action,  # np (4,)
    base_cmd=mobile_base_action,  # np (3,)
)
```

To run real-robot JoyLo control, simply run 

```bash
python3 scripts/joylo/real_joylo.py
```

To run data collection, simply run 

```bash
python3 scripts/data_collection/start_data_collection.py
```

## Check out Our Paper
Our paper is posted on [arXiv](). If you find our work useful, please consider citing us! 

```bibtex
@article{jiang2025brs,
  title = {BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities},
  author = {Yunfan Jiang and Ruohan Zhang and Josiah Wong and Chen Wang and Yanjie Ze and Hang Yin and Cem Gokmen and Shuran Song and Jiajun Wu and Li Fei-Fei},
  year = {2025}
}
```

## License
This codebase is released under the [MIT License](LICENSE).