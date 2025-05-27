import os
import time
import multiprocessing as mp
import numpy as np
import pybullet as pb
import cv2
import ctypes
from brs_ctrl.asset_root import ASSET_ROOT
from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1JoyConInterface
import pybullet_data
from urdf_models import models_data
import random

# 连接物理引擎
pb_client = pb.connect(pb.GUI)
pb.setGravity(0, 0, -9.8)

# 加载机器人
robot = pb.loadURDF(
    os.path.join(ASSET_ROOT, "robot/r1_pro/r1_pro.urdf"),
    [0, 0, 0],
    useFixedBase=True,
)

# 定义关键参数
end_effector_link = 20  # 假设末端执行器是左臂的第20号链接
num_joints = pb.getNumJoints(robot)
print(f"机器人共有 {num_joints} 个关节")


def print_joint_info(robot):
    """
    打印机器人所有关节的详细信息（兼容不同PyBullet版本）
    :param robot: 机器人URDF的加载ID
    """
    print("\n" + "=" * 50)
    print(f"{'机器人关节信息':^50}")
    print("=" * 50)

    for i in range(pb.getNumJoints(robot)):
        joint_info = pb.getJointInfo(robot, i)

        # 安全获取元组元素
        joint_name = joint_info[1].decode('utf-8') if len(joint_info) > 1 else "Unknown"
        joint_type = {
            0: "旋转关节(revolute)",
            1: "滑动关节(prismatic)",
            2: "球关节(spherical)",
            3: "平面关节(planar)",
            4: "固定关节(fixed)"
        }.get(joint_info[2], "未知类型") if len(joint_info) > 2 else "未知类型"

        # 动态参数（检查索引是否存在）
        damping = joint_info[6] if len(joint_info) > 6 else 0.0
        friction = joint_info[7] if len(joint_info) > 7 else 0.0

        # 运动范围（检查索引是否存在）
        lower_limit = joint_info[8] if len(joint_info) > 8 else 0.0
        upper_limit = joint_info[9] if len(joint_info) > 9 else 0.0

        # 极限参数（检查索引是否存在）
        max_force = joint_info[10] if len(joint_info) > 10 else 0.0
        max_velocity = joint_info[11] if len(joint_info) > 11 else 0.0

        # 链接关系（检查索引是否存在）
        parent_idx = joint_info[16] if len(joint_info) > 16 else -1
        child_idx = joint_info[17] if len(joint_info) > 17 else -1

        # 局部坐标系（检查索引是否存在）
        pos = np.round(joint_info[14], 3) if len(joint_info) > 14 else [0, 0, 0]
        orn = np.round(joint_info[15], 3) if len(joint_info) > 15 else [0, 0, 0, 1]

        # 格式化输出
        print(f"\n◆ 关节索引 [{i}] {joint_name}")
        print(f"├─ 类型: {joint_type}")
        print(f"├─ 动态参数: 阻尼={damping:.2f} 摩擦={friction:.2f}")
        print(f"├─ 运动范围: [{lower_limit:.2f}, {upper_limit:.2f}] rad")
        print(f"├─ 极限参数: 最大力={max_force:.1f}N 最大速度={max_velocity:.1f}rad/s")
        print(f"├─ 父子链接: 父={parent_idx} 子={child_idx}")
        print(f"└─ 局部坐标系: 位置={pos} 朝向={orn}")


# 使用示例
print_joint_info(robot)
# 获取当前关节状态
joint_positions = [pb.getJointState(robot, i)[0] for i in range(num_joints)]


def advanced_calculate_ik(target_pos, target_orn=None, arm='left', constraints=None):
    """
    针对R1 Pro机器人的精确逆运动学计算
    :param target_pos: 目标位置[x,y,z]（世界坐标系）
    :param target_orn: 目标朝向四元数[x,y,z,w]（可选）
    :param arm: 'left'或'right'指定机械臂
    :param constraints: 关节约束字典（可选）
    :return: 长度为6的关节角度列表（弧度）
    """
    # 机械臂参数配置
    ARM_CONFIG = {
        'left': {
            'joint_indices': [15, 16, 17, 18, 19, 20],
            'end_effector': 20,
            'default_constraints': {
                'lower': [-2.88, 0.0, -3.32, -2.88, -1.66, -2.88],
                'upper': [2.88, 3.23, 0.0, 2.88, 1.66, 2.88],
                'max_force': [40.0, 40.0, 27.0, 7.0, 7.0, 7.0]
            }
        },
        'right': {
            'joint_indices': [30, 31, 32, 33, 34, 35],
            'end_effector': 35,
            'default_constraints': {
                'lower': [-2.88, 0.0, -3.32, -2.88, -1.66, -2.88],
                'upper': [2.88, 3.23, 0.0, 2.88, 1.66, 2.88],
                'max_force': [40.0, 40.0, 27.0, 7.0, 7.0, 7.0]
            }
        }
    }

    cfg = ARM_CONFIG[arm]
    joint_indices = cfg['joint_indices']
    end_effector = cfg['end_effector']

    # 设置默认参数
    if target_orn is None:
        target_orn = pb.getQuaternionFromEuler([0, np.pi / 2, 0])  # 末端默认朝下

    if constraints is None:
        constraints = cfg['default_constraints']
    else:
        # 合并用户约束与默认约束
        constraints = {**cfg['default_constraints'], **constraints}

    # 获取当前关节状态作为初始猜测
    current_positions = [pb.getJointState(robot, i)[0] for i in joint_indices]
    print(current_positions)
    # 计算IK（使用更精确的参数配置）
    ik_solution = pb.calculateInverseKinematics(
        robot,
        endEffectorLinkIndex=end_effector,
        targetPosition=target_pos,
        targetOrientation=target_orn,
        lowerLimits=constraints['lower'],
        upperLimits=constraints['upper'],
        jointRanges=[u - l for l, u in zip(constraints['lower'], constraints['upper'])],
        restPoses=current_positions,  # 使用当前姿势作为初始猜测
        maxNumIterations=1000,  # 增加迭代次数
        residualThreshold=1e-5,  # 更严格的收敛阈值
        physicsClientId=pb_client
    )

    # 验证解的有效性
    solution = [ik_solution[i] for i in joint_indices]
    if not all(constraints['lower'][i] <= solution[i] <= constraints['upper'][i] for i in range(6)):
        print("警告：获得的解超出关节限制！")

    return solution
# ===== 使用示例 =====
# 1. 设置目标位置和朝向
target_pos = [0., 0.0, 0.0]  # 目标位置[x,y,z]
target_orn = pb.getQuaternionFromEuler([0, np.pi/2, 0])  # 末端朝下

# 2. 计算左臂逆解
left_arm_angles = advanced_calculate_ik(
    target_pos=target_pos,
    target_orn=target_orn,
    arm='left'
)

# 3. 计算右臂逆解


print("左臂关节角度(弧度):", left_arm_angles)

arm_joint_indices = [15, 16, 17, 18, 19, 20]
while pb.isConnected():
    # 4. 应用控制
    for joint_idx, angle in zip(arm_joint_indices, left_arm_angles):
        pb.setJointMotorControl2(
            robot,
            jointIndex=joint_idx,
            controlMode=pb.POSITION_CONTROL,
            targetPosition=angle,
            force=500,
            positionGain=0.8
        )
    pb.stepSimulation()
    time.sleep(0.01)

pb.disconnect()