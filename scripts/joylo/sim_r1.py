import os
import time
import multiprocessing as mp
import numpy as np
import pybullet as p
import cv2
import ctypes
from brs_ctrl.asset_root import ASSET_ROOT
from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1JoyConInterface
import pybullet_data
from urdf_models import models_data
import random
# 机械臂中立位置
neutral_left_arm_qs = np.array([1.56, 2.94, -2.54, 0, 0, 0])
neutral_right_arm_qs = np.array([-1.56, 2.94, -2.54, 0, 0, 0])

# 摄像头关节索引
CAMERA_JOINTS = {
    "head_zed2": 14,  # 头部主摄像头
    "left_arm_zedm": 27,  # 左臂摄像头
    "right_arm_zedm": 42  # 右臂摄像头
}


class SharedData(ctypes.Structure):
    _fields_ = [
        ('mobile_base_cmd', ctypes.c_float * 3),  # [0]:前后速度, [1]:左右转向, [2]:旋转速度,
        ('torso_cmd', ctypes.c_float * 4),
        ('left_arm_cmd', ctypes.c_float * 6),
        ('right_arm_cmd', ctypes.c_float * 6),
        ('left_gripper_cmd', ctypes.c_float),
        ('right_gripper_cmd', ctypes.c_float),
        ('updated', ctypes.c_bool),
        ('camera_joint_index', ctypes.c_int)  # 当前使用的摄像头索引
    ]


def get_motor_port():
    """检测可用的电机端口"""
    ports_to_try = ["/dev/ttyUSB0", "/dev/ttyUSB1"]
    for port in ports_to_try:
        if os.path.exists(port):
            return port
    raise FileNotFoundError("未找到可用电机端口 (尝试: %s)" % ports_to_try)


def get_camera_image(robot, cam_joint_index, width=640, height=480):
    """获取摄像头视图"""
    # 获取摄像头位姿
    cam_state = p.getLinkState(robot, cam_joint_index, computeForwardKinematics=True)
    cam_pos, cam_orn = cam_state[0], cam_state[1]

    # 转换为旋转矩阵
    rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
    forward = rot_matrix[:, 2]  # 前向向量
    up = -rot_matrix[:, 1]  # 上向量（光学坐标系Y向下，取反）

    # 计算视图矩阵
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=cam_pos,
        cameraTargetPosition=cam_pos + forward * 1.5,  # 看向前方1.5米处
        cameraUpVector=up
    )

    # 获取图像
    _, _, rgb, depth, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=p.computeProjectionMatrixFOV(60, width / height, 0.02, 5),
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    # 转换颜色空间
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
    cv2.putText(rgb, f"Camera: {cam_joint_index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return rgb, depth


def motor_control(shared_data):
    """机器人控制进程"""
    joylo_arms = JoyLoArmPositionController(
        left_motor_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        right_motor_ids=[8, 9, 10, 11, 12, 13, 14, 15],
        motors_port=get_motor_port(),
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
    joycon = R1JoyConInterface()
    joylo = JoyLoController(joycon=joycon, joylo_arms=joylo_arms)

    current_camera = CAMERA_JOINTS["head_zed2"]  # 默认摄像头
    ne_torso_qs=np.array([0.0, 0.0, 0.0, 0.0])
    while True:
        joylo_actions = joylo.act(ne_torso_qs)

        with shared_data.get_lock():
            # 控制指令
            for i in range(4):
                shared_data.torso_cmd[i] = joylo_actions["torso_cmd"][i]
            for i in range(6):
                shared_data.left_arm_cmd[i] = joylo_actions["arm_cmd"]["left"][i]
                shared_data.right_arm_cmd[i] = joylo_actions["arm_cmd"]["right"][i]
            # 底盘控制指令 (直接传递3维向量)
            for i in range(3):
                shared_data.mobile_base_cmd[i] = joylo_actions["mobile_base_cmd"][i]
            # 夹爪控制
            shared_data.left_gripper_cmd = 1.0 if joylo_actions["gripper_cmd"]["left"] >= 0.5 else 0.0
            shared_data.right_gripper_cmd = 1.0 if joylo_actions["gripper_cmd"]["right"] >= 0.5 else 0.0

            # 摄像头切换
            if joylo_actions.get("camera_switch", False):
                if current_camera == CAMERA_JOINTS["head_zed2"]:
                    current_camera = CAMERA_JOINTS["left_arm_zedm"]
                else:
                    current_camera = CAMERA_JOINTS["head_zed2"]
                shared_data.camera_joint_index = current_camera

            shared_data.updated = True

        time.sleep(0.01)


def compute_wheel_controls(vx, vy, omega):
    """修正版全向轮控制（确保前进/后退时轮子直行）"""
    wheel_positions = np.array([
        [0.21516, 0.28001],  # 轮1
        [0.21516, -0.27999],  # 轮2
        [-0.28085, 0]  # 轮3
    ])

    # 计算各轮速度分量
    wheel_vx = vx - omega * wheel_positions[:, 1]
    wheel_vy = vy + omega * wheel_positions[:, 0]

    # 关键修正：当不需要转向时强制角度为0
    is_straight = (abs(vy) < 0.001) and (abs(omega) < 0.001)
    speeds = np.sqrt(wheel_vx ** 2 + wheel_vy ** 2)

    if is_straight:
        # 纯前后运动：所有轮直行
        steers = np.zeros(3)
        signed_speeds = np.sign(vx + 1e-6) * speeds
    else:
        # 平移/旋转运动：计算实际转向角
        steers = np.arctan2(wheel_vy, wheel_vx)
        signed_speeds = speeds * np.sign(wheel_vx + 1e-6)

    return [{'steer': float(s), 'speed': float(v)}
            for s, v in zip(steers, signed_speeds)]

def visualization(shared_data):
    """可视化进程"""
    # 初始化PyBullet
    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(fixedTimeStep=1 / 240.)

    # 加载场景
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    p.changeDynamics(planeId, -1,
                     lateralFriction=1.5,
                     spinningFriction=0.5,
                     rollingFriction=0.3)
    # 加载内置场景
    # 加载桌子并固定到坐标 (1.0, 0.5, 0)
    table_pos = [1.0, 0.0, 0]  # 指定桌子位置
    table_orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])  # 旋转90度（根据需要调整）
    table = p.loadURDF("table/table.urdf",
                       basePosition=table_pos,
                       baseOrientation=table_orn,  # 添加朝向控制
                       globalScaling=1.2,
                       useFixedBase=True)

    # 获取桌面高度（桌子高度0.74m，桌面厚度0.05m）
    table_height = 0.9- 0.05 / 2  # 实际桌面表面高度

    # 加载内置模型库
    models = models_data.model_lib()
    namelist = models.model_name_list
    flags = p.URDF_USE_INERTIA_FROM_FILE
    # 在桌面上方堆叠8个随机物体
    for i in range(3):
        random_model = namelist[random.randint(0, len(namelist) - 1)]
        # 计算物体位置（桌面高度 + 堆叠偏移）
        obj_height = 0.15 * i  # 每个物体间隔15cm
        obj_pos = [table_pos[0]-0.3, table_pos[1], table_height + obj_height + 0.05]  # +0.05防止穿透

        # 加载物体（添加随机旋转）
        obj_orn = p.getQuaternionFromEuler([
            random.uniform(0, np.pi / 4),
            random.uniform(0, np.pi / 4),
            random.uniform(0, np.pi / 2)
        ])
        p.loadURDF(models[random_model],
                   basePosition=obj_pos,
                   baseOrientation=obj_orn,flags=flags)

    # 加载机器人
    robot = p.loadURDF(
        os.path.join(ASSET_ROOT, "robot/r1_pro/r1_pro.urdf"),
        [0, 0, 0.1],  # 稍微抬高避免碰撞
        useFixedBase=False
    )

    # 获取关节索引
    wheel_joints = [1, 3, 5]  # 轮子关节
    servo_joints = [0, 2, 4]  # 转向关节
    torso_joint_idxs = [6, 7, 8, 9]
    left_arm_joint_idxs = [15, 16, 17, 18, 19, 20]
    right_arm_joint_idxs = [30, 31, 32, 33, 34, 35]
    left_gripper_joint_idxs = [28, 29]
    right_gripper_joint_idxs = [43, 44]
    # 物理参数
    WHEEL_RADIUS = 0.1 # 根据实际轮子调整

    # 初始化轮子控制
    for joint in wheel_joints:
        p.setJointMotorControl2(
            robot, joint,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=50
        )

    # 参数配置（根据实际硬件调整）
    WHEEL_RADIUS = 0.05  # 轮子半径（米）
    STEER_FORCE = 500.0  # 转向力矩（N·m）
    DRIVE_FORCE = 550.0  # 驱动力矩（N·m）
    MAX_STEER_VEL = 3.0  # 最大转向速度（rad/s）
    STEER_KP = 5.5  # 转向位置增益
    STEER_KD = 0.2  # 转向速度增益
    # 主循环
    current_camera = CAMERA_JOINTS["head_zed2"]
    while p.isConnected():
        # 处理控制指令
        with shared_data.get_lock():
            if shared_data.updated:

                # 解析控制指令
                vx = shared_data.mobile_base_cmd[0]  # 前后速度
                vy = shared_data.mobile_base_cmd[1]  # 左右速度
                omega = shared_data.mobile_base_cmd[2]  # 旋转速度

                # 计算三个全向轮的目标速度
                controls = compute_wheel_controls(vx, vy, omega)
                print("controls",controls)
                # 应用控制
                for i, (servo_joint, wheel_joint) in enumerate(zip(servo_joints, wheel_joints)):
                    # 1. 转向控制（带角度平滑处理）
                    current_angle = p.getJointState(robot, servo_joint)[0]
                    target_steer = controls[i]['steer']

                    # 角度差计算（处理2π环绕）
                    angle_error = (target_steer - current_angle + np.pi) % (2 * np.pi) - np.pi

                    p.setJointMotorControl2(
                        robot,
                        servo_joint,
                        p.POSITION_CONTROL,
                        targetPosition=current_angle + angle_error,
                        force=STEER_FORCE,
                        maxVelocity=MAX_STEER_VEL,
                        positionGain=STEER_KP,
                        velocityGain=STEER_KD
                    )

                    # 2. 驱动控制（带速度滤波）
                    target_speed = controls[i]['speed']
                    angular_vel = target_speed / WHEEL_RADIUS  # 线速度转角速度


                    p.setJointMotorControl2(
                        robot,
                        wheel_joint,
                        p.VELOCITY_CONTROL,
                        targetVelocity=angular_vel*10,
                        force=DRIVE_FORCE
                    )

                # 更新机器人状态
                for i, q in zip(torso_joint_idxs, shared_data.torso_cmd):
                    p.resetJointState(robot, i, q)
                for i, q in enumerate(shared_data.left_arm_cmd):
                    p.resetJointState(robot, left_arm_joint_idxs[i], q)
                for i, q in enumerate(shared_data.right_arm_cmd):
                    p.resetJointState(robot, right_arm_joint_idxs[i], q)

                # 更新夹爪
                for joint_idx in left_gripper_joint_idxs:
                    p.setJointMotorControl2(
                        robot, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=shared_data.left_gripper_cmd,
                        force=30.0
                    )
                for joint_idx in right_gripper_joint_idxs:
                    p.setJointMotorControl2(
                        robot, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=shared_data.right_gripper_cmd,
                        force=30.0
                    )

                # 更新摄像头
                if hasattr(shared_data, 'camera_joint_index'):
                    current_camera = shared_data.camera_joint_index

                # 直接使用归一化输入（-1到1范围）

                shared_data.updated = False

        # 获取并显示摄像头视图
        rgb, _ = get_camera_image(robot, current_camera)
        #cv2.imshow("Robot Camera", rgb)
       # if cv2.waitKey(1) == ord('q'):
        #    break

        p.stepSimulation()
        time.sleep(0.1)

    p.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 创建共享内存
    shared_data = mp.Value(SharedData)
    shared_data.updated = False
    shared_data.left_gripper_cmd = 0.0
    shared_data.right_gripper_cmd = 0.0
    shared_data.camera_joint_index = CAMERA_JOINTS["head_zed2"]

    # 创建并启动进程
    processes = [
        mp.Process(target=motor_control, args=(shared_data,)),
        mp.Process(target=visualization, args=(shared_data,))
    ]

    try:
        for pr in processes:
            pr.start()

        for pr in processes:
            pr.join()

    except KeyboardInterrupt:
        for pr in processes:
            pr.terminate()
        for pr in processes:
            pr.join()