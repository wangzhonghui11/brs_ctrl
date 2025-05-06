import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import deque
import pinocchio as pin
# 初始化环境
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)
p.setTimeStep(0.001)

# 加载模型
urdf_path = os.path.abspath("A1.urdf")
try:
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
except:
    print(f"无法加载URDF文件: {urdf_path}")
    print("使用PyBullet示例模型...")
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 1])

# 获取关节信息
num_joints = p.getNumJoints(robot_id)
actuated_joint_ids = [i for i in range(num_joints) if p.getJointInfo(robot_id, i)[2] != p.JOINT_FIXED][:6]
joint_names = [p.getJointInfo(robot_id, i)[1].decode() for i in actuated_joint_ids]
print("控制关节:", joint_names)

# 初始化Pinocchio（修正后的正确方式）
model = None
data = None
try:
    # 正确的模型构建方式
    model = pin.buildModelFromUrdf(urdf_path)  # 注意是单数形式 buildModelFromUrdf
    data = model.createData()
    print("Pinocchio初始化成功，启用精确重力补偿")
except ImportError:
    print("警告：未安装Pinocchio，将使用简化重力补偿")
except Exception as e:
    print(f"Pinocchio初始化失败: {str(e)}")
    model = None
    data = None

# 控制参数
Kp = np.array([140.0, 500.0, 120.0, 20.0, 20.0, 20.0])
Kd = np.array([10.0, 50.0, 5.0, 1.0, 1.0, 0.4])
max_torque = np.array([100, 100, 50, 20, 20, 10])


# 轨迹规划器
class QuinticTrajectory:
    def __init__(self, start_pos, target_pos, duration):
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.duration = duration
        self.start_time = None
        self.a0 = start_pos
        self.a1 = np.zeros_like(start_pos)
        self.a2 = np.zeros_like(start_pos)
        self.a3 = 10 * (target_pos - start_pos) / duration ** 3
        self.a4 = -15 * (target_pos - start_pos) / duration ** 4
        self.a5 = 6 * (target_pos - start_pos) / duration ** 5

    def update(self, t):
        if self.start_time is None:
            self.start_time = t
            return self.a0, np.zeros_like(self.a0), np.zeros_like(self.a0)

        elapsed = t - self.start_time
        if elapsed > self.duration:
            return self.target_pos, np.zeros_like(self.a0), np.zeros_like(self.a0)

        t_norm = elapsed / self.duration
        pos = (self.a0 + self.a3 * elapsed ** 3 +
               self.a4 * elapsed ** 4 + self.a5 * elapsed ** 5)
        vel = (3 * self.a3 * elapsed ** 2 +
               4 * self.a4 * elapsed ** 3 + 5 * self.a5 * elapsed ** 4)
        acc = (6 * self.a3 * elapsed +
               12 * self.a4 * elapsed ** 2 + 20 * self.a5 * elapsed ** 3)

        return pos, vel, acc


# 初始化轨迹
start_pos = np.zeros(6)
target_pos = np.array([0, 0, -2, 0, 0, 0])
trajectory = QuinticTrajectory(start_pos, target_pos, duration=3.0)


# 数据记录器
class TrajectoryLogger:
    def __init__(self):
        self.time_history = deque(maxlen=1000)
        self.target_pos_history = [deque(maxlen=1000) for _ in range(6)]
        self.actual_pos_history = [deque(maxlen=1000) for _ in range(6)]
        self.torque_history = [deque(maxlen=1000) for _ in range(6)]
        self.gravity_comp_history = [deque(maxlen=1000) for _ in range(6)]

    def update(self, t, target, actual, torque, gravity_comp):
        self.time_history.append(t)
        for i in range(6):
            self.target_pos_history[i].append(target[i])
            self.actual_pos_history[i].append(actual[i])
            self.torque_history[i].append(torque[i])
            self.gravity_comp_history[i].append(gravity_comp[i])

    def plot(self):
        plt.figure(figsize=(15, 12))

        # 位置跟踪
        plt.subplot(3, 1, 1)
        for i in range(6):
            plt.plot(self.time_history, self.target_pos_history[i], '--', label=f'Target J{i}')
            plt.plot(self.time_history, self.actual_pos_history[i], '-', label=f'Actual J{i}')
        plt.ylabel('Position (rad)')
        plt.legend()

        # 力矩输出
        plt.subplot(3, 1, 2)
        for i in range(6):
            plt.plot(self.time_history, self.torque_history[i], label=f'Total Torque J{i}')
        plt.ylabel('Torque (Nm)')
        plt.legend()

        # 重力补偿
        plt.subplot(3, 1, 3)
        for i in range(6):
            plt.plot(self.time_history, self.gravity_comp_history[i], label=f'Gravity Comp J{i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Gravity Torque (Nm)')
        plt.legend()

        plt.tight_layout()
        plt.show()


logger = TrajectoryLogger()

# 禁用默认控制器
for joint_id in actuated_joint_ids:
    p.setJointMotorControl2(robot_id, joint_id, p.VELOCITY_CONTROL, force=0)

# 主控制循环
try:
    start_time = time.time()
    while True:
        current_time = time.time() - start_time

        # 获取轨迹点
        target_pos, target_vel, target_acc = trajectory.update(current_time)

        # 读取当前状态
        current_pos = np.zeros(6)
        current_vel = np.zeros(6)
        for i, joint_id in enumerate(actuated_joint_ids):
            current_pos[i], current_vel[i] = p.getJointState(robot_id, joint_id)[0:2]

        # 计算重力补偿（修正后的正确方式）
        gravity_comp = np.zeros(6)
        if model is not None:
            try:
                # 构建Pinocchio状态向量
                q = pin.neutral(model)
                for i, joint_id in enumerate(actuated_joint_ids):
                    q[model.idx_qs[i + 1]] = current_pos[i]  # 注意索引偏移

                # 计算广义重力
                pin.computeGeneralizedGravity(model, data, q)
                gravity_comp = data.g[6:12]  # 提取驱动关节对应的重力项
            except Exception as e:
                print(f"重力补偿计算错误: {str(e)}")

        # 计算PD控制
        error = target_pos - current_pos
        error_vel = target_vel - current_vel
        tau_feedback = Kp * error + Kd * error_vel
        tau = np.clip(tau_feedback + gravity_comp, -max_torque, max_torque)

        # 应用控制
        for i, joint_id in enumerate(actuated_joint_ids):
            p.setJointMotorControl2(
                robot_id, joint_id,
                controlMode=p.TORQUE_CONTROL,
                force=tau[i]
            )

        # 记录数据
        logger.update(current_time, target_pos, current_pos, tau, gravity_comp)

        # 显示调试信息
        if len(logger.time_history) % 10 == 0:
            p.addUserDebugText(
                text=f"Time: {current_time:.2f}s\n" +
                     "\n".join([f"J{i}: {current_pos[i]:.2f}/{target_pos[i]:.2f}\n" +
                                f"G{i}: {gravity_comp[i]:.2f}Nm"
                                for i in range(3)]),  # 只显示前3个关节避免信息过载
                textPosition=[0, 0, 0.5],
                textColorRGB=[1, 1, 0],
                lifeTime=0.1
            )

        p.stepSimulation()
        time.sleep(0.001)

except KeyboardInterrupt:
    p.disconnect()
    logger.plot()