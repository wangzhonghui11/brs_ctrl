import os
import time
import numpy as np
import pybullet as pb
from brs_ctrl.asset_root import ASSET_ROOT
import pybullet_data

import numpy as np
import pybullet as pb
from scipy.interpolate import CubicSpline


class PathPlanner:
    def __init__(self, joint_constraints):
        """
        :param joint_constraints: 关节约束字典 {
            'lower': [min_angle1, ...],
            'upper': [max_angle1, ...],
            'max_velocity': [max_vel1, ...]
        }
        """
        self.constraints = joint_constraints

    def generate_joint_trajectory(self, start, target, steps=500, method='linear'):
        """
        生成关节空间轨迹
        :param method: 'linear'/'cubic'/'quintic'
        :return: 轨迹数组 (n_joints × steps)
        """
        if method == 'linear':
            return self._linear_interpolation(start, target, steps)
        elif method == 'cubic':
            return self._cubic_spline(start, target, steps)
        else:
            raise ValueError(f"未知插值方法: {method}")

    def _linear_interpolation(self, start, target, steps):
        """线性插值"""
        return np.column_stack([
            np.linspace(s, t, steps)
            for s, t in zip(start, target)
        ]).T

    def _cubic_spline(self, start, target, steps):
        """三次样条插值（带速度曲线）"""
        time_points = np.array([0, steps // 2, steps - 1])
        trajectories = []

        for j in range(len(start)):
            # 中间点取均值
            mid = (start[j] + target[j]) / 2
            spline = CubicSpline(
                time_points,
                [start[j], mid, target[j]],
                bc_type='clamped'
            )
            trajectories.append(spline(np.arange(steps)))

        return np.array(trajectories)

    def check_joint_limits(self, positions):
        """检查关节位置是否在限制范围内"""
        return all(
            lower <= pos <= upper
            for pos, lower, upper in zip(
                positions,
                self.constraints['lower'],
                self.constraints['upper']
            )
        )

    def generate_cartesian_path(self, start_pose, target_pose, steps):
        """
        笛卡尔空间直线规划
        :param start_pose: (position, orientation)
        :param target_pose: (position, orientation)
        :return: 位姿列表 [(pos1, orn1), ...]
        """
        start_pos, start_orn = start_pose
        target_pos, target_orn = target_pose

        # 位置线性插值
        positions = np.linspace(start_pos, target_pos, steps)

        # 姿态球面插值
        orientations = [
            pb.getQuaternionSlerp(start_orn, target_orn, t)
            for t in np.linspace(0, 1, steps)
        ]

        return list(zip(positions, orientations))

class PyBulletSimulator:
    def __init__(self, gui=True):
        self.physics_client = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        self.robot = None
        self.arm_controllers = {}
    def load_sence(self):
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = pb.loadURDF("plane.urdf")
        pb.changeDynamics(planeId, -1,
                         lateralFriction=1.5,
                         spinningFriction=0.5,
                         rollingFriction=0.3)
        # 加载桌子并固定到坐标 (1.0, 0.5, 0)
        table_pos = [0.7, 0.0, 0]  # 指定桌子位置
        table_orn = pb.getQuaternionFromEuler([0, 0, np.pi / 2])  # 旋转90度（根据需要调整）
        table = pb.loadURDF("table/table.urdf",
                           basePosition=table_pos,
                           baseOrientation=table_orn,  # 添加朝向控制
                           globalScaling=1.2,
                           useFixedBase=True)
        print(pybullet_data.getDataPath())  # 输出数据目录路径
        # 获取桌面高度（桌子高度0.74m，桌面厚度0.05m）
        pb.setPhysicsEngineParameter(
            fixedTimeStep=1 / 500,  # 更小的仿真步长
            numSolverIterations=100,  # 更多求解器迭代
            contactBreakingThreshold=0.0001,  # 更敏感的接触检测
            enableConeFriction=1  # 启用锥形摩擦模型
        )
    def load_robot(self, urdf_path, position=[0, 0, 0.1]):
        #self.load_sence()
        # 加载机器人时启用自碰撞但排除所有默认碰撞
        flags = (pb.URDF_USE_SELF_COLLISION |
                 pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT |
                 pb.URDF_USE_INERTIA_FROM_FILE)

        self.robot = pb.loadURDF(
            os.path.join(ASSET_ROOT, urdf_path),
            position,
            useFixedBase=True,
            physicsClientId=self.physics_client,
            #flags=flags ,  # 启用自碰撞
        )
        return self.robot

    def get_joint_info(self, joint_index):
        return pb.getJointInfo(self.robot, joint_index, physicsClientId=self.physics_client)
    def init_joint(self,joint_indices,init_joint):
        for i, q in zip(joint_indices,init_joint):
            pb.resetJointState(self.robot, i, q, physicsClientId=self.physics_client)

    def print_all_joints_info(self):
        print("\n" + "=" * 50)
        print(f"{'机器人关节信息':^50}")
        print("=" * 50)

        for i in range(pb.getNumJoints(self.robot)):
            joint_info = self.get_joint_info(i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = {
                0: "旋转关节(revolute)",
                1: "滑动关节(prismatic)",
                2: "球关节(spherical)",
                3: "平面关节(planar)",
                4: "固定关节(fixed)"
            }.get(joint_info[2], "未知类型")

            print(f"\n◆ 关节索引 [{i}] {joint_name}")
            print(f"├─ 类型: {joint_type}")
            print(f"├─ 动态参数: 阻尼={joint_info[6]:.2f} 摩擦={joint_info[7]:.2f}")
            print(f"├─ 运动范围: [{joint_info[8]:.2f}, {joint_info[9]:.2f}] rad")
            print(f"├─ 极限参数: 最大力={joint_info[10]:.1f}N 最大速度={joint_info[11]:.1f}rad/s")

    def add_arm_controller(self, name, config):
        self.arm_controllers[name] = ArmController(self, config)

    def run_simulation(self, duration=None):
        start_time = time.time()
        while pb.isConnected():
            if duration and (time.time() - start_time) > duration:
                break
            pb.stepSimulation()
            time.sleep(1. / 240.)

    def disconnect(self):
        pb.disconnect(physicsClientId=self.physics_client)


class ArmController:
    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config
        self.joint_indices = config['joint_indices']
        self.init_joint = config['init_joint']
        self.end_effector = config['end_effector']
        self.ik_inds = config['ik_inds']
        self.constraints = config['default_constraints']
        self.torso_joint_idxs = [6, 7, 8, 9]
        self.link_names=self.get_r1_link_info()
        self.planner = PathPlanner(config['default_constraints'])
    def get_r1_link_info(self):
        link_names = []
        for i in range(pb.getNumJoints(self.simulator.robot)):
            joint_info = pb.getJointInfo(self.simulator.robot, i)
            link_names.append(joint_info[12].decode('utf-8'))  # 获取连杆名称

        print("link_names",link_names)
        return link_names

    def check_arm_torso_collision(self):
        """精确检测机械臂与腰部的连杆碰撞"""
        # 定义需要检测的连杆组合
        arm_links = [
            'left_arm_link1', 'left_arm_link2', 'left_arm_link3',
            'left_arm_link4', 'left_arm_link5', 'left_arm_link6',
            'right_arm_link1', 'right_arm_link2', 'right_arm_link3',
            'right_arm_link4', 'right_arm_link5', 'right_arm_link6'
        ]

        torso_links = ['torso_link1', 'torso_link2', 'torso_link3', 'torso_link4']

        # 排除直接连接的连杆对（根据实际物理连接关系）
        excluded_pairs = {
            ('left_arm_link1', 'torso_link4'),  # 左臂基座与腰部连接
            ('right_arm_link1', 'torso_link4')  # 右臂基座与腰部连接
        }

        # 建立名称到索引的映射
        link_name_to_index = {name: i for i, name in enumerate(self.link_names)}
        # 检测所有可能的碰撞组合
        for arm_link in arm_links:
            for torso_link in torso_links:
                # 跳过排除的组合
                if (arm_link, torso_link) in excluded_pairs:
                    continue
                pb.stepSimulation(physicsClientId=self.simulator.physics_client)
                contacts = pb.getClosestPoints(
                    bodyA=self.simulator.robot,
                    bodyB=self.simulator.robot,
                    distance=0.01,
                    linkIndexA=link_name_to_index[arm_link],
                    linkIndexB=link_name_to_index[torso_link],
                    physicsClientId=self.simulator.physics_client
                )
                # 有效碰撞判断（深度>1cm且至少2个接触点）
                if contacts :
                    return True
        return False

    def _is_valid_collision(self, contacts):
        """验证是否为有效碰撞"""
        min_depth = 0.01  # 1cm深度阈值
        min_contacts = 2  # 至少2个接触点

        if len(contacts) < min_contacts:
            return False

        max_depth = max(abs(c[8]) for c in contacts)
        return max_depth > min_depth

    def init_arm_joints(self):
        """初始化机械臂关节到预设角度"""
        self.simulator.init_joint(self.joint_indices, self.init_joint)

    def get_current_joint_positions(self):
        return [pb.getJointState(self.simulator.robot, i,
                                 physicsClientId=self.simulator.physics_client)[0]
                for i in self.joint_indices]

    def get_end_effector_state(self):
        """
        获取末端执行器状态
        返回: (position, orientation, linear_velocity, angular_velocity)
        """
        link_state = pb.getLinkState(
            self.simulator.robot,
            self.end_effector,
            computeLinkVelocity=1,
            physicsClientId=self.simulator.physics_client
        )
        return {
            'position': link_state[4],        # 世界坐标系位置 [x,y,z]
            'orientation': link_state[5],    # 世界坐标系姿态四元数 [x,y,z,w]
            'linear_velocity': link_state[6],  # 线速度 [vx,vy,vz]
            'angular_velocity': link_state[7]  # 角速度 [wx,wy,wz]
        }

    def get_end_effector_pose(self):
        """仅获取末端执行器位姿"""
        state = self.get_end_effector_state()
        return state['position'], state['orientation']

    def print_end_effector_info(self):
        """打印末端执行器状态信息"""
        state = self.get_end_effector_state()
        print(f"\n末端执行器 {self.end_effector} 状态:")
        print(f"位置: {np.round(state['position'], 4)}")
        print(f"姿态: {np.round(state['orientation'], 4)}")
        print(f"线速度: {np.round(state['linear_velocity'], 4)}")
        print(f"角速度: {np.round(state['angular_velocity'], 4)}")
    def calculate_ik(self, target_pos, target_orn=None):
        if target_orn is None:
            target_orn = pb.getQuaternionFromEuler([0, np.pi / 2, 0])

        current_positions = self.get_current_joint_positions()

        ik_solution = pb.calculateInverseKinematics(
            self.simulator.robot,
            endEffectorLinkIndex=self.end_effector,
            targetPosition=target_pos,
            targetOrientation=target_orn,
            lowerLimits=self.constraints['lower'],
            upperLimits=self.constraints['upper'],
            jointRanges=[u - l for l, u in zip(self.constraints['lower'], self.constraints['upper'])],
            restPoses=current_positions,
            maxNumIterations=1000,
            residualThreshold=1e-5,
            physicsClientId=self.simulator.physics_client
        )

        solution = [ik_solution[i] for i in self.ik_inds]
        if not all(self.constraints['lower'][i] <= solution[i] <= self.constraints['upper'][i] for i in range(6)):
            print("警告：获得的解超出关节限制！")


        return solution

    def safe_move_to_pose(self, target_pos, target_orn=None, duration=1.0, steps=100):
        """
        带腰部碰撞检测的安全运动
        :return: bool (True表示运动完成且无碰撞)
        """
        # 1. 计算目标关节角度
        target_joints = self.calculate_ik(target_pos, target_orn)
        if target_joints is None:
            print("IK求解失败：目标不可达")
            return False

        # 2. 获取当前关节角度
        current_joints = self.get_current_joint_positions()

        # 3. 生成平滑轨迹
        trajectory = self._generate_trajectory(current_joints, target_joints, steps)

        # 4. 执行轨迹并检测碰撞
        step_delay = duration / len(trajectory[0])

        for i in range(len(trajectory[0])):
            # 设置当前关节角度
            joint_positions = [traj[i] for traj in trajectory]
            self.set_joint_positions(joint_positions)

            # 实时碰撞检测
            if self.check_arm_torso_collision():
                print(f"在路径点 {i} 检测到腰部碰撞，终止运动")
                return False

            time.sleep(step_delay)
        return True
    def move_to_pose(self, target_pos, target_orn=None, duration=1.0, steps=100):
        """
        平滑移动到目标位姿
        :param target_pos: 目标位置 [x,y,z]
        :param target_orn: 目标姿态四元数 [x,y,z,w] (可选)
        :param duration: 运动持续时间(秒)
        :param steps: 插值步数
        """
        # 1. 计算目标关节角度
        target_joints = self.calculate_ik(target_pos, target_orn)

        # 2. 获取当前关节角度
        current_joints = self.get_current_joint_positions()

        # 3. 生成平滑轨迹
        trajectory = self.planner.generate_joint_trajectory(
            start=current_joints,
            target=target_joints,
            steps=steps,
            method='cubic'  # 可配置
        )

        # 4. 执行轨迹
        self._execute_trajectory(trajectory, duration)

    def _generate_trajectory(self, start, end, steps):
        """生成关节空间平滑轨迹"""
        return [np.linspace(s, e, steps) for s, e in zip(start, end)]

    def _execute_trajectory(self, trajectory, duration):
        """执行生成的轨迹"""
        step_delay = duration / trajectory.shape[1]

        for i in range(trajectory.shape[1]):
            joint_positions = trajectory[:, i]
            self.set_joint_positions(joint_positions)
            time.sleep(step_delay)

    def set_joint_positions(self, positions, max_velocity=None):
        """
        设置关节位置（带速度限制）
        :param positions: 目标角度列表
        :param max_velocity: 最大关节速度(rad/s)
        """
        for joint_idx, angle in zip(self.joint_indices, positions):
            if max_velocity is not None:
                # 使用速度控制实现平滑
                pb.setJointMotorControl2(
                    bodyUniqueId=self.simulator.robot,
                    jointIndex=joint_idx,
                    controlMode=pb.POSITION_CONTROL,
                    targetPosition=angle,
                    targetVelocity=max_velocity,
                    force=self.constraints['max_force'][self.joint_indices.index(joint_idx)],
                    positionGain=0.5,
                    velocityGain=0.5,
                    physicsClientId=self.simulator.physics_client
                )
            else:
                # 直接设置
                pb.resetJointState(
                    self.simulator.robot,
                    joint_idx,
                    angle,
                    physicsClientId=self.simulator.physics_client
                )
            # 确保立即更新
            pb.stepSimulation()


# 使用示例
if __name__ == "__main__":
    # 1. 初始化仿真环境
    sim = PyBulletSimulator(gui=True)

    # 2. 加载机器人模型
    sim.load_robot("robot/r1_pro/r1_pro.urdf")

    # 3. 打印关节信息
    sim.print_all_joints_info()

    # 4. 配置机械臂控制器
    left_arm_config = {
        'joint_indices': [15, 16, 17, 18, 19, 20],
        'end_effector': 21,
        'ik_inds': [6, 7, 8, 9, 10, 11],
        'init_joint':[1.56, 2.94, -2.54, 0, 0, 0],
        'default_constraints': {
            'lower': [-2.88, 0.0, -3.32, -2.88, -1.66, -2.88],
            'upper': [2.88, 3.23, 0.0, 2.88, 1.66, 2.88],
            'max_force': [40.0, 40.0, 27.0, 7.0, 7.0, 7.0]
        }
    }
    # 5. 配置机械臂控制器 - 右臂
    right_arm_config = {
        'joint_indices': [30, 31, 32, 33, 34, 35],
        'end_effector': 36,
        'ik_inds': [14, 15, 16, 17, 18, 19],
        'init_joint': [-1.56, 2.94, -2.54, 0, 0, 0],
        'default_constraints': {
            'lower': [-2.88, 0.0, -3.32, -2.88, -1.66, -2.88],
            'upper': [2.88, 3.23, 0.0, 2.88, 1.66, 2.88],
            'max_force': [40.0, 40.0, 27.0, 7.0, 7.0, 7.0]
        }
    }
    sim.add_arm_controller("left_arm", left_arm_config)
    sim.add_arm_controller("right_arm", right_arm_config)
    # 5. 初始化双臂关节角度
    arm_left=sim.arm_controllers["left_arm"]
    arm_right=sim.arm_controllers["right_arm"]
    arm_left.init_arm_joints()
    arm_right.init_arm_joints()
    # 5. 获取末端执行器初始位置
    arm_left.print_end_effector_info()
    arm_right.print_end_effector_info()
    # 6. 设置目标位置并移动
    target_pos = [ 0.8127 ,0.5,  0.9306]
    target_orn = [0.9793 , 0.0011 , 0.0037, -0.2025]
    arm_right.move_to_pose(target_pos, target_orn)
    # 7. 运行仿真
    sim.run_simulation(duration=50)
    sim.disconnect()