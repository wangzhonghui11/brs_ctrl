import os
import time
import pybullet as pb
import numpy as np
import rosbag
from sensor_msgs.msg import JointState
from brs_ctrl.asset_root import ASSET_ROOT
from brs_ctrl.joylo import JoyLoController
from brs_ctrl.joylo.joylo_arms import JoyLoArmPositionController
from brs_ctrl.joylo.joycon import R1JoyConInterface

# 配置参数
BAG_FILE = "1-0084-20250417112844.bag"
TOPIC_LEFT = "/motion_target/target_joint_state_arm_left"
TOPIC_RIGHT = "/motion_target/target_joint_state_arm_right"
TOPIC_TORSO = "/hdas/feedback_torso"  # 新增腰部数据主题

# 中性位置（弧度）
neutral_left_arm_qs = np.array([1.56, 2.94, -2.54, 0, 0, 0])
neutral_right_arm_qs = np.array([-1.56, 2.94, -2.54, 0, 0, 0])
neutral_torso_qs = np.array([0.0, 0.0, 0.0, 0.0])  # 腰部中性位置


class FullBodyMotionPlayer:
    def __init__(self, robot, left_arm_idxs, right_arm_idxs, torso_idxs):
        self.robot = robot
        self.left_arm_idxs = left_arm_idxs
        self.right_arm_idxs = right_arm_idxs
        self.torso_idxs = torso_idxs

        # 运动数据存储
        self.left_arm_data = []
        self.right_arm_data = []
        self.torso_data = []

        # 播放控制
        self.current_frame = 0
        self.is_playing = False
        self.max_frames = 0

        # 重置控制
        self.reset_in_progress = False
        self.reset_start_time = 0
        self.reset_duration = 1.5  # 更长的过渡时间保证稳定性
        self.reset_target = None

    def load_motion_data(self, bag_file):
        """从ROS bag加载全身运动数据"""
        print(f"Loading motion data from {bag_file}...")
        try:
            with rosbag.Bag(bag_file, 'r') as bag:
                # 加载左臂数据
                for _, msg, _ in bag.read_messages(topics=[TOPIC_LEFT]):
                    if hasattr(msg, 'position') and len(msg.position) == len(self.left_arm_idxs):
                        self.left_arm_data.append(list(msg.position))

                # 加载右臂数据
                for _, msg, _ in bag.read_messages(topics=[TOPIC_RIGHT]):
                    if hasattr(msg, 'position') and len(msg.position) == len(self.right_arm_idxs):
                        self.right_arm_data.append(list(msg.position))

                # 新增：加载腰部数据
                for _, msg, _ in bag.read_messages(topics=[TOPIC_TORSO]):
                    if hasattr(msg, 'position') and len(msg.position) == len(self.torso_idxs):
                        self.torso_data.append(list(msg.position))

            # 确保所有数据长度一致
            self.max_frames = min(len(self.left_arm_data),
                                  len(self.right_arm_data),
                                  len(self.torso_data))

            print(f"Loaded motion data: {self.max_frames} frames")
            print(f"Arm joints: {len(self.left_arm_idxs)} left, {len(self.right_arm_idxs)} right")
            print(f"Torso joints: {len(self.torso_idxs)}")
            return True

        except Exception as e:
            print(f"Error loading motion data: {str(e)}")
            return False

    def update_playback(self):
        """更新全身运动回放"""
        if self.reset_in_progress:
            self._handle_reset_transition()
            return

        if not self.is_playing or self.current_frame >= self.max_frames:
            return

        # 设置腰部关节
        torso_positions = self.torso_data[self.current_frame]
        for joint_idx, pos in zip(self.torso_idxs, torso_positions):
            pb.setJointMotorControl2(
                self.robot,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=pos,
                force=400,
                positionGain=0.4,
                velocityGain=0.25
            )

        # 设置左臂关节
        left_positions = self.left_arm_data[self.current_frame]
        for joint_idx, pos in zip(self.left_arm_idxs, left_positions):
            pb.setJointMotorControl2(
                self.robot,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=pos,
                force=350,
                positionGain=0.35,
                velocityGain=0.2
            )

        # 设置右臂关节
        right_positions = self.right_arm_data[self.current_frame]
        for joint_idx, pos in zip(self.right_arm_idxs, right_positions):
            pb.setJointMotorControl2(
                self.robot,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=pos,
                force=350,
                positionGain=0.35,
                velocityGain=0.2
            )

        self.current_frame = (self.current_frame + 1) % self.max_frames

    def _handle_reset_transition(self):
        """处理平滑重置过渡"""
        elapsed = time.time() - self.reset_start_time
        if elapsed >= self.reset_duration:
            self.reset_in_progress = False
            # 重置后直接设置目标位置确保精度
            self._force_set_target_positions()
            return

        progress = min(elapsed / self.reset_duration, 1.0)
        progress = self._ease_in_out(progress)  # 使用缓动函数使运动更自然

        # 确定目标位置
        if self.reset_target == "neutral":
            torso_target = neutral_torso_qs
            left_target = neutral_left_arm_qs
            right_target = neutral_right_arm_qs
        else:  # 重置到第一帧
            torso_target = self.torso_data[0] if self.torso_data else neutral_torso_qs
            left_target = self.left_arm_data[0] if self.left_arm_data else neutral_left_arm_qs
            right_target = self.right_arm_data[0] if self.right_arm_data else neutral_right_arm_qs

        # 当前实际位置
        current_torso = [pb.getJointState(self.robot, i)[0] for i in self.torso_idxs]
        current_left = [pb.getJointState(self.robot, i)[0] for i in self.left_arm_idxs]
        current_right = [pb.getJointState(self.robot, i)[0] for i in self.right_arm_idxs]

        # 计算插值位置
        torso_pos = current_torso + progress * (np.array(torso_target) - np.array(current_torso))
        left_pos = current_left + progress * (np.array(left_target) - np.array(current_left))
        right_pos = current_right + progress * (np.array(right_target) - np.array(current_right))

        # 应用插值位置（使用较低的力避免振荡）
        for joint_idx, pos in zip(self.torso_idxs, torso_pos):
            pb.setJointMotorControl2(
                self.robot,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=pos,
                force=200,
                positionGain=0.2,
                velocityGain=0.15
            )

        for joint_idx, pos in zip(self.left_arm_idxs, left_pos):
            pb.setJointMotorControl2(
                self.robot,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=pos,
                force=180,
                positionGain=0.18,
                velocityGain=0.12
            )

        for joint_idx, pos in zip(self.right_arm_idxs, right_pos):
            pb.setJointMotorControl2(
                self.robot,
                joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=pos,
                force=180,
                positionGain=0.18,
                velocityGain=0.12
            )

    def _force_set_target_positions(self):
        """强制设置到目标位置（重置完成后调用）"""
        if self.reset_target == "neutral":
            # 设置腰部中性位置
            for joint_idx, pos in zip(self.torso_idxs, neutral_torso_qs):
                pb.resetJointState(self.robot, joint_idx, pos)
                pb.setJointMotorControl2(
                    self.robot,
                    joint_idx,
                    pb.POSITION_CONTROL,
                    targetPosition=pos,
                    force=300
                )

            # 设置双臂中性位置
            for joint_idx, pos in zip(self.left_arm_idxs, neutral_left_arm_qs):
                pb.resetJointState(self.robot, joint_idx, pos)
                pb.setJointMotorControl2(
                    self.robot,
                    joint_idx,
                    pb.POSITION_CONTROL,
                    targetPosition=pos,
                    force=250
                )

            for joint_idx, pos in zip(self.right_arm_idxs, neutral_right_arm_qs):
                pb.resetJointState(self.robot, joint_idx, pos)
                pb.setJointMotorControl2(
                    self.robot,
                    joint_idx,
                    pb.POSITION_CONTROL,
                    targetPosition=pos,
                    force=250
                )
        else:
            # 设置到第一帧位置
            if self.torso_data:
                for joint_idx, pos in zip(self.torso_idxs, self.torso_data[0]):
                    pb.resetJointState(self.robot, joint_idx, pos)
                    pb.setJointMotorControl2(
                        self.robot,
                        joint_idx,
                        pb.POSITION_CONTROL,
                        targetPosition=pos,
                        force=300
                    )

            if self.left_arm_data:
                for joint_idx, pos in zip(self.left_arm_idxs, self.left_arm_data[0]):
                    pb.resetJointState(self.robot, joint_idx, pos)
                    pb.setJointMotorControl2(
                        self.robot,
                        joint_idx,
                        pb.POSITION_CONTROL,
                        targetPosition=pos,
                        force=250
                    )

            if self.right_arm_data:
                for joint_idx, pos in zip(self.right_arm_idxs, self.right_arm_data[0]):
                    pb.resetJointState(self.robot, joint_idx, pos)
                    pb.setJointMotorControl2(
                        self.robot,
                        joint_idx,
                        pb.POSITION_CONTROL,
                        targetPosition=pos,
                        force=250
                    )

    def _ease_in_out(self, t):
        """缓动函数：平滑的加速和减速"""
        return t * t * (3 - 2 * t)

    def reset_playback(self):
        """平滑重置到第一帧"""
        if not self.reset_in_progress:
            self.is_playing = False
            self.reset_in_progress = True
            self.reset_start_time = time.time()
            self.reset_target = "playback"
            print("Resetting to first frame...")

    def reset_to_neutral(self):
        """平滑重置到中性位置"""
        if not self.reset_in_progress:
            self.is_playing = False
            self.reset_in_progress = True
            self.reset_start_time = time.time()
            self.reset_target = "neutral"
            print("Resetting to neutral position...")

    def toggle_playback(self):
        """切换回放状态"""
        if not self.reset_in_progress:
            self.is_playing = not self.is_playing
            print(f"Playback {'STARTED' if self.is_playing else 'PAUSED'}")


if __name__ == "__main__":
    # 初始化PyBullet
    pb_client = pb.connect(pb.GUI)
    pb.setGravity(0, 0, -9.8)
    pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    # 加载机器人模型
    robot = pb.loadURDF(
        os.path.join(ASSET_ROOT, "robot/r1_pro/r1_pro.urdf"),
        [0, 0, 0],
        useFixedBase=True,
    )

    # 初始化关节索引
    torso_joint_idxs = [6, 7, 8, 9]  # 根据实际URDF调整
    left_arm_joint_idxs = [15, 16, 17, 18, 19, 20]
    right_arm_joint_idxs = [30, 31, 32, 33, 34, 35]

    # 初始化所有关节（禁用电机控制避免初始抖动）
    for i in range(pb.getNumJoints(robot)):
        pb.resetJointState(robot, i, 0)
        pb.setJointMotorControl2(robot, i, pb.VELOCITY_CONTROL, force=0)

    # 初始化全身运动播放器
    motion_player = FullBodyMotionPlayer(
        robot,
        left_arm_joint_idxs,
        right_arm_joint_idxs,
        torso_joint_idxs
    )

    if not motion_player.load_motion_data(BAG_FILE):
        print("Failed to load motion data!")
        exit()

    # 初始化JoyCon接口
    joycon = R1JoyConInterface()

    # 主循环
    print("\nControls:")
    print("P - 播放/暂停")
    print("R - 重置回放")
    print("N - 重置到中性位置")

    while pb.isConnected():
        # 更新运动播放
        motion_player.update_playback()

        # 处理键盘输入
        keys = pb.getKeyboardEvents()
        if ord('p') in keys and keys[ord('p')] & pb.KEY_WAS_TRIGGERED:
            motion_player.toggle_playback()
        if ord('r') in keys and keys[ord('r')] & pb.KEY_WAS_TRIGGERED:
            motion_player.reset_playback()
        if ord('n') in keys and keys[ord('n')] & pb.KEY_WAS_TRIGGERED:
            motion_player.reset_to_neutral()

        # 步进仿真
        pb.stepSimulation()
        time.sleep(1. / 240.)  # 保持240Hz的仿真频率