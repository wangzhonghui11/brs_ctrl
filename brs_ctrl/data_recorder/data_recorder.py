from typing import Optional, Dict, Union, List, Literal
import threading
import time
from functools import partial
import cv2
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState, PointCloud2, Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import ros_numpy
import h5py

from brs_ctrl.robot_interface.utils import get_xyz_points
from brs_ctrl.utils import any_concat


class R1DataRecorder:
    def __init__(
        self,
        *,
        joint_state_topics: Optional[Dict[str, str]] = None,
        gripper_state_topics: Optional[Dict[str, str]] = None,
        point_cloud_topics: Optional[Dict[str, str]] = None,
        odom_topics: Optional[str] = None,
        action_topics: Optional[Dict[str, str]] = None,
        joycon_y_button_topic: str = "/joycon/functional_buttons/y_button",
        joycon_x_button_topic: str = "/joycon/functional_buttons/x_button",
        joycon_a_button_topic: str = "/joycon/functional_buttons/a_button",
        save_rgbd: bool = False,
        save_point_cloud: bool = False,  # New parameter for point cloud
        save_odometry: bool = False,  # New parameter for odometry
        save_action: bool = False,
        rgb_topics: Optional[Dict[str, str]] = None,
        depth_topics: Optional[Dict[str, str]] = None,

        record_freq: Union[int, float] = 10,
    ):
        joint_state_topics = joint_state_topics or {
            "left_arm": "/hdas/feedback_arm_left",
            "right_arm": "/hdas/feedback_arm_right",
            "torso": "/hdas/feedback_torso",
        }
        gripper_state_topics = gripper_state_topics or {
            "left_gripper": "/hdas/feedback_gripper_left",
            "right_gripper": "/hdas/feedback_gripper_right",
        }
        self._save_point_cloud = save_point_cloud  # Store point cloud flag
        self._save_odometry = save_odometry      # Store odometry flag
        point_cloud_topics = point_cloud_topics or {
            "fused": "/r1_jetson/fused_pcd",
        }
        odom_topics = odom_topics or "/camera/odom/sample"
        self._save_rgbd = save_rgbd
        rgb_topics = rgb_topics or {
            "head": "/hdas/camera_head/left_raw/image_raw_color",
            "left_wrist": "/hdas/camera_wrist_left/color/image_raw",
            "right_wrist": "/hdas/camera_wrist_right/color/image_raw",
        }
        depth_topics = depth_topics or {
            "head": "/hdas/camera_head/depth/depth_registered",
            "left_wrist": "/hdas/camera_wrist_left/aligned_depth_to_color/image_raw",
            "right_wrist": "/hdas/camera_wrist_right/aligned_depth_to_color/image_raw",
        }
        self._save_action=save_action
        action_topics = action_topics or {
            "left_arm": "/motion_target/target_joint_state_arm_left",
            "right_arm": "/motion_target/target_joint_state_arm_right",
            "torso": "/motion_target/target_joint_state_torso",
            "left_gripper": "/motion_control/position_control_gripper_left",
            "right_gripper": "/motion_control/position_control_gripper_right",
            "mobile_base": "/motion_target/target_speed_chassis",
        }
        all_action_topic_dtypes = {
            "left_arm": JointState,
            "right_arm": JointState,
            "torso": JointState,
            "left_gripper": Float32,
            "right_gripper": Float32,
            "mobile_base": Twist,
        }

        self._joint_state_data: Dict[str, Optional[np.array]] = {
            k: None for k in joint_state_topics
        }
        print("_joint_state_data",self._joint_state_data)
        self._gripper_state_data: Dict[str, Optional[np.array]] = {
            k: None for k in gripper_state_topics
        }
        if self._save_point_cloud:
            self._point_cloud_data: Dict[str, Optional[Dict[str, np.array]]] = {
                k: None for k in point_cloud_topics
            }
        if self._save_odometry:
            self._odom_data: Dict[str, Optional[np.array]] = {
                "position": None,
                "orientation": None,
                "linear_velocity": None,
                "angular_velocity": None,
                "stamp": None,
            }
        if self._save_action:
            self._action_data: Dict[str, Optional[np.array]] = {
                k: None for k in action_topics
            }
        self._joycon_functional_buttons_data = {
            "confirm": None,
            "discard": None,
        }
        self._rgb_data: Dict[str, Optional[np.array]] = (
            {k: None for k in rgb_topics} if self._save_rgbd else None
        )
        self._depth_data: Dict[str, Optional[np.array]] = (
            {k: None for k in depth_topics} if self._save_rgbd else None
        )
        self._cv_bridge = CvBridge() if self._save_rgbd else None

        # ros node initialization
        rospy.init_node("R1DataRecorder", anonymous=True)
        # create the rater
        self._rate = rospy.Rate(record_freq)

        self._joint_state_subs = {
            k: rospy.Subscriber(
                v, JointState, partial(self._update_joint_state_callback, name=k)
            )
            for k, v in joint_state_topics.items()
        }
        print("_joint_state_subs",self._joint_state_subs)
        self._gripper_state_subs = {
            k: rospy.Subscriber(
                v, JointState, partial(self._update_gripper_callback, name=k)
            )
            for k, v in gripper_state_topics.items()
        }
        if self._save_point_cloud:
            self._point_cloud_subs = {
                k: rospy.Subscriber(
                    v, PointCloud2, partial(self._update_pointcloud_callback, name=k)
                )
                for k, v in point_cloud_topics.items()
            }
        if self._save_odometry:
            self._odom_sub = rospy.Subscriber(
                odom_topics, Odometry, self._update_odom_callback
            )
        if self._save_action:
            self._action_subs = {
                k: rospy.Subscriber(
                    v,
                    all_action_topic_dtypes[k],
                    partial(self._update_action_callback, name=k),
                )
                for k, v in action_topics.items()
            }
        self._joycon_functional_buttons_subs = {
            "y": rospy.Subscriber(
                joycon_y_button_topic,
                Bool,
                partial(self._update_joycon_functional_buttons_callback, name="y"),
            ),
            "x": rospy.Subscriber(
                joycon_x_button_topic,
                Bool,
                partial(self._update_joycon_functional_buttons_callback, name="x"),
            ),
            "a": rospy.Subscriber(
                joycon_a_button_topic,
                Bool,
                partial(self._update_joycon_functional_buttons_callback, name="a"),
            ),
        }
        if self._save_rgbd:
            self._rgb_subs = {
                k: rospy.Subscriber(
                    v, Image, partial(self._update_rgb_callback, name=k)
                )
                for k, v in rgb_topics.items()
            }
            self._depth_subs = {
                k: rospy.Subscriber(
                    v, Image, partial(self._update_depth_callback, name=k)
                )
                for k, v in depth_topics.items()
            }

        # wait for the first message
        rospy.loginfo("Waiting for the first message...")
        rospy.wait_for_message(joycon_y_button_topic, Bool)
        rospy.wait_for_message(joycon_x_button_topic, Bool)
        rospy.wait_for_message(joycon_a_button_topic, Bool)
        rospy.loginfo("All joycon functional buttons topics are ready!")
        for topic in joint_state_topics.values():
            rospy.wait_for_message(topic, JointState)
        rospy.loginfo("All joint state topics are ready!")
        for topic in gripper_state_topics.values():
            rospy.wait_for_message(topic, JointState)
        rospy.loginfo("All gripper state topics are ready!")
        if self._save_point_cloud:
            for topic in point_cloud_topics.values():
               rospy.wait_for_message(topic, PointCloud2)
            rospy.loginfo("All point cloud topics are ready!")
        if self._save_odometry:
            rospy.wait_for_message(odom_topics, Odometry)
            rospy.loginfo("Odometry topic is ready!")
        if self._save_action:
            for k, topic in action_topics.items():
               rospy.wait_for_message(topic, all_action_topic_dtypes[k])
            rospy.loginfo("All action topics are ready!")
        if self._save_rgbd:
            for topic in rgb_topics.values():
                rospy.wait_for_message(topic, Image)
            rospy.loginfo("All RGB topics are ready!")
            for topic in depth_topics.values():
                rospy.wait_for_message(topic, Image)
            rospy.loginfo("All depth topics are ready!")
        rospy.loginfo("All TOPICS READY!")

        # thread
        self._data_recording_thread = None
        self._stop_data_recording_event = threading.Event()
        self._obs_buffer_lock = threading.Lock()
        self._action_buffer_lock = threading.Lock()

        # for data recording
        self._joint_state_buffers: Dict[str, Optional[List]] = {
            k: None for k in joint_state_topics
        }
        self._gripper_state_buffers: Dict[str, Optional[List]] = {
            k: None for k in gripper_state_topics
        }
        self._point_cloud_buffers: Dict[str, Optional[Dict[str, List]]] = {
            k: None for k in point_cloud_topics
        }

        self._odom_buffers: Dict[str, Optional[List]] = {
                "position": None,
                "orientation": None,
                "linear_velocity": None,
                "angular_velocity": None,
                "stamp": None,
        }
        self._rgb_buffers: Dict[str, Optional[List]] = (
            {k: None for k in rgb_topics} if self._save_rgbd else None
        )
        self._depth_buffers: Dict[str, Optional[List]] = (
            {k: None for k in depth_topics} if self._save_rgbd else None
        )
        self._action_buffers: Dict[str, Optional[List]] = {
           k: None for k in action_topics
        }
        self._timesteps_buffers = None

    def _update_joint_state_callback(self, js_msg: JointState, name: str):

        self._joint_state_data[name] = {
            "joint_position": np.array([js_msg.position]),
            "joint_velocity": np.array([js_msg.velocity]),
            "joint_effort": np.array([js_msg.effort]),
            "seq": np.array([js_msg.header.seq]),
            "stamp": np.array(
                [js_msg.header.stamp.secs + js_msg.header.stamp.nsecs * 1e-9]
            ),
        }
        #print("_joint_state_data", self._joint_state_data["torso"])

    def _update_gripper_callback(self, gs_msg: JointState, name: str):
        self._gripper_state_data[name] = {
            "gripper_position": np.array([gs_msg.position[0]]),
            "gripper_velocity": np.array([gs_msg.velocity[0]]),
            "gripper_effort": np.array([gs_msg.effort[0]]),
            "seq": np.array([gs_msg.header.seq]),
            "stamp": np.array(
                [gs_msg.header.stamp.secs + gs_msg.header.stamp.nsecs * 1e-9]
            ),
        }

    def _update_pointcloud_callback(self, pcd_msg: PointCloud2, name: str):
        stamp = pcd_msg.header.stamp.secs + pcd_msg.header.stamp.nsecs * 1e-9
        pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msg)
        pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        self._point_cloud_data[name] = {
            "xyz": pcd_xyz,
            "rgb": pcd_rgb,
            "stamp": np.array([stamp]),
        }

    def _update_odom_callback(self, odom_msg: Odometry):
        stamp = odom_msg.header.stamp.secs + odom_msg.header.stamp.nsecs * 1e-9
        position = np.array(
            [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z,
            ]
        )
        orientation = np.array(
            [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ]
        )  # follow PyBullet's quaternion convention
        linear_velocity = np.array(
            [
                odom_msg.twist.twist.linear.x,
                odom_msg.twist.twist.linear.y,
                odom_msg.twist.twist.linear.z,
            ]
        )
        angular_velocity = np.array(
            [
                odom_msg.twist.twist.angular.x,
                odom_msg.twist.twist.angular.y,
                odom_msg.twist.twist.angular.z,
            ]
        )
        self._odom_data["position"] = position[np.newaxis]
        self._odom_data["orientation"] = orientation[np.newaxis]
        self._odom_data["linear_velocity"] = linear_velocity[np.newaxis]
        self._odom_data["angular_velocity"] = angular_velocity[np.newaxis]
        self._odom_data["stamp"] = np.array([stamp])

    def _update_rgb_callback(self, rgb_msg: Image, name: str):
        stamp = rgb_msg.header.stamp.secs + rgb_msg.header.stamp.nsecs * 1e-9
        img = np.asarray(self._cv_bridge.imgmsg_to_cv2(rgb_msg, "rgb8"))[np.newaxis]
        self._rgb_data[name] = {
            "img": img,
            "stamp": np.array([stamp]),
        }

    def _update_depth_callback(self, depth_msg: Image, name: str):
        stamp = depth_msg.header.stamp.secs + depth_msg.header.stamp.nsecs * 1e-9
        depth_image = self._cv_bridge.imgmsg_to_cv2(depth_msg, "32FC1")[np.newaxis]
        self._depth_data[name] = {
            "depth": depth_image,
            "stamp": np.array([stamp]),
        }

    def _update_action_callback(
        self,
        data: Union[JointState, Float32],
        name: str,
    ):
        if isinstance(data, JointState):
            # joint position action
            self._action_data[name] = {
            "joint_position": np.array([data.position]),
            "stamp": np.array(
                [data.header.stamp.secs + data.header.stamp.nsecs * 1e-9]
            ),
        }
        elif isinstance(data, Float32):
            # gripper action
            self._action_data[name] = np.array([data.data])
        elif isinstance(data, Twist):
            # mobile base action
            # we only care about linear xy and angular z
            self._action_data[name] = np.array(
                [[data.linear.x, data.linear.y, data.angular.z]]
            )
        else:
            raise ValueError(f"Unsupported action data type: {type(data)}")

    def _update_joycon_functional_buttons_callback(
        self, data: Bool, name: Literal["x", "y", "a"]
    ):
        self._joycon_functional_buttons_data[name] = data.data

    def _data_recording_loop(self):
        while not rospy.is_shutdown() and not self._stop_data_recording_event.is_set():
            # record when all data are ready
            if (
                all(v is not None for v in self._joint_state_data.values())
                and all(v is not None for v in self._gripper_state_data.values())
                #and all(v is not None for v in self._point_cloud_data.values())
            ):
                with self._obs_buffer_lock:
                    with self._action_buffer_lock:
                        for k, v in self._joint_state_data.items():
                            if self._joint_state_buffers[k] is None:
                                self._joint_state_buffers[k] = [v]
                            else:
                                self._joint_state_buffers[k].append(v)
                        for k, v in self._gripper_state_data.items():
                            if self._gripper_state_buffers[k] is None:
                                self._gripper_state_buffers[k] = [v]
                            else:
                                self._gripper_state_buffers[k].append(v)
                        if self._save_point_cloud:
                            for k, v in self._point_cloud_data.items():
                                if self._point_cloud_buffers[k] is None:
                                    self._point_cloud_buffers[k] = {
                                        "xyz": [v["xyz"]],
                                        "rgb": [v["rgb"]],
                                        "stamp": [v["stamp"]],
                                    }
                                else:
                                    self._point_cloud_buffers[k]["xyz"].append(v["xyz"])
                                    self._point_cloud_buffers[k]["rgb"].append(v["rgb"])
                                    self._point_cloud_buffers[k]["stamp"].append(v["stamp"])
                        if self._save_odometry:
                            for k, v in self._odom_data.items():
                                if self._odom_buffers[k] is None:
                                    self._odom_buffers[k] = [v]
                                else:
                                    self._odom_buffers[k].append(v)
                        if self._save_rgbd:
                            for k, v in self._rgb_data.items():
                                if self._rgb_buffers[k] is None:
                                    self._rgb_buffers[k] = [v]
                                else:
                                    self._rgb_buffers[k].append(v)
                            for k, v in self._depth_data.items():
                                if self._depth_buffers[k] is None:
                                    self._depth_buffers[k] = [v]
                                else:
                                    self._depth_buffers[k].append(v)
                        if self._timesteps_buffers is None:
                            self._timesteps_buffers = [rospy.get_time()]
                        else:
                            self._timesteps_buffers.append(rospy.get_time())
                        if self._save_action:
                            for k, v in self._action_data.items():
                                if self._action_buffers[k] is None:
                                    self._action_buffers[k] = [v]
                                else:
                                    self._action_buffers[k].append(v)
            # run at the desired frequency
            self._rate.sleep()

    def start_data_recording_thread(self):
        if (
            self._data_recording_thread is None
            or not self._data_recording_thread.is_alive()
        ):
            self._stop_data_recording_event.clear()
            self._data_recording_thread = threading.Thread(
                target=self._data_recording_loop
            )
            self._data_recording_thread.start()
            rospy.loginfo("Data recording thread started!")

    def stop_data_recording_thread(self):
        if (
            self._data_recording_thread is not None
            and self._data_recording_thread.is_alive()
        ):
            self._stop_data_recording_event.set()
            self._data_recording_thread.join()
            rospy.loginfo("Data recording thread stopped!")

    def reset_data_recording(self):
        with self._obs_buffer_lock:
            with self._action_buffer_lock:
                self._joint_state_buffers = {k: None for k in self._joint_state_buffers}
                self._gripper_state_buffers = {
                    k: None for k in self._gripper_state_buffers
                }
                if self._save_point_cloud:
                    self._point_cloud_buffers = {k: None for k in self._point_cloud_buffers}
                if self._save_odometry:
                    self._odom_buffers = {k: None for k in self._odom_buffers}
                if self._save_action:
                    self._action_buffers = {k: None for k in self._action_buffers}
                self._timesteps_buffers = None
                if self._save_rgbd:
                    self._rgb_buffers = {k: None for k in self._rgb_buffers}
                    self._depth_buffers = {k: None for k in self._depth_buffers}

    def save_data(self, save_path: str):
        assert save_path.endswith(".h5") or save_path.endswith(
            ".hdf5"
        ), "Only support hdf5 format"
        with self._obs_buffer_lock:
            with self._action_buffer_lock:
                # first print diagnostic statistics so that users can determine to save or discard the data
                print("_joint_state_buffers",self._joint_state_buffers["torso"])
                diag_stat = {
                    "collected_time": time.strftime("%Y-%m-%d-%H-%M-%S"),
                    "horizon": len(self._joint_state_buffers["torso"]),
                }
                # compute recording frequency
                timesteps_data = np.array(self._timesteps_buffers)
                delta_time_avg = np.mean(timesteps_data[1:] - timesteps_data[:-1])
                diag_stat["recording_freq"] = 1 / delta_time_avg
                # compute point cloud frequency
                if self._save_point_cloud:
                    pcd_freqs = {}
                    for k, pcd_buffer in self._point_cloud_buffers.items():
                        stamps = np.concatenate(pcd_buffer["stamp"])
                        delta_time = stamps[1:] - stamps[:-1]
                        pcd_freqs[f"obs_freq/pcd_{k}"] = 1 / np.mean(delta_time)
                    diag_stat.update(pcd_freqs)
                    # compute odom frequency
                if self._save_odometry:
                    stamps = np.concatenate(self._odom_buffers["stamp"])
                    delta_time = stamps[1:] - stamps[:-1]
                    diag_stat["obs_freq/odom"] = 1 / np.mean(delta_time)
                # compute rgbd frequency if enabled
                if self._save_rgbd:
                    rgb_freqs = {}
                    for k, rgb_buffer in self._rgb_buffers.items():
                        stamps = np.concatenate([v["stamp"] for v in rgb_buffer])
                        delta_time = stamps[1:] - stamps[:-1]
                        rgb_freqs[f"obs_freq/rgb_{k}"] = 1 / np.mean(delta_time)
                    diag_stat.update(rgb_freqs)
                    depth_freqs = {}
                    for k, depth_buffer in self._depth_buffers.items():
                        stamps = np.concatenate([v["stamp"] for v in depth_buffer])
                        delta_time = stamps[1:] - stamps[:-1]
                        depth_freqs[f"obs_freq/depth_{k}"] = 1 / np.mean(delta_time)
                    diag_stat.update(depth_freqs)
                # compute joint state frequency
                joint_state_freqs = {}
                for k, js_buffer in self._joint_state_buffers.items():
                    stamps = np.concatenate([v["stamp"] for v in js_buffer])
                    delta_time = stamps[1:] - stamps[:-1]
                    joint_state_freqs[f"obs_freq/joint_state_{k}"] = 1 / np.mean(
                        delta_time
                    )
                diag_stat.update(joint_state_freqs)
                # compute gripper state frequency
                gripper_state_freqs = {}
                for k, gs_buffer in self._gripper_state_buffers.items():
                    stamps = np.concatenate([v["stamp"] for v in gs_buffer])
                    delta_time = stamps[1:] - stamps[:-1]
                    gripper_state_freqs[f"obs_freq/gripper_state_{k}"] = 1 / np.mean(
                        delta_time
                    )
                diag_stat.update(gripper_state_freqs)
                # print diagnostic statistics
                rospy.loginfo("Diagnostic statistics:")
                for k, v in diag_stat.items():
                    rospy.loginfo(f"{k}: {v}")

                joint_state_data = {
                    k: any_concat(v, dim=0)
                    for k, v in self._joint_state_buffers.items()
                }  # dict of (T, 6)
                gripper_state_data = {
                    k: any_concat(v, dim=0)
                    for k, v in self._gripper_state_buffers.items()
                }
                if self._save_point_cloud:
                    pcd_data = {}
                    for k, pcd_buffer in self._point_cloud_buffers.items():
                        # find the max number of points
                        max_pcd_n = max(
                            len(pcd_buffer["xyz"][i]) for i in range(len(pcd_buffer["xyz"]))
                        )
                        # pad to the max number of points
                        padded_pcd_xyz, padded_pcd_rgb, padding_mask = [], [], []
                        for xyz, rgb in zip(pcd_buffer["xyz"], pcd_buffer["rgb"]):
                            padded_pcd_xyz.append(
                                np.concatenate(
                                    [
                                        xyz,
                                        np.zeros(
                                            (max_pcd_n - len(xyz), 3), dtype=xyz.dtype
                                        ),
                                    ],
                                    axis=0,
                                )
                            )
                            padded_pcd_rgb.append(
                                np.concatenate(
                                    [
                                        rgb,
                                        np.zeros(
                                            (max_pcd_n - len(rgb), 3), dtype=rgb.dtype
                                        ),
                                    ],
                                    axis=0,
                                )
                            )
                            padding_mask.append(
                                np.concatenate(
                                    [
                                        np.ones(len(xyz), dtype=bool),
                                        np.zeros(max_pcd_n - len(xyz), dtype=bool),
                                    ],
                                    axis=0,
                                )
                            )
                        pcd_data[k] = {
                            "xyz": np.stack(padded_pcd_xyz, axis=0),  # (T, N_max, 3)
                            "rgb": np.stack(padded_pcd_rgb, axis=0),  # (T, N_max, 3)
                            "padding_mask": np.stack(padding_mask, axis=0),  # (T, N_max)
                        }
                if self._save_odometry:
                    odom_data = {
                        k: any_concat(v, dim=0) for k, v in self._odom_buffers.items()
                    }
                if self._save_rgbd:
                    rgb_data = {
                        k: any_concat(v, dim=0) for k, v in self._rgb_buffers.items()
                    }
                    depth_data = {
                        k: any_concat(v, dim=0) for k, v in self._depth_buffers.items()
                    }
                if self._save_action:
                    action_data = {
                        k: any_concat(v, dim=0) for k, v in self._action_buffers.items()
                    }
                # prompt users to save or discard the data
                rospy.loginfo(
                    """Press [Y] to confirm saving the data, [X] to discard"""
                )
                while True:
                    proceed = False
                    if (
                        self._joycon_functional_buttons_data["y"]
                        and not self._joycon_functional_buttons_data["x"]
                    ):
                        proceed = True
                        break
                    elif (
                        not self._joycon_functional_buttons_data["y"]
                        and self._joycon_functional_buttons_data["x"]
                    ):
                        break
                    elif (
                        self._joycon_functional_buttons_data["y"]
                        and self._joycon_functional_buttons_data["x"]
                    ):
                        rospy.loginfo(
                            "Both confirm and discard buttons are pressed. Please press only one button."
                        )
                    else:
                        time.sleep(0.1)

                if proceed:
                    # save data
                    f = h5py.File(save_path, "w")
                    for k, v in diag_stat.items():
                        f.attrs[k] = v
                    obs_grp = f.create_group("obs")
                    for ks, vs in joint_state_data.items():
                        for k, v in vs.items():
                            obs_grp.create_dataset(f"joint_state/{ks}/{k}", data=v)
                    for ks, vs in gripper_state_data.items():
                        for k, v in vs.items():
                            obs_grp.create_dataset(f"gripper_state/{ks}/{k}", data=v)
                    if self._save_point_cloud:
                        for ks, vs in pcd_data.items():
                            for k, v in vs.items():
                                obs_grp.create_dataset(f"point_cloud/{ks}/{k}", data=v)
                    if self._save_odometry:
                        for k, v in odom_data.items():
                            obs_grp.create_dataset(f"odom/{k}", data=v)
                    if self._save_rgbd:
                        for ks, vs in rgb_data.items():
                            for k, v in vs.items():
                                obs_grp.create_dataset(f"rgb/{ks}/{k}", data=v)
                        for ks, vs in depth_data.items():
                            for k, v in vs.items():
                                obs_grp.create_dataset(f"depth/{ks}/{k}", data=v)
                    if self._save_action:
                        action_grp = f.create_group("action")
                        for k, v in action_data.items():
                            action_grp.create_dataset(k, data=v)
                    f.close()
                    rospy.loginfo(
                        f"One trajectory recorded. Data saved to {save_path}."
                    )
                else:
                    rospy.loginfo("Data discarded.")
        self.reset_data_recording()

    def get_functional_button(self, button_name: Literal["x", "y", "a"]) -> bool:
        return self._joycon_functional_buttons_data[button_name]

    def close(self):
        self.stop_data_recording_thread()
        rospy.signal_shutdown("Data recorder closed!")

    def __del__(self):
        self.close()
