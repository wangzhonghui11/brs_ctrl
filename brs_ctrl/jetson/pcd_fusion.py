from typing import Dict, Optional, List
from functools import partial

import rospy
from sensor_msgs.msg import JointState, PointCloud2
import numpy as np
import ros_numpy

try:
    import fpsample
except ImportError:
    print(
        "[WARNING] fpsample not found. PCDFusionPublisher won't work if set use_fps=True"
    )

from brs_ctrl.robot_interface.utils import get_xyz_points
from brs_ctrl.kinematics import R1Kinematics


def merge_xyz_rgb(xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    rgb_packed = np.asarray(
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32),
        dtype=np.uint32,
    ).view(np.float32)

    structured_array = np.zeros(
        xyz.shape[0],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.float32),
        ],
    )
    structured_array["x"] = xyz[:, 0]
    structured_array["y"] = xyz[:, 1]
    structured_array["z"] = xyz[:, 2]
    structured_array["rgb"] = rgb_packed

    return structured_array


class PCDFusionPublisher:
    torso_joint_high = np.array([1.8326, 2.5307, 1.8326, 3.0543])
    torso_joint_low = np.array([-1.1345, -2.7925, -2.0944, -3.0543])
    left_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    left_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])
    right_arm_joint_high = np.array([2.8798, 3.2289, 0, 2.8798, 1.6581, 2.8798])
    right_arm_joint_low = np.array([-2.8798, 0, -3.3161, -2.8798, -1.6581, -2.8798])

    def __init__(
        self,
        downsample_N: int,
        use_fps: bool,
        publish_freq: int,
        spatial_cutoff: List[float],
        fps_h: int = 7,
        joint_state_topics: Optional[Dict[str, str]] = None,
        point_cloud_topics: Optional[Dict[str, str]] = None,
        camera2link_names: Optional[Dict[str, str]] = None,
    ):
        joint_state_topics = joint_state_topics or {
            "left_arm": "/hdas/feedback_arm_left",
            "right_arm": "/hdas/feedback_arm_right",
            "torso": "/hdas/feedback_torso",
        }
        point_cloud_topics = point_cloud_topics or {
            "left_wrist": "/zed_multi_cams/zed2_left_wrist/zed_nodelet_left_wrist/point_cloud/cloud_registered",
            "right_wrist": "/zed_multi_cams/zed2_right_wrist/zed_nodelet_right_wrist/point_cloud/cloud_registered",
            "head": "/zed_multi_cams/zed2_head/zed_nodelet_head/point_cloud/cloud_registered",
        }
        camera2link_names = camera2link_names or {
            "left_wrist": "left_wrist_camera",
            "right_wrist": "right_wrist_camera",
            "head": "head_camera",
        }
        self._camera2link_names = camera2link_names
        self._xmin, self._xmax = spatial_cutoff[0], spatial_cutoff[1]
        self._ymin, self._ymax = spatial_cutoff[2], spatial_cutoff[3]
        self._zmin, self._zmax = spatial_cutoff[4], spatial_cutoff[5]
        self._downsample_N = downsample_N
        self._use_fps = use_fps
        self._fps_h = fps_h

        self._kin_model = R1Kinematics()

        self._joint_state_data: Dict[str, Optional[np.array]] = {
            k: None for k in joint_state_topics
        }
        self._point_cloud_data: Dict[str, Optional[Dict[str, np.array]]] = {
            k: None for k in point_cloud_topics
        }

        # ros node initialization
        rospy.init_node("fused_pcd_publisher_jetson", anonymous=True)
        self._rate = rospy.Rate(publish_freq)

        self._joint_state_subs = {
            k: rospy.Subscriber(
                v, JointState, partial(self._update_joint_state_callback, name=k)
            )
            for k, v in joint_state_topics.items()
        }
        self._point_cloud_subs = {
            k: rospy.Subscriber(
                v, PointCloud2, partial(self._update_pointcloud_callback, name=k)
            )
            for k, v in point_cloud_topics.items()
        }
        self._fused_pcd_pub = rospy.Publisher(
            "/r1_jetson/fused_pcd", PointCloud2, queue_size=1
        )

        for topic in point_cloud_topics.values():
            rospy.wait_for_message(topic, PointCloud2)
        for topic in joint_state_topics.values():
            rospy.wait_for_message(topic, JointState)

    def _update_joint_state_callback(self, js_msg: JointState, name: str):
        self._joint_state_data[name] = np.array(js_msg.position[:6])

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

    def _fused_pcd_pub_callback(self):
        link2base = self._kin_model.get_link_poses_in_base_link(
            curr_left_arm_joint=self._joint_state_data["left_arm"],
            curr_right_arm_joint=self._joint_state_data["right_arm"],
            curr_torso_joint=self._joint_state_data["torso"],
        )  # (4, 4)
        cam2base = {
            k: link2base[self._camera2link_names[k]] for k in self._point_cloud_data
        }
        transformed_pcd_xyz, pcd_rgb = [], []
        for k, pcd_data in self._point_cloud_data.items():
            xyz = pcd_data["xyz"]
            xyz = np.concatenate(
                [xyz, np.ones((xyz.shape[0], 1))], axis=-1
            )  # (N_points, 4)
            transform = cam2base[k]  # (4, 4)
            transformed_pcd_xyz.append((transform @ xyz.T).T[..., :3])
            pcd_rgb.append(pcd_data["rgb"])
        fused_pcd_xyz = np.concatenate(transformed_pcd_xyz, axis=0)  # (N_fused, 3)
        fused_pcd_rgb = np.concatenate(pcd_rgb, axis=0)  # (N_fused, 3)
        # spatial cutoff
        x_mask = np.logical_and(
            fused_pcd_xyz[:, 0] >= self._xmin, fused_pcd_xyz[:, 0] <= self._xmax
        )
        y_mask = np.logical_and(
            fused_pcd_xyz[:, 1] >= self._ymin, fused_pcd_xyz[:, 1] <= self._ymax
        )
        z_mask = np.logical_and(
            fused_pcd_xyz[:, 2] >= self._zmin, fused_pcd_xyz[:, 2] <= self._zmax
        )
        cutoff_mask = np.logical_and(x_mask, np.logical_and(y_mask, z_mask))
        fused_pcd_xyz = fused_pcd_xyz[cutoff_mask]
        fused_pcd_rgb = fused_pcd_rgb[cutoff_mask]
        # downsample
        if len(fused_pcd_xyz) > self._downsample_N:
            if self._use_fps:
                sampling_idx = fpsample.bucket_fps_kdline_sampling(
                    fused_pcd_xyz, n_samples=self._downsample_N, h=self._fps_h
                )
            else:
                sampling_idx = np.random.permutation(len(fused_pcd_xyz))[
                    : self._downsample_N
                ]
            fused_pcd_xyz = fused_pcd_xyz[sampling_idx]
            fused_pcd_rgb = fused_pcd_rgb[sampling_idx]
        # publish
        cloud_array = merge_xyz_rgb(fused_pcd_xyz, fused_pcd_rgb)
        pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
            cloud_array, rospy.Time.now(), "base_link"
        )
        self._fused_pcd_pub.publish(pointcloud_msg)
        self._rate.sleep()

    def run(self):
        while not rospy.is_shutdown():
            self._fused_pcd_pub_callback()
        rospy.spin()
