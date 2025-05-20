import h5py
import json
import numpy as np
import os
import shutil
from pathlib import Path
import cv2
from PIL import Image
import io


def convert_hdf5_to_nedf(hdf5_path: str, output_dir: str, video_id: str = "sample_001"):
    """
    将特定结构的HDF5文件转换为NEDF格式的目录结构

    参数:
        hdf5_path: 输入的HDF5文件路径
        output_dir: 输出目录路径
        video_id: 视频ID (如 sample_001)
    """
    # 创建输出目录结构
    output_path = Path(output_dir) / video_id
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(hdf5_path, 'r') as hdf:
            # 1. 创建metadata.json
            create_metadata(hdf, output_path)

            # 2. 处理观测数据
            if 'observations' in hdf:
                obs_group = hdf['observations']

                # 处理图像数据
                if 'images' in obs_group:
                    img_group = obs_group['images']

                    # 高分辨率相机
                    if 'cam_high' in img_group:
                        process_jpeg_camera_data(img_group['cam_high'], output_path, 'cam_high')

                    # 左腕相机
                    if 'cam_left_wrist' in img_group:
                        process_jpeg_camera_data(img_group['cam_left_wrist'], output_path, 'cam_left_wrist')

                    # 右腕相机
                    if 'cam_right_wrist' in img_group:
                        process_jpeg_camera_data(img_group['cam_right_wrist'], output_path, 'cam_right_wrist')

                # 处理关节位置数据
                if 'qpos' in obs_group:
                    save_timeseries(obs_group['qpos'], output_path / 'lowdim' / 'joint.npz')

                # 处理关节速度数据
                if 'qvel' in obs_group:
                    save_timeseries(obs_group['qvel'], output_path / 'lowdim' / 'joint_vel.npz')

                # 处理effort数据
                if 'effort' in obs_group:
                    save_timeseries(obs_group['effort'], output_path / 'lowdim' / 'force_torque.npz')

            # 3. 处理动作数据
            if 'action' in hdf:
                save_timeseries(hdf['action'], output_path / 'lowdim' / 'ee_command.npz')

            # 4. 处理指令数据
            if 'instruction' in hdf:
                with open(output_path / 'instruction.txt', 'w') as f:
                    if isinstance(hdf['instruction'][()], bytes):
                        f.write(hdf['instruction'][()].decode('utf-8'))
                    else:
                        f.write(str(hdf['instruction'][()]))

            print(f"成功转换HDF5文件到NEDF格式，输出目录: {output_path}")

    except Exception as e:
        print(f"转换过程中发生错误: {str(e)}")
        shutil.rmtree(output_path, ignore_errors=True)
        raise


def create_metadata(hdf: h5py.File, output_path: Path):
    """创建metadata.json文件"""
    metadata = {
        "task_name": "unknown_task",
        "camera_info": {},
        "main_camera": "cam_high",  # 默认主相机
        "robot_type": "dual_arm",  # 假设有左右腕相机，所以是双臂
        "arm_base_to_world_transform": [],
        "end_effector": "gripper",
        "extra_keys": ["joint_pos", "joint_vel", "force_torque"]
    }

    # 填充相机信息
    if 'observations/images' in hdf:
        img_group = hdf['observations/images']

        if 'cam_high' in img_group:
            metadata['camera_info']['cam_high'] = 'global'

        if 'cam_left_wrist' in img_group:
            metadata['camera_info']['cam_left_wrist'] = 'inhand_left'

        if 'cam_right_wrist' in img_group:
            metadata['camera_info']['cam_right_wrist'] = 'inhand_right'

    # 写入文件
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def process_jpeg_camera_data(jpeg_dataset: h5py.Dataset, output_path: Path, cam_name: str):
    """处理JPEG格式的相机数据"""
    cam_path = output_path / f'camera_{cam_name}'
    cam_path.mkdir(exist_ok=True)

    color_path = cam_path / 'color'
    color_path.mkdir(exist_ok=True)

    # 保存JPEG图像
    for i in range(len(jpeg_dataset)):
        jpeg_bytes = jpeg_dataset[i]

        # 使用PIL直接保存JPEG数据
        img = Image.open(io.BytesIO(jpeg_bytes))
        img.save(color_path / f"{i}.jpg", "JPEG")

    # 添加默认相机参数
    default_intrinsic = np.array([
        [600, 0, 320],
        [0, 600, 240],
        [0, 0, 1]
    ])
    np.save(cam_path / 'intrinsic.npy', default_intrinsic)

    default_extrinsic = np.eye(4).tolist()
    with open(cam_path / 'extrinsic.json', 'w') as f:
        json.dump(default_extrinsic, f)


def save_timeseries(data: h5py.Dataset, output_path: Path):
    """保存时间序列数据为npz文件"""
    output_path.parent.mkdir(exist_ok=True, parents=True)

    # 创建以时间步为键的字典
    data_dict = {str(i): data[i] for i in range(data.shape[0])}
    np.savez(output_path, **data_dict)


if __name__ == "__main__":
    # 使用示例
    convert_hdf5_to_nedf(
        hdf5_path="episode_0.hdf5",
        output_dir="output_nedf",
        video_id="sample_001"
    )