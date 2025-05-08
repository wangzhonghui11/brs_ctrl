import sys, select
import os
import time
import argparse
import numpy as np
import rospy
from sympy import false

from brs_ctrl.data_recorder import R1DataRecorder


neutral_left_arm_qs = np.array([1.56, 2.94, -2.54, 0, 0, 0])
neutral_right_arm_qs = np.array([-1.56, 2.94, -2.54, 0, 0, 0])


def collect_data(args):
    record_freq = args.record_freq
    data_folder = args.data_folder
    os.makedirs(data_folder, exist_ok=True)

    data_recorder = R1DataRecorder(record_freq=record_freq, save_rgbd=True,save_odometry=false,save_point_cloud=false,_save_action=false)
    data_recorder.start_data_recording_thread()
    while not rospy.is_shutdown():
        rospy.loginfo("First press [A] to start recording data.")
        while True:
            if data_recorder.get_functional_button("a"):
                break
            elif sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                exit()
            time.sleep(0.1)
        rospy.loginfo(
            "Start recording data. At any time, press [A] again to stop recording."
        )
        time.sleep(1)
        data_recorder.reset_data_recording()
        while True:
            if data_recorder.get_functional_button("a"):
                break
            time.sleep(0.1)
        data_recorder.save_data(
            os.path.join(
                data_folder, f"collected_data-{time.strftime('%Y-%m-%d-%H-%M-%S')}.h5"
            )
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--record_freq", type=int, default=10)
    args.add_argument(
        "--data_folder",
        type=str,
        default=f"collected_data_{time.strftime('%Y-%m-%d-%H-%M-%S')}",
    )
    args = args.parse_args()
    collect_data(args)
