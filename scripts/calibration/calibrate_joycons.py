from pyjoycon import JoyCon, get_R_id, get_L_id
import sys, select, os
import time
import numpy as np
import yaml
from brs_ctrl.asset_root import ASSET_ROOT


def calibrate():
    pkg_parent_dir = os.path.dirname(ASSET_ROOT)
    save_fpath = os.path.join(
        pkg_parent_dir, "brs_ctrl", "joylo", "joycon", "joycon_calibration.yaml"
    )

    jc_id_left = get_L_id()
    jc_id_right = get_R_id()
    assert jc_id_left[0] is not None, "Left joycon not found"
    assert jc_id_right[0] is not None, "Right joycon not found"
    jc_left = JoyCon(*jc_id_left)
    jc_right = JoyCon(*jc_id_right)
    time.sleep(1)

    input(
        "Put both joycons in a neutral position and press ENTER to start initial-pose calibration."
    )
    left_horizontal_range = [np.inf, -np.inf]
    left_vertical_range = [np.inf, -np.inf]
    right_horizontal_range = [np.inf, -np.inf]
    right_vertical_range = [np.inf, -np.inf]
    st_time = time.time()
    while True:
        init_left_horizontal_value = jc_left.get_stick_left_horizontal()
        init_left_vertical_value = jc_left.get_stick_left_vertical()
        init_right_horizontal_value = jc_right.get_stick_right_horizontal()
        init_right_vertical_value = jc_right.get_stick_right_vertical()

        left_horizontal_range[0] = min(
            left_horizontal_range[0], init_left_horizontal_value
        )
        left_horizontal_range[1] = max(
            left_horizontal_range[1], init_left_horizontal_value
        )
        left_vertical_range[0] = min(left_vertical_range[0], init_left_vertical_value)
        left_vertical_range[1] = max(left_vertical_range[1], init_left_vertical_value)
        right_horizontal_range[0] = min(
            right_horizontal_range[0], init_right_horizontal_value
        )
        right_horizontal_range[1] = max(
            right_horizontal_range[1], init_right_horizontal_value
        )
        right_vertical_range[0] = min(
            right_vertical_range[0], init_right_vertical_value
        )
        right_vertical_range[1] = max(
            right_vertical_range[1], init_right_vertical_value
        )

        if time.time() - st_time > 3:
            break

    joystick_limits = {
        "left": {
            "horizontal": {
                "center_range": left_horizontal_range,
                "limits": [init_left_horizontal_value, init_left_horizontal_value],
            },
            "vertical": {
                "center_range": left_vertical_range,
                "limits": [init_left_vertical_value, init_left_vertical_value],
            },
        },
        "right": {
            "horizontal": {
                "center_range": right_horizontal_range,
                "limits": [init_right_horizontal_value, init_right_horizontal_value],
            },
            "vertical": {
                "center_range": right_vertical_range,
                "limits": [init_right_vertical_value, init_right_vertical_value],
            },
        },
    }
    input(
        "Move joycon joysticks in a circular motion. First press ENTER to start. Then press ENTER to quit calibration."
    )
    while True:
        # Update limits
        left_horizontal_value = jc_left.get_stick_left_horizontal()
        left_vertical_value = jc_left.get_stick_left_vertical()
        right_horizontal_value = jc_right.get_stick_right_horizontal()
        right_vertical_value = jc_right.get_stick_right_vertical()
        for val, side, direction in zip(
            (
                left_horizontal_value,
                left_vertical_value,
                right_horizontal_value,
                right_vertical_value,
            ),
            ("left", "left", "right", "right"),
            ("horizontal", "vertical", "horizontal", "vertical"),
        ):
            limits = joystick_limits[side][direction]["limits"]
            limits[0] = min(limits[0], val)
            limits[1] = max(limits[1], val)
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            break
    # Save output
    with open(save_fpath, "w+") as f:
        yaml.dump({"joystick": joystick_limits}, f)


if __name__ == "__main__":
    calibrate()
