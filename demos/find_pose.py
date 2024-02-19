# libs
import argparse
import cvpd as pd
import camera_kit as ck
from pathlib import Path
from time import perf_counter

# typing
from argparse import Namespace

_log_freq = 10
_parent_dir = Path(__file__).absolute().parent
_cc_path = 'camera_info/build_in/calibration/coefficients.toml'


def find_pose(opt: Namespace) -> None:

    with ck.camera_manager('build_in') as cam:
        cam.load_coefficients(_parent_dir.joinpath(_cc_path))
        dtt = pd.factory.create(_parent_dir.joinpath('dtt_config', opt.config_file))
        dtt.register_camera(cam)
        log_interval = 1.0 / _log_freq
        _t_start = perf_counter()
        while not ck.user.stop():
            found, T_cam2obj = dtt.find_pose(render=True)
            if perf_counter() - _t_start > log_interval and found:
                print(f"Transformation Camera - Object: {T_cam2obj.t.tolist()} {T_cam2obj.eulervec().tolist()}")
                _t_start = perf_counter()


if __name__ == '__main__':
    des = """ Demo to find the object pose """
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('config_file', type=str, help='Configuration file describing the pattern')
    # Parse input arguments
    args = parser.parse_args()
    find_pose(args)
