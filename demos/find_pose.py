# global
import argparse
import cvpd as pd
import camera_kit as ck
from pathlib import Path

# typing
from argparse import Namespace


_parent_dir = Path(__file__).absolute().parent
_cc_path = 'camera_info/build_in/calibration/coefficients.toml'


def find_pose(opt: Namespace) -> None:

    with ck.camera_manager('build_in') as cam:
        cam.load_coefficients(_parent_dir.joinpath(_cc_path))
        dtt = pd.factory.get_detector(_parent_dir.joinpath('dtt_config', opt.config_file))
        dtt.register_camera(cam)
        while not ck.user.stop():
            _, _ = dtt.find_pose(render=True)


if __name__ == '__main__':
    des = """ Demo to find the object pose """
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('config_file', type=str, help='Configuration file describing the pattern')
    # Parse input arguments
    args = parser.parse_args()
    find_pose(args)
