import logging
import numpy as np
from scipy.spatial.transform import Rotation as R

# typing
from numpy import typing as npt


def rotate_rot_vec(r_vec: npt.NDArray[np.float_], axis: str = 'x', ang: float = 0.0) -> npt.NDArray[np.float_]:
    # Build rotation vector
    axis = axis.lower()
    if axis == 'x':
        r_vec23 = (ang, 0.0, 0.0)
    elif axis == 'y':
        r_vec23 = (0.0, ang, 0.0)
    elif axis == 'z':
        r_vec23 = (0.0, 0.0, ang)
    else:
        raise ValueError(f"Unknown rotation axis: {axis}. Use 'x', 'y' or 'z'.")
    return np.reshape((R.from_rotvec(np.squeeze(r_vec)) * R.from_rotvec(r_vec23)).as_rotvec(), [3, 1])
