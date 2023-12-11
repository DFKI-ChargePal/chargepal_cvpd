from __future__ import annotations
import yaml
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# typing
from typing import Any
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


def load_yaml(file_path: Path) -> dict[str, Any]:
    with file_path.open("r") as filestream:
        try:
            yaml_dict: dict[str, Any] = yaml.safe_load(filestream)
        except Exception as e:
            raise RuntimeError(f"Error while reading {file_path.name} configuration. {e}")
    return yaml_dict


def dump_yaml(data: dict[str, Any], file_path: Path) -> None:
    with file_path.open('wt') as fs:
        yaml.safe_dump(data, fs)
