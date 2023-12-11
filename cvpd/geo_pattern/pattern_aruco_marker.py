from __future__ import annotations
# global
import logging
import cv2 as cv
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

# local
from cvpd.utilities import load_yaml, dump_yaml
from cvpd.geo_pattern._aruco_types import ARUCO_DICT

# typing
from typing import Sequence


LOGGER = logging.getLogger(__name__)


class ArucoMarker:

    @dataclass
    class Config:
        marker_id: int
        marker_size: int
        marker_type: str
        marker_offset: dict[str, Sequence[float]]

    def __init__(self, config_fp: Path):
        self.cfg_fp = config_fp
        self.cfg = ArucoMarker.Config(**load_yaml(config_fp))
        self.ar_type_dict = ARUCO_DICT.get(self.cfg.marker_type)
        if self.ar_type_dict is None:
            error_msg = f"No AR tag with key '{self.cfg.marker_type}' found"
            LOGGER.error(error_msg)
            raise KeyError(error_msg)
        self.aruco_dict = cv.aruco.getPredefinedDictionary(self.ar_type_dict)
        # Check if aruco_id is valid
        if 0 <= self.cfg.marker_id < self.id_range:
            self.aruco_id = self.cfg.marker_id
        else:
            raise ValueError(f"Given id of ArUco marker '{self.cfg.marker_id}' "
                             f"not in valid range 0...{self.id_range - 1}")
        self.offset_p = np.reshape(self.cfg.marker_offset['xyz'], 3)
        self.offset_q = np.reshape(self.cfg.marker_offset['xyzw'], 4)
        # added object points
        ar_size_m_2 = self.cfg.marker_size / 2 / 1000
        self.obj_pts_marker = np.array([
            [-ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, -ar_size_m_2, 0.0],
            [-ar_size_m_2, -ar_size_m_2, 0.0],
        ], dtype=np.float_)

    @property
    def id_range(self) -> int:
        return int(self.aruco_dict.bytesList.shape[0])

    def adjust_configuration(self, offset_p: Sequence[float] | None, offset_q: Sequence[float] | None) -> None:
        if offset_p is not None:
            if len(offset_p) != 3:
                raise ValueError(f"Expect position vector with three values. Got {len(offset_p)} values.")
            off_p = list(offset_p)
        else:
            off_p = self.offset_p.tolist()

        if offset_q is not None:
            if len(offset_q) != 4:
                raise ValueError(f"Expect quaternion vector with four values. Got {len(offset_q)} values.")
            off_q = list(offset_q)
        else:
            off_q = self.offset_q.tolist()

        adj_cfg = ArucoMarker.Config(
            marker_id=self.cfg.marker_id,
            marker_size=self.cfg.marker_size,
            marker_type=self.cfg.marker_type,
            marker_offset={
                'xyz': off_p,
                'xyzw': off_q,
            }
        )
        # Store adjusted configuration
        adj_cfg_fp = self.cfg_fp.parent.joinpath(self.cfg_fp.stem + '_adj' + self.cfg_fp.suffix)
        dump_yaml(asdict(adj_cfg), adj_cfg_fp)
