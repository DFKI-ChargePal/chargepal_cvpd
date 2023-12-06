from __future__ import annotations
# global
import logging
import cv2 as cv
import numpy as np

# local
from cvpd.geo_pattern._aruco_types import ARUCO_DICT

# typing
from numpy import typing as npt


LOGGER = logging.getLogger(__name__)


class ArucoMarker:

    def __init__(self,
                 aruco_id: int,
                 aruco_size_mm: int,
                 aruco_type: str,
                 marker_offset_xyz: npt.ArrayLike | None = None,
                 marker_offset_wxyz: npt.ArrayLike | None = None,
                 ) -> None:
        self.ar_type_dict = ARUCO_DICT.get(aruco_type)
        if self.ar_type_dict is None:
            error_msg = f"No AR tag with key '{aruco_type}' found"
            LOGGER.error(error_msg)
            raise KeyError(error_msg)
        self.aruco_dict = cv.aruco.getPredefinedDictionary(self.ar_type_dict)
        self.cv_detector = cv.aruco.ArucoDetector(self.aruco_dict)
        # Check if aruco_id is valid
        if 0 <= aruco_id < self.id_range:
            self.aruco_id = aruco_id
        else:
            raise ValueError(f"Given id of ArUco marker '{aruco_id}' not in valid range 0...{self.id_range - 1}")
        # Store offset as numpy array
        self.offset_p = np.array(marker_offset_xyz)
        self.offset_q = np.array(marker_offset_wxyz)
        # added object points
        ar_size_m_2 = aruco_size_mm / 2 / 1000
        self.obj_pts_marker = np.array([
            [-ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, -ar_size_m_2, 0.0],
            [-ar_size_m_2, -ar_size_m_2, 0.0],
        ], dtype=np.float_)

    @property
    def id_range(self) -> int:
        return int(self.aruco_dict.bytesList.shape[0])
