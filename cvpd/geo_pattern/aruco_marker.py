from __future__ import annotations
# global
import logging
import cv2 as cv
import numpy as np

# local
from cvpd.geo_pattern._aruco_types import ARUCO_DICT


LOGGER = logging.getLogger(__name__)


class ArucoMarker:

    def __init__(self, aruco_id: int, aruco_size_mm: int, aruco_type: str) -> None:
        # TODO: Check if aruco_id is valid
        self.aruco_id = aruco_id
        self.ar_type_dict = ARUCO_DICT.get(aruco_type)
        if self.ar_type_dict is None:
            error_msg = f"No AR tag with key '{aruco_type}' found"
            LOGGER.error(error_msg)
            raise KeyError(error_msg)
        self.aruco_dict = cv.aruco.getPredefinedDictionary(self.ar_type_dict)
        self.cv_detector = cv.aruco.ArucoDetector(self.aruco_dict)
        # added object points
        ar_size_m_2 = aruco_size_mm / 2 / 1000
        self.obj_pts_marker = np.array([
            [-ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, -ar_size_m_2, 0.0],
            [-ar_size_m_2, -ar_size_m_2, 0.0],
        ], dtype=np.float_)
