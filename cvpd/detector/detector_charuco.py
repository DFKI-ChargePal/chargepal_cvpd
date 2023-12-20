from __future__ import annotations

# global
import cv2 as cv
import numpy as np
from pathlib import Path
from camera_kit import converter

# local
from cvpd.detector.detector_abc import DetectorABC
from cvpd.config.config_charuco import Charuco

# typing
from camera_kit.core import PosOrinType


class CharucoDetector(DetectorABC):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Get charuco configuration
        self.config_charuco = Charuco(**self.config_dict)
        self.config.add(self.config_charuco)
        # Create OpenCV interface
        self.cv_board = cv.aruco.CharucoBoard(
            tuple(self.config_charuco.checker_grid_size),
            self.config_charuco.checker_size_m,
            self.config_charuco.marker_size_m,
            self.config_charuco.cv_aruco_dict
        )
        self.cv_detector = cv.aruco.ArucoDetector(self.config_charuco.cv_aruco_dict)

    def _find_pose(self) -> tuple[bool, PosOrinType]:
        """ Finding the pose of a charuco board

        Returns:
            (True if pose was found; Pose containing position [xyz] and quaternion [wxyz] vector
        """
        # Initialize return variables with default values
        found = False
        pq = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        img = self.camera.get_color_frame()

        # load camera matrix and distortion coefficients from camera
        cam_intrinsic = self.camera.cc.intrinsic
        cam_distortion = self.camera.cc.distortion

        marker_corners, marker_ids, _ = self.cv_detector.detectMarkers(img)

        if marker_ids is not None and len(marker_ids) >= 4:
            obj_p, img_p = self.cv_board.matchImagePoints(marker_corners, np.array(marker_ids))
            found, r_vec, t_vec = cv.solvePnP(obj_p, img_p, cam_intrinsic, cam_distortion, flags=cv.SOLVEPNP_IPPE)
            if found:
                r_vec, t_vec = cv.solvePnPRefineLM(
                    objectPoints=obj_p,
                    imagePoints=img_p,
                    cameraMatrix=cam_intrinsic,
                    distCoeffs=cam_distortion,
                    rvec=r_vec,
                    tvec=t_vec,
                    criteria=(cv.TermCriteria_EPS + cv.TermCriteria_COUNT, 30, 0.001)
                )
                pq = converter.cv_to_pq(r_vec, t_vec)
                pq = self.config_offset.apply_offset(pq)
        return found, pq
