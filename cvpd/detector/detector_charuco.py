from __future__ import annotations

# global
import cv2 as cv
import numpy as np
from pathlib import Path
from camera_kit import converter
from camera_kit import DetectorBase

# local
from cvpd.geo_pattern.charuco_board import CharucoBoard

# typing
from camera_kit.core import PosOrinType


class CharucoDetector(DetectorBase):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Get charuco configuration
        marker_size = self.config['marker_size']
        marker_type = self.config['marker_type']
        checker_size = self.config['checker_size']
        checker_grid_size = self.config['checker_grid_size']
        self.board = CharucoBoard(marker_size, marker_type, checker_size, checker_grid_size)
        self.cv_detector = cv.aruco.ArucoDetector(self.board.aruco_dict)

    def _find_pose(self) -> tuple[bool, PosOrinType]:
        """ Finding the pose of a charuco board

        Returns:
            (True if pose was found; Pose containing position [xyz] and quaternion [wxyz] vector
        """
        # Initialize return variables with default values
        found = False
        p, q = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
        img = self.camera.get_color_frame()

        # load camera matrix and distortion coefficients from camera
        cam_intrinsic = self.camera.cc.intrinsic
        cam_distortion = self.camera.cc.distortion

        marker_corners, marker_ids, _ = self.cv_detector.detectMarkers(img)

        if marker_ids is not None and len(marker_ids) >= 4:
            obj_p, img_p = self.board.cv_board.matchImagePoints(marker_corners, np.array(marker_ids))
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
                p, q = converter.cv_to_pq(r_vec, t_vec)
        return found, (p, q)
