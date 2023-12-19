from __future__ import annotations

# global
import cv2 as cv
import numpy as np
from pathlib import Path
from camera_kit import converter

# local
from cvpd.detector.detector import Detector
from cvpd.detector.helper import ArucoOpenCV
from cvpd.config.config_aruco_pattern import ArucoPattern

# typing
from camera_kit.core import PosOrinType


class ArucoPatternDetector(Detector):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Create configuration
        self.config_pattern = ArucoPattern(**self.config_dict)
        self.config.add(self.config_pattern)
        # Set OpenCV detector up
        self.cv_detector = ArucoOpenCV(cv.aruco.ArucoDetector(self.config_pattern.cv_aruco_dict))
        # Check if aruco ids are valid
        id_range = self.config_pattern.id_range
        for m_id in self.config_pattern.marker_ids:
            if 0 <= m_id < id_range:
                self.aruco_id = m_id
            else:
                raise ValueError(f"Given id of ArUco marker '{m_id}' not in valid range 0...{id_range - 1}")

    def _find_pose(self) -> tuple[bool, PosOrinType]:
        """ Finding the pose the pattern layout describes

        Returns:
            (True if pose was found; Pose containing position [xyz] and quaternion [xyzw] vector)
        """
        # Initialize return variables with default values
        found = False
        pq = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        img = self.camera.get_color_frame()
        # Get all markers on image
        marker_ids, marker_corners = self.cv_detector.find_group_marker_corners(
            img, list(self.config_pattern.marker_ids))

        # Check how many markers are found:
        found_marker_ids = set(marker_ids) & self.config_pattern.marker_ids
        if len(found_marker_ids) >= 4:
            obj_points_list = []
            img_points_list = []
            for mark_id in found_marker_ids:
                mark_crs = marker_corners[marker_ids.index(mark_id)]
                mark_ctr = self.cv_detector.get_center_point(mark_crs)
                mark_pos = np.array(self.config_pattern.get_marker_position(mark_id) + [0],
                                    dtype=np.float64) / 1000  # change unit to meter

                img_points_list.append(mark_ctr)
                obj_points_list.append(mark_pos)

            obj_points = np.array(obj_points_list)
            img_points = np.array(img_points_list)
            # found, r_vec, t_vec = cv.solveP3P(
            #     obj_points, img_points, self.camera.cc.intrinsic, self.camera.cc.distortion, flags=cv.SOLVEPNP_P3P)
            found, r_vec, t_vec = cv.solvePnP(
                obj_points, img_points, self.camera.cc.intrinsic, self.camera.cc.distortion, flags=cv.SOLVEPNP_IPPE
            )
            if found:
                r_vec, t_vec = cv.solvePnPRefineLM(
                    objectPoints=obj_points,
                    imagePoints=img_points,
                    cameraMatrix=self.camera.cc.intrinsic,
                    distCoeffs=self.camera.cc.distortion,
                    rvec=r_vec,
                    tvec=t_vec,
                    criteria=(cv.TermCriteria_EPS + cv.TermCriteria_COUNT, 30, 0.001)
                )
                # Rotate estimated pose around 180 degrees in x axes
                pq = converter.cv_to_pq(r_vec, t_vec)
                pq = self.config_offset.apply_offset(pq)
        return found, pq
