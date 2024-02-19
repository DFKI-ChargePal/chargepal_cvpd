from __future__ import annotations

# global
import cv2 as cv
import numpy as np
import spatialmath as sm
from pathlib import Path
from camera_kit import converter

# local
from cvpd.detector.detector_abc import DetectorABC
from cvpd.config.config_aruco_marker import ArucoMarker

# typing
from numpy import typing as npt


class ArucoMarkerDetector(DetectorABC):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Create configuration
        self.config_marker = ArucoMarker(**self.config_dict)
        self.config.add(self.config_marker)
        # Set OpenCV detector up
        self.cv_detector = cv.aruco.ArucoDetector(self.config_marker.cv_aruco_dict)
        # Check if aruco_id is valid
        if 0 <= self.config_marker.marker_id < self.config_marker.id_range:
            self.aruco_id = self.config_marker.marker_id
        else:
            raise ValueError(f"Given id of ArUco marker '{self.config_marker.marker_id}' "
                             f"not in valid range 0...{self.config_marker.id_range - 1}")
        # Define object points
        ar_size_m_2 = self.config_marker.marker_size / 2 / 1000
        self.obj_pts_marker = np.array([
            [-ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, +ar_size_m_2, 0.0],
            [+ar_size_m_2, -ar_size_m_2, 0.0],
            [-ar_size_m_2, -ar_size_m_2, 0.0],
        ], dtype=np.float_)

    def _find_pose(self) -> tuple[bool, sm.SE3]:
        """ Finding the pose of a marker described object

        Returns:
            (True if pose was found; Pose as SE(3) transformation matrix)
        """
        # Initialize return variables with default values
        found, mat = False, sm.SE3()
        img = self.camera.get_color_frame()
        # get all markers on image
        found_corners, found_ids, _ = self.cv_detector.detectMarkers(img)
        # if corners are detected, check if they include the searched marker
        if len(found_corners) > 0:
            det_id_list = [_id[0] for _id in found_ids.tolist()]
            if self.config_marker.marker_id in det_id_list:
                # Get first index
                id_idx = det_id_list.index(self.config_marker.marker_id)
                corners = found_corners[id_idx].reshape((4, 2))
                # estimate pose for the single marker
                found, mat = self._estimate_pose_single_marker(corners, self.obj_pts_marker)
                if found:
                    mat = self.config_offset.apply_offset(mat)
        return found, mat

    def _estimate_pose_single_marker(self,
                                     marker_corners: npt.NDArray[np.float_],
                                     marker_obj_pts: npt.NDArray[np.float_]
                                     ) -> tuple[bool, sm.SE3]:
        """ Method to estimate the pose of a single ArUco marker.

        Args:
            marker_corners: The marker corner points in the image

        Returns:
            (True if pose was found; The marker pose as SE(3) object)
        """
        found, r_vec, t_vec = cv.solvePnP(
            marker_obj_pts,
            marker_corners,
            self.camera.cc.intrinsic,
            self.camera.cc.distortion,
            flags=cv.SOLVEPNP_IPPE_SQUARE
        )
        r_vec, t_vec = cv.solvePnPRefineLM(
            objectPoints=marker_obj_pts,
            imagePoints=marker_corners,
            cameraMatrix=self.camera.cc.intrinsic,
            distCoeffs=self.camera.cc.distortion,
            rvec=r_vec,
            tvec=t_vec,
            criteria=(cv.TermCriteria_EPS + cv.TermCriteria_COUNT, 30, 0.001)
            )
        mat = converter.cv_to_se3(r_vec, t_vec)
        return found, mat
