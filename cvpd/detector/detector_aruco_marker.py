from __future__ import annotations

# global
import cv2 as cv
import numpy as np
from pathlib import Path
from camera_kit import converter
from camera_kit import DetectorBase
from scipy.spatial.transform import Rotation as R

# local
from cvpd.utilities import rotate_rot_vec
from cvpd.geo_pattern.pattern_aruco_marker import ArucoMarker

# typing
from numpy import typing as npt
from camera_kit.core import PosOrinType


class ArucoMarkerDetector(DetectorBase):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Get marker configuration
        self.marker = ArucoMarker(self.config_fp)
        self.cv_detector = cv.aruco.ArucoDetector(self.marker.aruco_dict)

    def _find_pose(self) -> tuple[bool, PosOrinType]:
        """ Finding the pose of a marker described object

        Returns:
            (True if pose was found; Pose containing position [xyz] and quaternion [wxyz] vector)
        """
        # Initialize return variables with default values
        found = False
        p, q = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        img = self.camera.get_color_frame()
        # get all markers on image
        found_corners, found_ids, _ = self.cv_detector.detectMarkers(img)
        # if corners are detected, check if they include the searched marker
        if len(found_corners) > 0:
            det_id_list = [_id[0] for _id in found_ids.tolist()]
            if self.marker.aruco_id in det_id_list:
                # Get first index
                id_idx = det_id_list.index(self.marker.aruco_id)
                corners = found_corners[id_idx].reshape((4, 2))
                # estimate pose for the single marker
                found, pq = self._estimate_pose_single_marker(corners, self.marker.obj_pts_marker)
                if found:
                    T_12 = np.diag([0., 0., 0., 1.])
                    T_23 = np.diag([0., 0., 0., 1.])
                    t_12 = np.reshape(pq[0], 3)
                    t_23 = np.reshape(self.marker.offset_p, 3)
                    rot_12 = R.from_quat(pq[1]).as_matrix()
                    rot_23 = R.from_quat(self.marker.offset_q).as_matrix()
                    T_12[:3, :3] = rot_12
                    T_12[:3, 3] = t_12
                    T_23[:3, :3] = rot_23
                    T_23[:3, 3] = t_23
                    T_13 = T_12 @ T_23
                    p = tuple(np.reshape(T_13[:3, 3], 3).tolist())
                    q = tuple((R.from_matrix(T_13[:3, :3])).as_quat().tolist())
        return found, (p, q)

    def _estimate_pose_single_marker(self,
                                     marker_corners: npt.NDArray[np.float_],
                                     marker_obj_pts: npt.NDArray[np.float_]
                                     ) -> tuple[bool, PosOrinType]:
        """ Method to estimate the pose of a single ArUco marker.

        Args:
            marker_corners: The marker corner points in the image

        Returns:
            The rotation and translation vector of the marker pose
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
        # r_vec = np.reshape(rotate_rot_vec(r_vec, 'x', np.pi), 3)
        pq = converter.cv_to_pq(r_vec, t_vec)
        return found, pq
