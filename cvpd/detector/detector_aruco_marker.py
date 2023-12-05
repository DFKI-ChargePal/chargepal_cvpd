from __future__ import annotations

# global
import cv2 as cv
import numpy as np
from pathlib import Path
from camera_kit import DetectorBase

# local
from cvpd.utilities import rotate_rot_vec
from cvpd.geo_pattern.aruco_marker import ArucoMarker

# typing
from typing import Any
from numpy import typing as npt


class ArucoMarkerDetector(DetectorBase):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Get searched geometric pattern
        obj_cfg: dict[str, dict[str, Any]] | None = self.config.get('object_config')
        if obj_cfg is None:
            raise RuntimeError(f"Can't find configuration values for aruco markers.")
        self.obj_marker: dict[str, ArucoMarker] = {}
        for obj_name, obj_val in obj_cfg.items():
            self.obj_marker[obj_name] = ArucoMarker(
                aruco_id=obj_val['marker_id'],
                aruco_size_mm=obj_val['marker_size'],
                aruco_type=obj_val['marker_type']
            )

    def _find_pose(self, object_name: str) -> tuple[bool, npt.NDArray[np.float_]]:
        """ Finding the pose of a marker described object

        Args:
            object_name: Unique name of the searched object type

        Returns:
            (True if pose was found; Pose as numpy array. Containing the rotation and translation vector)
        """
        # Initialize return variables with default values
        pose_found = False
        r_vec, t_vec = np.array([], dtype=np.float_), np.array([], dtype=np.float_)
        # Find marker configuration
        aruco_marker = self.obj_marker.get(object_name)
        if aruco_marker is not None:
            img = self.camera.get_color_frame()
            # get all markers on image
            found_corners, found_ids, _ = aruco_marker.cv_detector.detectMarkers(img)
            # if corners are detected, check if they include the searched marker
            if len(found_corners) > 0:
                det_id_list = [_id[0] for _id in found_ids.tolist()]
                if aruco_marker.aruco_id in det_id_list:
                    # Get first index
                    id_idx = det_id_list.index(aruco_marker.aruco_id)
                    corners = found_corners[id_idx].reshape((4, 2))
                    # estimate pose for the single marker
                    pose_found, r_vec, t_vec = self._estimate_pose_single_marker(corners, aruco_marker.obj_pts_marker)
        return pose_found, np.asarray((r_vec, t_vec))

    def _estimate_pose_single_marker(self,
                                     marker_corners: npt.NDArray[np.float_],
                                     marker_obj_pts: npt.NDArray[np.float_]
                                     ) -> tuple[bool, npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """ Method to estimate the pose of a single ArUco marker.

        Args:
            marker_corners: The marker corner points in the image

        Returns:
            The rotation and translation vector of the marker pose
        """
        _ret, r_vec, t_vec = cv.solvePnP(
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
        r_vec = rotate_rot_vec(r_vec, 'x', np.pi)
        return _ret, r_vec, t_vec
