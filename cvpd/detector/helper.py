from __future__ import annotations

import cv2 as cv


# global
import numpy as np

# typing
from typing import Sequence
from numpy import typing as npt


class ArucoOpenCV:

    def __init__(self, cv_aruco_detector: cv.aruco.ArucoDetector):
        self.cv_detector = cv_aruco_detector

    def find_single_marker_corners(self,
                                   img: npt.NDArray[np.uint8], marker_id: int) -> tuple[int, npt.NDArray[np.float64]]:
        """ Finding the corners of a single marker

        Args:
            img:       RGB image
            marker_id: ArUco marker id

        Returns:
            (Marker id or -1 if marker was not found; The corners of the marker)
        """
        # get all markers on image
        found_corners, found_ids, _ = self.cv_detector.detectMarkers(img)
        # instantiate empty return values
        ret_id = -1
        corners = np.array([])
        # if corners are detected, check if they include the searched marker
        if len(found_corners) > 0:
            det_id_list = [_id[0] for _id in found_ids.tolist()]
            if marker_id in det_id_list:
                # Get first index with matching marker id
                id_idx = det_id_list.index(marker_id)
                ret_id = marker_id
                corners = found_corners[id_idx].reshape((4, 2))
        return ret_id, corners

    def find_group_marker_corners(self,
                                  img: npt.NDArray[np.uint8],
                                  marker_ids: Sequence[int] | None = None
                                  ) -> tuple[list[int], list[npt.NDArray[np.float64]]]:
        """ Finding the corners of a group of markers

        Args:
            img:        RGB image
            marker_ids: List of ArUco marker ids

        Returns:
            (List of marker ids or -1 of not found; List of marker corners)
        """
        # get all markers on image
        found_corners, found_marker_ids, _ = self.cv_detector.detectMarkers(img)
        if found_marker_ids is None:
            found_marker_ids = []
        else:
            found_marker_ids = [_id[0] for _id in found_marker_ids.tolist()]  # transform to flat list
        # if corners are detected, check if they include the searched marker
        if marker_ids:
            # Create an id and corner list in parallel
            ret_marker_ids = []
            ret_corners: list[npt.NDArray[np.float64]] = []
            for id_ in marker_ids:
                # if corners are detected, check if they include the searched marker
                if len(found_corners) > 0 and id_ in found_marker_ids:
                    # Get first index with matching marker id
                    id_idx = found_marker_ids.index(id_)
                    ret_marker_ids.append(id_)
                    ret_corners.append(found_corners[id_idx].reshape((4, 2)))
                else:
                    # Replace id with -1 if marker was not found
                    ret_marker_ids.append(-1)
                    ret_corners.append(np.array([]))
        else:
            # Return all markers that are found
            ret_marker_ids = found_marker_ids
            ret_corners = [m_corners.reshape((4, 2)) for m_corners in found_corners]
        return ret_marker_ids, ret_corners

    @staticmethod
    def get_center_point(marker_corners: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """ Helper function to determine the marker center from the corner points

        Args:
            marker_corners: Array with corner points

        Returns:
            Array with center point
        """
        top_left, top_right, bottom_right, bottom_left = marker_corners
        center_X = (top_left[0] + bottom_right[0]) / 2.0
        center_Y = (top_left[1] + bottom_right[1]) / 2.0
        return np.array([center_X, center_Y], dtype=np.float64)
