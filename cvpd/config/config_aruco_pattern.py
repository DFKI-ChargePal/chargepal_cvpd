from __future__ import annotations

# global
import cv2 as cv

# local
from cvpd.config.config import Configurable
from cvpd.config._aruco_types import ARUCO_DICT

# typing
from typing import Any


class ArucoPattern(Configurable):

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.marker_size: int = kwargs['marker_size']
        self.marker_type: str = kwargs['marker_type']
        self.marker_layout: dict[int, list[int]] = {}
        raw_marker_layout = kwargs['marker_layout']
        for k, v in raw_marker_layout.items():
            assert len(v) == 2
            self.marker_layout[int(k)] = [int(_v) for _v in v]

    def to_dict(self) -> dict[str, Any]:
        return {
            'marker_size': self.marker_size,
            'marker_type': self.marker_type,
            'marker_layout': self.marker_layout,
        }

    @property
    def marker_ids(self) -> set[int]:
        return set(self.marker_layout.keys())

    @property
    def id_range(self) -> int:
        return int(self.cv_aruco_dict.bytesList.shape[0])

    def get_marker_position(self, marker_id: int) -> list[int]:
        marker_pos = self.marker_layout.get(marker_id)
        if marker_pos is None:
            raise KeyError(f"There is no position defined for the given marker id: {marker_id}")
        return marker_pos

    @property
    def cv_aruco_dict(self) -> cv.aruco.Dictionary:
        ar_type_dict = ARUCO_DICT.get(self.marker_type)
        if ar_type_dict is None:
            error_msg = f"No AR tag with key '{self.marker_type}' found"
            raise KeyError(error_msg)
        return cv.aruco.getPredefinedDictionary(ar_type_dict)

