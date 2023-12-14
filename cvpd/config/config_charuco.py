from __future__ import annotations
# global
import cv2 as cv

# local
from cvpd.config.config import Configurable
from cvpd.config._aruco_types import ARUCO_DICT

# typing
from typing import Any


class Charuco(Configurable):

    def __init__(self, **kwargs: Any):
        super().__init__()
        self.marker_size: int = kwargs['marker_size']
        self.marker_type: str = kwargs['marker_type']
        self.checker_size: int = kwargs['checker_size']
        self.checker_grid_size: list[int] = kwargs['checker_grid_size']

    def to_dict(self) -> dict[str, Any]:
        return {
            'marker_size': self.marker_size,
            'marker_type': self.marker_type,
            'checker_size': self.checker_size,
            'checker_grid_size': self.checker_grid_size,
        }

    @property
    def cv_aruco_dict(self) -> cv.aruco.Dictionary:
        ar_type_dict = ARUCO_DICT.get(self.marker_type)
        if ar_type_dict is None:
            error_msg = f"No AR tag with key '{self.marker_type}' found"
            raise KeyError(error_msg)
        return cv.aruco.getPredefinedDictionary(ar_type_dict)

    @property
    def checker_size_m(self) -> float:
        return self.checker_size / 1000.0

    @property
    def marker_size_m(self) -> float:
        return self.marker_size / 1000.0
