from __future__ import annotations
# global
import logging
import cv2 as cv
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# local
from cvpd.utilities import load_yaml
from cvpd.geo_pattern._aruco_types import ARUCO_DICT

# typing
from typing import Any
from numpy import typing as npt


LOGGER = logging.getLogger(__name__)


class CharucoBoard:

    @dataclass
    class Config:
        marker_size: int
        marker_type: str
        checker_size: int
        checker_grid_size: tuple[int, int]

    def __init__(self, config_fp: Path) -> None:
        self.cfg_fp = config_fp
        self.cfg = CharucoBoard.Config(**load_yaml(config_fp))
        self.ar_type_dict = ARUCO_DICT.get(self.cfg.marker_type)
        if self.ar_type_dict is None:
            error_msg = f"No AR tag with key '{self.cfg.marker_type}' found"
            LOGGER.error(error_msg)
            raise KeyError(error_msg)
        self.aruco_dict = cv.aruco.getPredefinedDictionary(self.ar_type_dict)
        self.grid_size = self.cfg.checker_grid_size
        self.checker_size_m = self.cfg.checker_size / 1000.0
        self.aruco_size_m = self.cfg.marker_size / 1000.0
        self.cv_board = cv.aruco.CharucoBoard(self.grid_size, self.checker_size_m, self.aruco_size_m, self.aruco_dict)

        id_list = self.cv_board.getIds()
        n_cols, n_rows = self.grid_size
        # mpc_even = n_rows // 2
        # mpc_odd = mpc_even if n_rows % 2 == 0 else mpc_even + 1

        mpr_odd = (n_cols + 1) // 2
        mpr_even = mpr_odd if n_cols % 2 == 0 else mpr_odd - 1
        self.id_matrix = -1 * np.ones([n_rows, n_cols], dtype=int)
        for i in range(n_rows):
            if (i + 1) % 2 == 0:
                self.id_matrix[i, 0::2] = id_list[:mpr_odd]
                id_list = id_list[mpr_odd:]
            else:
                self.id_matrix[i, 1::2] = id_list[:mpr_even]
                id_list = id_list[mpr_even:]

    @property
    def board_size_m(self) -> tuple[float, float]:
        sx = self.grid_size[0] * self.checker_size_m
        sy = self.grid_size[1] * self.checker_size_m
        return sx, sy

    @property
    def number_ids(self) -> int:
        n_ids: int = self.aruco_dict.bytesList.shape[0]
        return n_ids

    @property
    def meta_dict(self) -> dict[str, Any]:
        meta_data = {
            'marker_size': self.aruco_dict.markerSize,
            'number_ids': self.number_ids,
            'grid_size': self.grid_size,
            'marker_size_m': self.aruco_size_m,
            'checker_size_m': self.checker_size_m,
        }
        return meta_data

    def generate_image(self,
                       out_size_px: tuple[int, int],
                       margin_size: int = 0,
                       board_bits: int = 1) -> npt.NDArray[np.uint8]:
        img: npt.NDArray[np.uint8] = self.cv_board.generateImage(out_size_px, margin_size, board_bits)
        return img
