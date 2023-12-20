from __future__ import annotations

# global
from pathlib import Path
from typing import Dict, Type

from cvpd.detector.detector_charuco import CharucoDetector
from cvpd.detector.detector_aruco_marker import ArucoMarkerDetector
from cvpd.detector.detector_aruco_pattern import ArucoPatternDetector


class Factory:

    dtt_dict: Dict[str, Type[
        ArucoMarkerDetector | CharucoDetector | ArucoPatternDetector
    ]] = {
        'charuco': CharucoDetector,
        'aruco_marker': ArucoMarkerDetector,
        'aruco_pattern': ArucoPatternDetector,
    }

    @staticmethod
    def create(config_fp: str | Path) -> ArucoMarkerDetector | CharucoDetector | ArucoPatternDetector:
        config_fp = Path(config_fp)
        cfg_fn = config_fp.name
        for dtt_name, dtt_class in Factory.dtt_dict.items():
            if cfg_fn.startswith(dtt_name):
                return dtt_class(config_fp)
        raise ValueError(f"Configuration file with name {cfg_fn} contains no pattern to match to a detector class.")
