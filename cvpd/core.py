from __future__ import annotations

# global
from pathlib import Path

# local
from cvpd.detector.detector_abc import DetectorABC
from cvpd.detector.detector_charuco import CharucoDetector
from cvpd.detector.detector_aruco_marker import ArucoMarkerDetector
from cvpd.detector.detector_aruco_pattern import ArucoPatternDetector

# typing
from typing import Type


class DetectorFactory:

    def __init__(self) -> None:
        self._detectors: dict[str, Type[DetectorABC]] = {}

    def register_detector(self, name: str, detector: Type[DetectorABC]) -> None:
        self._detectors[name] = detector

    def get_detector(self, config_fp: str | Path) -> DetectorABC:
        config_fp = Path(config_fp)
        cfg_fn = config_fp.name
        for dtt_name, dtt_class in self._detectors.items():
            if cfg_fn.startswith(dtt_name):
                return dtt_class(config_fp)
        raise ValueError(f"Configuration file with name {cfg_fn} contains no pattern to match to a detector class.")


factory = DetectorFactory()
factory.register_detector('charuco', CharucoDetector)
factory.register_detector('aruco_marker', ArucoMarkerDetector)
factory.register_detector('aruco_pattern', ArucoPatternDetector)
