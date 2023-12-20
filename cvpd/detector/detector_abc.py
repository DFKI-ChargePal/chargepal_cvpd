from __future__ import annotations

# global
import abc
from pathlib import Path
from camera_kit import DetectorBase

# local
from cvpd.utilities import dump_yaml
from cvpd.config.config import Configuration
from cvpd.config.config_offset import Offset

# typing
from numpy import typing as npt
from camera_kit.core import PosOrinType


class DetectorABC(DetectorBase, metaclass=abc.ABCMeta):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Create configuration
        self.config_offset = Offset(**self.config_dict)
        self.config = Configuration(self.config_offset)

    @abc.abstractmethod
    def _find_pose(self) -> tuple[bool, PosOrinType]:
        """ Abstract class method to get the object pose estimate

        Returns:
            (True if pose was found; Pose containing position [xyz] and quaternion [xyzw] vector)
        """
        raise NotImplementedError("Must be implemented in subclass")

    def adjust_offset(self, offset_p: npt.ArrayLike | None, offset_q: npt.ArrayLike | None) -> None:
        self.config_offset.adjust_offset(offset_p, offset_q)
        # Store adjusted configuration
        adj_cfg_fp = self.config_fp.parent.joinpath(self.config_fp.stem + '_adj' + self.config_fp.suffix)
        dump_yaml(self.config.to_dict(), adj_cfg_fp)
