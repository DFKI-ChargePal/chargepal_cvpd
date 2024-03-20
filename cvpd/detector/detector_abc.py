from __future__ import annotations

# global
import abc
import spatialmath as sm
from pathlib import Path
from camera_kit import DetectorBase

# local
from cvpd.utilities import dump_yaml
from cvpd.config.config import Configuration
from cvpd.config.config_offset import Offset
from cvpd.config.config_preproc import Preprocessing


class DetectorABC(DetectorBase, metaclass=abc.ABCMeta):

    def __init__(self, config_file: str | Path):
        # Read configuration via base class
        super().__init__(config_file)
        # Create configuration
        self.config_preproc = Preprocessing(**self.config_dict)
        self.config_offset = Offset(**self.config_dict)
        self.config = Configuration(self.config_preproc, self.config_offset)

    @abc.abstractmethod
    def _find_pose(self) -> tuple[bool, sm.SE3]:
        """ Abstract class method to get the object pose estimate

        Returns:
            (True if pose was found; Pose as SE(3) transformation matrix)
        """
        raise NotImplementedError("Must be implemented in subclass")

    def adjust_offset(self, offset_mat: sm.SE3 | None) -> None:
        self.config_offset.adjust_offset(offset_mat)
        # Store adjusted configuration
        adj_cfg_fp = self.config_fp.parent.joinpath(self.config_fp.stem + '_adj' + self.config_fp.suffix)
        dump_yaml(self.config.to_dict(), adj_cfg_fp)
