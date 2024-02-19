from __future__ import annotations

# libs
import spatialmath as sm
from spatialmath.base import r2q, q2r

from cvpd.config.config import Configurable

# typing
from typing import Any


class Offset(Configurable):

    def __init__(self, **kwargs: Any):
        super().__init__()
        offset = kwargs.get('offset')
        if offset is None:
            offset = {
                'xyz': [0.0, 0.0, 0.0],
                'xyzw': [0.0, 0.0, 0.0, 1.0],
            }
        self.mat = sm.SE3().Rt(R=q2r(offset['xyzw'], order='xyzs'), t=offset['xyz'])

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            'offset': {
                'xyz': self.mat.t.tolist(),
                'xyzw': r2q(self.mat.R, order='xyzs').tolist(),
            },
        }

    def adjust_offset(self, offset_mat: sm.SE3 | None) -> None:
        self.mat = self.mat if offset_mat is None else offset_mat

    def apply_offset(self, mat: sm.SE3) -> sm.SE3:
        """ Helper function to apply offset to a given pose

        Args:
            mat: Pose SE(3) object

        Returns:
            Modified pose
        """
        mat12 = mat
        mat23 = self.mat
        mat13 = mat12 * mat23
        return mat13
