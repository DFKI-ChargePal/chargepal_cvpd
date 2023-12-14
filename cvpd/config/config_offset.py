from __future__ import annotations

# global
import numpy as np
from scipy.spatial.transform import Rotation as R

# local
from cvpd.config.config import Configurable

# typing
from typing import Any, cast
from numpy import typing as npt
from camera_kit.core import PosOrinType


class Offset(Configurable):

    def __init__(self, **kwargs: Any):
        super().__init__()
        offset = kwargs.get('offset')
        if offset is None:
            offset = {
                'xyz': [0.0, 0.0, 0.0],
                'xyzw': [0.0, 0.0, 0.0, 1.0],
            }
        self.p = np.reshape(offset['xyz'], 3)
        self.q = np.reshape(offset['xyzw'], 4)

    def to_dict(self) -> dict[str, dict[str, list[float]]]:
        return {
            'offset': {
                'xyz': self.p.tolist(),
                'xyzw': self.q.tolist(),
            },
        }

    def adjust_offset(self, offset_p: npt.ArrayLike | None, offset_q: npt.ArrayLike | None) -> None:
        self.p = self.p if offset_p is None else np.reshape(offset_p, 3)
        self.q = self.q if offset_q is None else np.reshape(offset_q, 4)

    def apply_offset(self, pq: PosOrinType) -> PosOrinType:
        """ Helper function to apply offset to a given pose

        Args:
            pq: Pose containing position [xyz] and quaternion [xyzw] vector

        Returns:
            Modified pose
        """
        T_12 = np.diag([0., 0., 0., 1.])
        T_23 = np.diag([0., 0., 0., 1.])
        t_12 = np.reshape(pq[0], 3)
        t_23 = np.reshape(self.p, 3)
        rot_12 = R.from_quat(pq[1]).as_matrix()
        rot_23 = R.from_quat(self.q).as_matrix()
        T_12[:3, :3] = rot_12
        T_12[:3, 3] = t_12
        T_23[:3, :3] = rot_23
        T_23[:3, 3] = t_23
        T_13 = T_12 @ T_23
        p = tuple(np.reshape(T_13[:3, 3], 3).tolist())
        q = tuple((R.from_matrix(T_13[:3, :3])).as_quat().tolist())
        pq_offset: PosOrinType = cast(PosOrinType, (p, q))
        return pq_offset
