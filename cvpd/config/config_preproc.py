from __future__ import annotations

# libs
from cvpd.config.config import Configurable

# typing
from typing import Any


class Preprocessing(Configurable):

    def __init__(self, **kwargs: Any):
        super().__init__()
        inv = kwargs.get('invert_img')
        if inv is None:
            self.invert_img = False
        else:
            self.invert_img = bool(inv)

    def to_dict(self) -> dict[str, bool]:
        return {'invert_img': self.invert_img}
