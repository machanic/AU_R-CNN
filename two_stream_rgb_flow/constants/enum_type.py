from enum import Enum

import chainer






class TwoStreamMode(Enum):
    rgb_flow = "rgb_flow"
    rgb = "rgb"
    optical_flow = "optical_flow"

    def __str__(self):
        return self.value

