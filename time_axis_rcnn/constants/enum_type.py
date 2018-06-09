from enum import Enum

import chainer

class TwoStreamMode(Enum):
    rgb_flow = "rgb_flow"
    rgb = "rgb"
    optical_flow = "optical_flow"

    def __str__(self):
        return self.value

class FasterBackboneType(Enum):
    conv1d = "conv1d"
    tcn = "tcn"
    def __str__(self):
        return self.value

class OptimizerType(Enum):
    AdaGrad = 'AdaGrad'
    RMSprop = "RMSprop"
    Adam = "Adam"
    SGD = "SGD"
    AdaDelta = "AdaDelta"
    def __str__(self):
        return self.value