from enum import Enum

import chainer



class OptimizerType(Enum):
    AdaGrad = 'AdaGrad'
    RMSprop = "RMSprop"
    Adam = "Adam"
    SGD = "SGD"
    AdaDelta = "AdaDelta"
    def __str__(self):
        return self.value