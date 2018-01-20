import chainer
import cupy
import numpy
from chainer import training


def convert(batch, device):

    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = chainer.cuda.to_cpu
    else:
        def to_device(x):
            return chainer.cuda.to_gpu(x, device, chainer.cuda.Stream.null)

    def to_device_batch(batch):
        x_batch = []
        crf_pact_structure_batch = []
        if device is None:
            xp = numpy
        else:
            xp = cupy if device >= 0 else numpy
        for x, crf_pact_structure in batch:
            x_batch.append(to_device(x))
            if hasattr(crf_pact_structure, "A"):
                crf_pact_structure.A = to_device(crf_pact_structure.A)
            crf_pact_structure_batch.append(crf_pact_structure)
        return xp.stack(x_batch, axis=0), crf_pact_structure_batch

    return to_device_batch(batch)


class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, device):
        super(BPTTUpdater, self).__init__(train_iter, optimizer,converter=convert, device=device)

    def update_core(self):

        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.__next__()
        xs, crf_pact_structure_batch = self.converter(batch, self.device)
        loss = optimizer.target(chainer.Variable(xs), crf_pact_structure_batch)

        optimizer.target.cleargrads() # Clear the parameter gradients
        loss.backward() # Backprop
        loss.unchain_backward() # Truncate the graph
        optimizer.update() # Update the parameters
