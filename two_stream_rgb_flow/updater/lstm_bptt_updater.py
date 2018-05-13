import chainer
from chainer import training

class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, converter, device):
        super(BPTTUpdater, self).__init__(train_iter, optimizer, converter=converter, device=device)


    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        optimizer.target.cleargrads()  # Clear the parameter gradients
        batch = train_iter.__next__()
        cropped_face, bbox, label = self.converter(batch, device=self.device)
        loss = optimizer.target(chainer.Variable(cropped_face), chainer.Variable(bbox), label)
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

