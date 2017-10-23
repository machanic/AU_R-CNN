from chainer import training
import chainer
from overrides import overrides

# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device,
                 converter=chainer.dataset.concat_examples):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device, converter=converter)
        self.bprop_len = bprop_len
        print("BPTT constructor over")


    def clear_loss_backward(self, optimizer, loss, is_reset_state=True):
        if isinstance(loss, chainer.variable.Variable):
            optimizer.target.cleargrads()  # Clear the parameter gradients
            loss.backward()  # Backprop
            loss.unchain_backward()  # Truncate the graph
            optimizer.update()  # Update the parameters
        if is_reset_state:
            optimizer.target.reset_state()


    # The core part of the update routine can be customized by overriding.
    @overrides
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        prev_video_id = None
        prev_box_num = None
        prev_img_id = None
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            entrys = train_iter.__next__()
            cropped_face, bbox, label, img_id = entrys[0]
            video_id = img_id[:img_id.rindex("/")]
            _mini_batch = [(cropped_face, bbox, label), ]
            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            cropped_face, bbox, label = self.converter(_mini_batch, self.device)
            box_num = bbox.shape[1]
            if prev_box_num is None:
                prev_box_num = box_num
            if prev_box_num != box_num:
                print("box_number not equal! box_num:{0} prev_box_num:{1} prev_img_id:{2}, img_id:{3}".format(box_num,
                                                                                                              prev_box_num,
                                                                                                              prev_img_id,
                                                                                                              img_id))
                prev_box_num = box_num
                self.clear_loss_backward(optimizer, loss)
                loss = optimizer.target(chainer.Variable(cropped_face), chainer.Variable(bbox),
                                        chainer.Variable(label))
            if prev_video_id is None:
                prev_video_id = video_id
            if prev_video_id != video_id:
                prev_video_id = video_id
                self.clear_loss_backward(optimizer,loss)
                loss = optimizer.target(chainer.Variable(cropped_face), chainer.Variable(bbox),
                                        chainer.Variable(label))
            else:
                # Compute the loss at this time step and accumulate it
                # print("batch shape:{}".format(label.shape[0]))
                try:
                    loss += optimizer.target(chainer.Variable(cropped_face), chainer.Variable(bbox),chainer.Variable(label))

                except TypeError:
                    self.clear_loss_backward(optimizer, loss)
                    loss = optimizer.target(chainer.Variable(cropped_face), chainer.Variable(bbox),
                                            chainer.Variable(label))
                    print("The batch size of x must be equal to or less thanthe size of the previous state h. prev_img_id:{0}, img_id:{1} label shape:{2} ".format(prev_img_id,
                                                                                                                                                                   img_id,
                                                                                                                                                                   label.shape))
            prev_img_id = img_id
        self.clear_loss_backward(optimizer,loss, is_reset_state=False)