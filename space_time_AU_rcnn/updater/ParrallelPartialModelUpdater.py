import chainer
import chainer.dataset.convert as convert
import six
import copy
import config
import chainer.functions as F

class ParallelUpdater(chainer.training.StandardUpdater):

    """Implementation of a parallel GPU Updater.

    This is an implementation of :class:`Updater` that uses multiple GPUs.
    It behaves similarly to :class:`~chainer.training.StandardUpdater`. The
    update routine is modified to support data-parallel computation on multiple
    GPUs in one machine. It is based on synchronous parallel SGD: it
    parallelizes the gradient computation over a mini-batch, and updates the
    parameters only in the main device.

    Args:
        iterator: Dataset iterator for the training dataset. It can also be a
            dictionary that maps strings to iterators.
            If this is just an iterator, then the
            iterator is registered by the name ``'main'``.
        optimizer: Optimizer to update parameters. It can also be a dictionary
            that maps strings to optimizers.
            If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter: Converter function to build input arrays. Each batch
            extracted by the main iterator is split equally between the
            devices and then passed with corresponding ``device`` option to
            this function. :func:`~chainer.dataset.concat_examples` is used by
            default.
        models: Dictionary of models. The main model should be the same model
            attached to the ``'main'`` optimizer.
        devices: Dictionary of devices to which the training data is sent. The
            devices should be arranged in a dictionary with the same structure
            as ``models``.
        loss_func: Loss function. The model is used as a loss function by
            default.

    """

    def __init__(self, iterator, optimizer, database, converter=convert.concat_examples,
                 models=None, devices=None, loss_func=None):  # devices = {"main":0, "gpu_1":1, "gpu_2":2}
        super(ParallelUpdater, self).__init__(
            iterator=iterator,
            optimizer=optimizer,
            converter=converter,
            loss_func=loss_func,
        )
        assert models is not None  # models = { "parallel":AU_rcnn_train_chain}.  models do not need to pass main wrap model
        names = list(six.iterkeys(devices))  # ["main", "gpu_1", "gpu_2"]
        self.database = database
        try:
            names.remove('main')  # in case of copy
        except ValueError:
            raise KeyError("'devices' must contain a 'main' key.")
        for name in names:  # models key == devices key
            parallel_model = copy.deepcopy(models["parallel"])
            if devices[name] >= 0:
                parallel_model.to_gpu(devices[name])
            models[name] = parallel_model  # "parallel": AU_rcnn_train_chain, "gpu_1": AU_rcnn_train_chain, "gpu_2": ...}
        if devices['main'] >= 0:
            optimizer.target.to_gpu(devices['main'])   # 包装类的model copy到主gpu上, 这也就将刚才未拷的parallel拷贝到gpu

        self._devices = devices
        # self._models = {"parallel": AU_rcnn_train_chain # resident on the main gpu, "gpu_1": AU_rcnn_copy, ...}
        self._models = models  # this self._models is all other models except for model_main(wrap_model)

    def connect_trainer(self, trainer):
        # Add observers for all (other) models.
        model_main = self.get_optimizer('main').target
        models_others = {
            k: v for k, v in self._models.items() if v != model_main.au_rcnn_train_chain
        }
        for name, model in models_others.items():
            trainer.reporter.add_observer(name, model)

    def split_list(self, alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
                 for i in range(wanted_parts) ]



    def update_core(self):
        optimizer = self.get_optimizer('main')
        # it is main wrapper class: au_rcnn_train_chain, space_time_rnn
        model_main = optimizer.target
        space_time_rnn = model_main.space_time_rnn

        models_others = {k: v for k, v in self._models.items()
                         if v != model_main.au_rcnn_train_chain}

        batch = self.get_iterator('main').next()
        in_arrays = self.converter(batch, self.device)
        images, bboxes, labels = in_arrays
        batch_size, T, channel, height, width = images.shape
        images = images.reshape(batch_size * T, channel, height, width)  # B*T, C, H, W
        bboxes = bboxes.reshape(batch_size * T, config.BOX_NUM[self.database], 4)  # B*T, 9, 4
        # labels = labels.reshape(batch_size * T, config.BOX_NUM[self.database], -1)  # B*T, 9, 12/22

        # For reducing memory
        for model in six.itervalues(models_others):
            model.cleargrads()
        model_main.cleargrads()
        #
        # Split the batch to sub-batches.
        #
        n = len(self._models)
        in_arrays_list = {}
        sub_index = self.split_list(list(range(batch_size * T)), n)
        for i, key in enumerate(sorted(self._models.keys())):  # self._models are all au_rcnn_train_chain includes main gpu
            in_arrays_list[key] = (F.copy(images[sub_index[i]], self._devices[key]),
                                   F.copy(bboxes[sub_index[i]], self._devices[key]))

        # self._models are all au_rcnn_train_chain includes main gpu
        with function.force_backprop_mode():
            roi_feature_multi_gpu = []
            for model_key, au_rcnn_train_chain in sorted(self._models.items(), key=lambda e:e[0]):
                images, bboxes = in_arrays_list[model_key]
                roi_feature = au_rcnn_train_chain(images, bboxes)  # shape =(B*T//n, F, D)
                roi_feature_multi_gpu.append(F.copy(roi_feature, self._devices["main"]))
            roi_feature = F.concat(roi_feature_multi_gpu, axis=0)  # multiple batch combine
            roi_feature = roi_feature.reshape(batch, T, config.BOX_NUM[self.database], -1)
            loss = space_time_rnn(roi_feature)

        model_main.cleargrads()
        for model in six.itervalues(models_others):
            model.cleargrads()
        loss.backward()
        for model in six.itervalues(models_others):
            model_main.au_rcnn_train_chain.addgrads(model)
        optimizer.update()
        for model in six.itervalues(models_others):
            model.copyparams(model_main.au_rcnn_train_chain)  # only the main model will update parameter, so copy to each other models