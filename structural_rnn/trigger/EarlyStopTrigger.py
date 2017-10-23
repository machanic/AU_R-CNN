
from chainer import reporter
from chainer.training import util

class EarlyStoppingTrigger(object):

    """Early stopping trigger

    It observes the value specified by `key`, and invoke a trigger only when
    observing value satisfies the `stop_condition`.
    The trigger may be used to `stop_trigger` option of Trainer module for
    early stopping the training.
    Args:
        max_epoch (int or float): Max epoch for the training, even if the value
            is not reached to the condition specified by `stop_condition`,
            finish the training if it reaches to `max_epoch` epoch.
        key (str): Key of value to be observe for `stop_condition`.
        stop_condition (callable): To check the previous value and current value
            to decide early stop timing. Default value is `None`, in that case
            internal `_stop_condition` method is used.
        eps (float): It is used by the internal `_stop_condition`.
        trigger: Trigger that decides the comparison interval between previous
            best value and current value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.
    """

    def __init__(self, max_epoch, key, stop_condition=None, eps=1e-3,
                 trigger=(1, 'epoch')):
        self.max_epoch = max_epoch
        self.eps = eps
        self._key = key
        self._current_value = None
        self._interval_trigger = util.get_trigger(trigger)
        self._init_summary()
        self.stop_condition = stop_condition or self._stop_condition

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.
        Returns:
            bool: ``True`` if the corresponding extension should be invoked in
                this iteration.
        """

        epoch_detail = trainer.updater.epoch_detail
        if self.max_epoch <= epoch_detail:
            print('Reached to max_epoch.')
            return True

        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._current_value is None:
            self._current_value = value
            return False
        else:
            if self.stop_condition(self._current_value, value):
                # print('Previous value {}, Current value {}'
                #       .format(self._current_value, value))
                print('Invoke EarlyStoppingTrigger...')
                self._current_value = value
                return True
            else:
                self._current_value = value
                return False

    def _init_summary(self):
        self._summary = reporter.DictSummary()

    def _stop_condition(self, current_value, new_value):
        return current_value - new_value < self.eps