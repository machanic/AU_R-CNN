
import chainer
import chainer.functions as F
from chainer import configuration
from chainer.utils import argument


class SeparatedBatchNorm1d(chainer.Chain):
    """
        A batch normalization module which keeps its running mean
        and variance separately per timestep.
    """
    def __init__(self, num_features, max_length, eps=1e-5,  affine=True):
        """
           Most parts are copied from
           torch.nn.modules.batchnorm._BatchNorm.
       """
        super(SeparatedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.max_length = max_length
        self.affine = affine
        self.eps = eps
        self.N = 0
        self.register_persistent('N')
        self.decay = 0.9

        with self.init_scope():
            if affine:
                self.gamma = chainer.Parameter(chainer.initializers.Constant(0.1), shape=(num_features,))
                self.beta = chainer.Parameter(chainer.initializers.Constant(0.0,dtype=self.xp.float32), shape=(num_features,))
            else:
                self.add_persistent("gamma", None)
                self.add_persistent("beta", None)
            for i in range(max_length):
                setattr(self, 'running_mean_{}'.format(i), self.xp.zeros(num_features, dtype=self.xp.float32))
                self.register_persistent('running_mean_{}'.format(i))
                setattr(self, "running_var_{}".format(i), self.xp.ones(num_features, dtype=self.xp.float32))
                self.register_persistent("running_var_{}".format(i))
            self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.max_length):
            running_mean_i = getattr(self, 'running_mean_{}'.format(i))
            running_var_i = getattr(self, 'running_var_{}'.format(i))
            running_mean_i[...] = 0.0
            running_var_i[...] = 1

    def _check_input_dim(self, input_):
        if input_.shape[1] != self.running_mean_0.size:
            raise ValueError('got {}-feature tensor, expected {}'
                             .format(input_.shape[1], self.num_features))

    def __call__(self, input_, time, **kwargs):
        argument.check_unexpected_kwargs(
            kwargs, test='test argument is not supported anymore. '
                         'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        self._check_input_dim(input_)
        if time >= self.max_length:
            time = self.max_length - 1
        running_mean = getattr(self, 'running_mean_{}'.format(time))
        running_var = getattr(self, 'running_var_{}'.format(time))
        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1./ self.N
            else:
                decay = self.decay


            return F.batch_normalization(input_, running_mean=running_mean, running_var=running_var,
                                     gamma=self.gamma, beta=self.beta, eps=self.eps, decay=decay)
        else:
            running_mean = chainer.Variable(running_mean)
            running_var = chainer.Variable(running_var)
            return F.fixed_batch_normalization(input_, mean=running_mean, var=running_var,
                                        gamma=self.gamma,beta=self.beta, eps=self.eps)

    def __repr__(self):
        return ('{name}({num_features}, eps={eps}, momentum={momentum},'
                ' max_length={max_length}, affine={affine})'
                .format(name=self.__class__.__name__, **self.__dict__))

