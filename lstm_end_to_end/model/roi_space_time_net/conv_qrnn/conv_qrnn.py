import chainer
import chainer.links as L
import chainer.functions as functions
from chainer.utils import type_check
from chainer import Function,  initializers
import chainer.cuda as cuda
import numpy as np
import math


class Zoneout(Function):

    def __init__(self, p):
        self.p = p

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        if not hasattr(self, "mask"):
            xp = cuda.get_array_module(*x)
            if xp == np:
                flag = xp.random.rand(*x[0].shape) >= self.p  # 大于某个ratio的数值的留下
            else:
                flag = xp.random.rand(*x[0].shape, dtype=np.float32) >= self.p
            self.mask = flag

        return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,

def zoneout(x, ratio=.5):
    return Zoneout(ratio)(x)


class ConvQRNN(chainer.Chain):
    def __init__(self,  in_channels, out_channels, ksize, pooling, zoneout, bias, wgain=1.):
        """
          Initialize ConvLSTM cell.
          Parameters
          ----------
          input_size: (int, int)
              Height and width of input tensor as (height, width).
          input_dim: int
              Number of channels of input tensor.
          out_channels: int
              Number of channels of hidden state.
          ksize: (int, int, int)
              Size of the convolutional kernel.
          bias: bool
              Whether or not to add the bias.
        """
        super(ConvQRNN, self).__init__()
        self.num_split = len(pooling) + 1
        self._pooling = pooling  # f or fo or ifo

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = ksize  # typical is (3, 1, 1)
        self.bias = bias
        self._zoneout = zoneout
        self._using_zoneout = True if self._zoneout > 0 else False
        self.padding = (ksize[0]-1, ksize[1]//2, ksize[2]//2)
        wstd = math.sqrt(wgain / in_channels / ksize[0])
        with self.init_scope():
            self.conv = L.ConvolutionND(ndim=3, in_channels=self.in_channels,
                                        out_channels=self.num_split * self.out_channels,
                                        ksize=self.kernel_size,  stride=1, pad=self.padding,
                                        nobias=not self.bias, initialW=initializers.Normal(wstd))



    def __call__(self, X): # B, T, C, H, W
        # remove right paddings
        # e.g.
        # kernel_size = 3
        # pad = 2
        # input sequence with paddings:
        # [0, 0, x1, x2, x3, 0, 0]
        # |< t1 >|
        #     |< t2 >|
        #         |< t3 >|

        X = functions.transpose(X, axes=(0, 2, 1, 3, 4))  # B, C, T, H ,W
        # thus the kernal_size is (3, 1, 1)
        combined_conv = self.conv(X)  # B, num_split * hidden_dim, T, H, W
        pad = self.kernel_size[0] - 1
        WX = combined_conv[:, :, :-pad, :, :]
        pool_result = self.pool(functions.split_axis(WX, self.num_split, axis=1))  # B, C, T, H, W
        return functions.transpose(pool_result, axes=(0, 2, 1, 3, 4)) # B, T, C, H, W



    def zoneout(self, U):
        if self._using_zoneout and chainer.config.train:
            return 1- zoneout(functions.sigmoid(-U), self._zoneout)
        return functions.sigmoid(U)




    def pool(self, WX):
        Z, F, O, I = None, None, None, None

        # f-pooling
        if len(self._pooling) == 1:
            assert len(WX) == 2
            Z, F = WX  # Z and F is shape of (B, C, T, H, W)
            Z = functions.tanh(Z)
            F = self.zoneout(F)

        # fo-pooling
        if len(self._pooling) == 2:
            assert len(WX) == 3
            Z, F, O = WX
            Z = functions.tanh(Z)
            F = self.zoneout(F)
            O = functions.sigmoid(O)

        # ifo-pooling
        if len(self._pooling) == 3:
            assert len(WX) == 4
            Z, F, O, I = WX
            Z = functions.tanh(Z)
            F = self.zoneout(F)
            O = functions.sigmoid(O)
            I = functions.sigmoid(I)

        assert Z is not None
        assert F is not None

        T = Z.shape[2]
        ct = None
        H = None

        i_all = 1 - F if I is None else I
        i_mul_z = i_all * Z

        C = []
        for t in range(T):

            ft = F[:, :, t, : ,:]     # B, C, H, W
            # ot = 1 if O is None else O[:, :, t, :, :]
            # it = 1 - ft if I is None else I[:, :, t, :, :]
            # it = i_all[:, :, t, :, :]
            if ct is None:
                zt = Z[:, :, t, :, :]  # B, C, H, W
                ct = (1 - ft) * zt # 没乘以 xt
            else:
                ct = ft * ct + i_mul_z[:, :, t, :, :]
            C.append(ct)
        C = functions.stack(C, axis=2)
        O_result = 1 if O is None else O
        H = C if O is None else O_result * C
        return H  # B, C, T, H, W




# deprecated, DON'T use below code!!!
class ConvSRUCell(chainer.Chain):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        """
          Initialize ConvLSTM cell.
          Parameters
          ----------
          input_size: (int, int)
              Height and width of input tensor as (height, width).
          input_dim: int
              Number of channels of input tensor.
          hidden_dim: int
              Number of channels of hidden state.
          kernel_size: (int, int)
              Size of the convolutional kernel.
          bias: bool
              Whether or not to add the bias.
        """
        super(ConvSRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        if isinstance(kernel_size, tuple):
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        else:
            self.padding = kernel_size // 2
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=self.input_dim,
                                        out_channels=3 * self.hidden_dim,
                                        ksize=self.kernel_size, pad=self.padding,
                                        nobias=False)
            if self.input_dim != self.hidden_dim:
                self.residule_conv = L.Convolution2D(in_channels=self.input_dim,
                                                     out_channels=self.hidden_dim,
                                                     ksize=1, pad=0, nobias=True)


    def __call__(self, input_tensor, zero_state):   # input_tensor is B, T, C, H, W
        assert input_tensor.ndim == 5
        mini_batch, seq_len, channel, height, width = input_tensor.shape
        c_zero = zero_state
        x = input_tensor.reshape(mini_batch * seq_len, channel, height, width)
        u = self.conv(x)
        x_tilde, f, r = functions.split_axis(u, 3, axis=1) # each is B*T, hidden_dim, H, W
        x_tilde = x_tilde.reshape(mini_batch, seq_len, self.hidden_dim, height, width)
        f = f.reshape(mini_batch, seq_len, self.hidden_dim, height, width)
        r = r.reshape(mini_batch, seq_len, self.hidden_dim, height, width) # B, T, C, H, W
        f = functions.sigmoid(f)  # B, T, C, H, W
        r = functions.sigmoid(r)
        c_cur = c_zero

        second_term_f_mul_x_tilde = (1-f) * x_tilde
        all_c = []
        for t in range(seq_len):
            c_next = f[:, t, :, :, :] * c_cur + second_term_f_mul_x_tilde[:, t, :, :, :]
            c_cur = c_next
            all_c.append(c_next)
        all_c = functions.stack(all_c, axis=1)  # B, T, C, H, W
        if self.input_dim != self.hidden_dim:
            x = self.residule_conv(x)
        x = x.reshape(mini_batch, seq_len, self.hidden_dim, height, width)
        all_h = r * functions.tanh(all_c) + (1-r) * x  # B, T, C, H, W
        return all_h

    # init c
    def init_hidden(self, batch_size, height, width): # B, C, H, W
        return chainer.Variable(self.xp.zeros((batch_size, self.hidden_dim, height, width), dtype=self.xp.float32))

class ConvSRU(chainer.Chain):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvSRU, self).__init__()
        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers


        self.cell_list = []

        with self.init_scope():
            for i in range(self.num_layers):
                cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
                setattr(self, "cell_{}".format(i), ConvSRUCell(
                                                               input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                                               kernel_size=self.kernel_size[i]))
                self.cell_list.append("cell_{}".format(i))

    def __call__(self, input_tensor):
        """
            Parameters
            ----------
            input_tensor: todo
                5-D Tensor either of shape (b, t, c, h, w)
            hidden_state: todo
                None. todo implement stateful
            Returns
            -------
            last_state_list, layer_output
        """

        # Implement stateless ConvSRU
        hidden_state = self._init_hidden(batch_size=input_tensor.shape[0], height=input_tensor.shape[-2],
                                         width=input_tensor.shape[-1])
        cur_layer_input = input_tensor
        last_layer_output = None
        for layer_idx in range(self.num_layers):
            c = hidden_state[layer_idx]
            h = getattr(self, self.cell_list[layer_idx])(input_tensor=cur_layer_input, zero_state=c)  # B, T, C, H, W
            cur_layer_input = h
            last_layer_output = h

        return last_layer_output

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(getattr(self, self.cell_list[i]).init_hidden(batch_size, height, width))
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
