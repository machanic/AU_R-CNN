import chainer
import chainer.links as L
import chainer.functions as F

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
        x_tilde, f, r = F.split_axis(u, 3, axis=1) # each is B*T, hidden_dim, H, W
        x_tilde = x_tilde.reshape(mini_batch, seq_len, self.hidden_dim, height, width)
        f = f.reshape(mini_batch, seq_len, self.hidden_dim, height, width)
        r = r.reshape(mini_batch, seq_len, self.hidden_dim, height, width) # B, T, C, H, W
        f = F.sigmoid(f)  # B, T, C, H, W
        r = F.sigmoid(r)
        c_cur = c_zero

        second_term_f_mul_x_tilde = (1-f) * x_tilde
        all_c = []
        for t in range(seq_len):
            c_next = f[:, t, :, :, :] * c_cur + second_term_f_mul_x_tilde[:, t, :, :, :]
            c_cur = c_next
            all_c.append(c_next)
        all_c = F.stack(all_c, axis=1)  # B, T, C, H, W
        if self.input_dim != self.hidden_dim:
            x = self.residule_conv(x)
        x = x.reshape(mini_batch, seq_len, self.hidden_dim, height, width)
        all_h = r * F.tanh(all_c) + (1-r) * x  # B, T, C, H, W
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







class ConvLSTMCell(chainer.Chain):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
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
        super(ConvLSTMCell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        with self.init_scope():
            self.conv = L.Convolution2D(in_channels=self.input_dim + self.hidden_dim,
                                        out_channels=4 * self.hidden_dim,
                                        ksize=self.kernel_size, pad=self.padding,
                                        nobias=not self.bias)


    def __call__(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = F.concat([input_tensor, h_cur], axis=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        assert combined_conv.shape[1] % self.hidden_dim == 0
        assert combined_conv.shape[1]//self.hidden_dim == 4
        cc_i, cc_f, cc_o, cc_g = F.split_axis(combined_conv, combined_conv.shape[1]//self.hidden_dim, axis=1)
        i = F.sigmoid(cc_i)
        f = F.sigmoid(cc_f)
        o = F.sigmoid(cc_o)
        g = F.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * F.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size):
        return chainer.Variable(self.xp.zeros((batch_size, self.hidden_dim, self.height, self.width), dtype=self.xp.float32)), \
            chainer.Variable(self.xp.zeros((batch_size, self.hidden_dim, self.height, self.width), dtype=self.xp.float32))


class ConvLSTM(chainer.Chain):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
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
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.cell_list = []

        with self.init_scope():
            for i in range(self.num_layers):
                cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
                setattr(self, "cell_{}".format(i), ConvLSTMCell(input_size=(self.height, self.width),
                                                                input_dim=cur_input_dim, hidden_dim=self.hidden_dim[i],
                                                                kernel_size=self.kernel_size[i], bias=self.bias))
                self.cell_list.append("cell_{}".format(i))

    def __call__(self, input_tensor, hidden_state=None):
        """
            Parameters
            ----------
            input_tensor: todo
                5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
            hidden_state: todo
                None. todo implement stateful
            Returns
            -------
            last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = F.transpose(input_tensor, (1, 0, 2, 3, 4))
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.shape[0])
        layer_output_list = []
        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = getattr(self, self.cell_list[layer_idx])(input_tensor=cur_layer_input[:, t, :, :, :],
                                                                cur_state=[h,c])
                output_inner.append(h) # h is shape = B, C, H, W
            layer_output = F.stack(output_inner, axis=1)  # B, T, C, H, W, where T = seq_len
            cur_layer_input = layer_output
            layer_output_list.append(layer_output)
        if not self.return_all_layers:  # only return last layer of (B, T, C, H, W)
            layer_output_list = layer_output_list[-1:]

        return layer_output_list[0]

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(getattr(self, self.cell_list[i]).init_hidden(batch_size))
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