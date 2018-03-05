import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
from itertools import combinations

linear_init = chainer.initializers.LeCunUniform()




def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3

    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.

    """

    batch, units, length = x.shape
    e = F.transpose(x, (0, 2, 1)).reshape(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]
    e = F.transpose(e.reshape((batch, length, out_units)), (0, 2, 1))
    assert(e.shape == (batch, out_units, length))
    return e


def sentence_block_embed(embed, x):
    """ Change implicitly embed_id function's target to ndim=2

    Apply embed_id for array of ndim 2,
    shape (batchsize, sentence_length),
    instead for array of ndim 1.

    """

    batch, length = x.shape
    _, units = embed.W.shape
    e = embed(x.reshape((batch * length, )))
    assert(e.shape == (batch * length, units))
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1)) # cut along axis=0 to batch slices
    assert(e.shape == (batch, units, length))
    return e


class LayerNormalizationSentence(L.LayerNormalization):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).

    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__(*args, **kwargs)

    def __call__(self, x):
        y = seq_func(super(LayerNormalizationSentence, self).__call__, x) # shape = batch, out_units, length
        return y


class ConvolutionSentence(L.Convolution2D):

    """ Position-wise Linear Layer for Sentence Block

    Position-wise linear layer for array of shape
    (batchsize, dimension, sentence_length)
    can be implemented a convolution layer.

    """

    def __init__(self, in_channels, out_channels,
                 ksize=1, stride=1, pad=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(ConvolutionSentence, self).__init__(
            in_channels, out_channels,
            ksize, stride, pad, nobias,
            initialW, initial_bias)

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vector block. Its shape is
                (batchsize, in_channels, sentence_length).

        Returns:
            ~chainer.Variable: Output of the linear layer. Its shape is
                (batchsize, out_channels, sentence_length).

        """
        x = F.expand_dims(x, axis=3) # shape = (batch_size, in_channels, sentence_length, 1)
        # 1d fc along sentence_length in parallel can be simulated by 1x1 convolution
        y = super(ConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)
        return y

class AdaptConvolutionSentence(L.Convolution2D):

    def __init__(self, n_layers, in_channels, out_channels,
                 ksize=1, stride=1, pad=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(AdaptConvolutionSentence, self).__init__(
            in_channels, out_channels,
            ksize, stride, pad, nobias,
            initialW, initial_bias)
        # n_layers is fake parameter just to adaptive
    def __call__(self, x_lst):
        """Applies the linear layer.

        Args:
            x_lst (~chainer.Variable): List of input vector block. Each shape is
                (sentence_length, in_channels).

        Returns:
            list of ~chainer.Variable: Output of the linear layer. Its shape is
                (sentence_length, out_channels).

        """
        x = F.stack(x_lst)  # shape = (B, T, D)
        x = F.transpose(x, (0,2,1))
        x = F.expand_dims(x, axis=3) # shape = (batch_size, in_channels, sentence_length, 1)
        # 1d fc along sentence_length in parallel can be simulated by 1x1 convolution
        y = super(AdaptConvolutionSentence, self).__call__(x)
        y = F.squeeze(y, axis=3)  # shape = (B, D, T)
        return list(F.split_axis(F.transpose(y, axes=(0,2,1)), y.shape[0], axis=0,force_tuple=True))  # list of T,D




class MultiHeadAttention(chainer.Chain):

    """ Multi Head Attention Layer for Sentence Blocks

    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.

    """

    def __init__(self, n_units, h=8, self_attention=True):
        # n_units is d_k,also as query and key's input dim
        super(MultiHeadAttention, self).__init__()

        with self.init_scope():
            if self_attention:
                self.W_QKV = ConvolutionSentence(
                    n_units, n_units * 3, nobias=True,
                    initialW=linear_init)
            else:
                self.W_Q = ConvolutionSentence(
                    n_units, n_units, nobias=True,
                    initialW=linear_init)
                self.W_KV = ConvolutionSentence(
                    n_units, n_units * 2, nobias=True,
                    initialW=linear_init)

            self.finishing_linear_layer = ConvolutionSentence(
                n_units, n_units, nobias=True,
                initialW=linear_init)  # input and output channel are same
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.is_self_attention = self_attention

    def __call__(self, x, z=None, mask=None):
        # pass in x and z shape = (batch, D, T), T is sequence length
        xp = self.xp
        h = self.h
        if self.is_self_attention:
            # expand dimension to 3 times first, then split, thus information of Q and V comes from x in self attenion.
            # self.W_QKV output (batchsize, n_units * 3, sentence_length), then split by axis=1
            Q, K, V = F.split_axis(self.W_QKV(x))# node_feature shape = (F,T,D), z shape = (F',T,D) indicate neighbor features, F' may not equals with F
        # F' may be edge_number in each frame, because it may come from EdgeRNN outputV(x), 3, axis=1)
        else:
            Q = self.W_Q(x) # Query is come from x, the Value we want to extract is from z
            K, V = F.split_axis(self.W_KV(z), 2, axis=1)

        batch, n_units, n_querys = Q.shape  # n_querys is count of query which can also named as "T_q"
        _, _, n_keys = K.shape # n_keys is sentence_length which can also named as "T_k"

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching: h * batch_size
        # all together at once for efficiency
        # batch, n_units, n_querys : split by axis=1 to cut n_heads slice,
        # each slice shape is (batch, n_units//8, n_querys)
        # then concat in axis=0
        batch_Q = F.concat(F.split_axis(Q, h, axis=1), axis=0)
        batch_K = F.concat(F.split_axis(K, h, axis=1), axis=0)
        batch_V = F.concat(F.split_axis(V, h, axis=1), axis=0)
        assert(batch_Q.shape == (batch * h, n_units // h, n_querys))
        assert(batch_K.shape == (batch * h, n_units // h, n_keys))
        assert(batch_V.shape == (batch * h, n_units // h, n_keys))

        # Notice that this formula is different from paper Eqn 1., this time is transpose of Q matrix,
        # This F.matmul actually perform batch_matmul
        batch_A = F.matmul(batch_Q, batch_K, transa=True) \
            * self.scale_score  # shape = batch * h, T_q , T_k，matrix mat along the dimension of n_units//8
        # if mask==False，use -np.inf, so above code:[mask] * h
        if mask is not None:
            mask = xp.concatenate([mask] * h, axis=0)
            batch_A = F.where(mask, batch_A, xp.full(batch_A.shape, -np.inf, 'f'))
        batch_A = F.softmax(batch_A, axis=2)  # axis=2 means attend along n_keys axis. Thus you can softly choose keys
        batch_A = F.where(
            xp.isnan(batch_A.data), xp.zeros(batch_A.shape, 'f'), batch_A) # push nan value to zero
        assert(batch_A.shape == (batch * h, n_querys, n_keys))

        # Calculate Weighted Sum before broad_cast (batch_A stores weights)
        # batch_A shape = (batch * h, 1, n_querys, n_keys); batch_V shape = (batch * h, n_units//8, 1, n_keys)
        batch_A, batch_V = F.broadcast(
            batch_A[:, None], batch_V[:, :, None]) # n_units//8 就是d_v
        # batch_C shape = batch * h, n_units//8, n_querys, n_keys, axis=3 means weighted sum along sequence
        batch_C = F.sum(batch_A * batch_V, axis=3) # shape = batch * h, n_units//8, n_querys
        assert(batch_C.shape == (batch * h, n_units // h, n_querys))
        # slice in h piece，then concat along axis=1, shape = (batch, n_units//8 * 8, n_querys), head = 8
        C = F.concat(F.split_axis(batch_C, h, axis=0), axis=1)
        # Notice that there is no n_keys in shape any more, because weighed sum already eliminated this dimension.
        assert(C.shape == (batch, n_units, n_querys))
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(chainer.Chain):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4  # hidden layer dimension is big than input/output dimension
        with self.init_scope():
            self.W_1 = ConvolutionSentence(n_units, n_inner_units,
                                           initialW=linear_init)
            self.W_2 = ConvolutionSentence(n_inner_units, n_units,
                                           initialW=linear_init)
            self.act = F.leaky_relu

    def __call__(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class EncoderLayer(chainer.Chain):

    def __init__(self, n_units, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h)  # n_units是输入数据的维度
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, e, xx_mask=None):
        sub = self.self_attention(e, e, xx_mask)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_2(e)
        return e  # shape = (batch, n_units, n_querys) alias of (B,D,T)


class DecoderLayer(chainer.Chain):

    def __init__(self, n_units, h=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        with self.init_scope():
            self.self_attention = MultiHeadAttention(n_units, h)
            self.source_attention = MultiHeadAttention(
                n_units, h, self_attention=False)
            self.feed_forward = FeedForwardLayer(n_units)
            self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)
            self.ln_3 = LayerNormalizationSentence(n_units, eps=1e-6)
        self.dropout = dropout

    def __call__(self, target_block, encoded_block, xy_mask, yy_mask):
        # encoded_block is all neighbors' encoded message, shape = (neighbors, batch, unit, seq_length),
        # encoded_block also (neighbors, B, D, T)
        sub = self.self_attention(target_block, target_block, yy_mask)
        e = target_block + F.dropout(sub, self.dropout)
        e = self.ln_1(e)  # shape = (batch, unit, seq_length)

        sub = self.source_attention(e, encoded_block, xy_mask)  # e is query Q

        e = e + F.dropout(sub, self.dropout)
        e = self.ln_2(e)
        sub = self.feed_forward(e)
        e = e + F.dropout(sub, self.dropout)
        e = self.ln_3(e)
        return e


class Encoder(chainer.Chain):

    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = EncoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, xx_mask=None):
        for name in self.layer_names:
            e = getattr(self, name)(e, xx_mask=None)
        return e


class Decoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Decoder, self).__init__()
        self.layer_names = []
        for i in range(1, n_layers + 1):
            name = 'l{}'.format(i)
            layer = DecoderLayer(n_units, h, dropout)
            self.add_link(name, layer)
            self.layer_names.append(name)

    def __call__(self, e, source, xy_mask, yy_mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, source, xy_mask, yy_mask)
        return e


class AttentionBlock(chainer.Chain):

    def __init__(self, n_layers, n_units, out_size, h=8, dropout=0.5, max_length=1500):
        super(AttentionBlock, self).__init__()
        self.layer_names = []
        self.n_units = n_units
        self.position_encoding_block = None
        self.initialize_position_encoding(max_length, n_units)

        with self.init_scope():
            self.encoder = Encoder(n_layers, n_units, h, dropout)
            self.final_linear = ConvolutionSentence(in_channels=n_units, out_channels=out_size)

    def initialize_position_encoding(self, length, n_units):
        xp = self.xp
        # Implementation in the Google tensor2tensor repo
        channels = n_units
        position = xp.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (
            xp.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * xp.exp(
            xp.arange(num_timescales).astype('f') * - log_timescale_increment)
        scaled_time = \
            xp.expand_dims(position, 1) * \
            xp.expand_dims(inv_timescales, 0)
        signal = xp.concatenate(
            [xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
        signal = xp.reshape(signal, [1, length, channels]) # shape = [1, length, channels(n_units)]
        self.position_encoding_block = signal




    def __call__(self, x_lst, mask=None):
        # x is shape = (batch, T, D)
        x = F.stack(x_lst)
        batch, length, unit = x.shape
        x += self.xp.array(self.position_encoding_block[:, :length, :])
        x = F.transpose(x, axes=(0, 2, 1)) # shape = (batch, D, T)
        h = self.encoder(x, mask)  # self attention
        h = self.final_linear(h)
        # list of shape = (out_size, T)
        return list([F.squeeze(e) for e in F.split_axis(F.transpose(h, axes=(0, 2, 1)), 1, axis=0, force_tuple=True)])




# reproduce of paper: <Relation Networks for Object Detection> arXiv:1711.11575,
# input : the whole image's object appearance feature shape=(N,fA) and object geometry feature shape=(N,fg)
# output: the whole image's relation feature which is shape=(N,F')
class ObjectRelationModule(chainer.Chain):
    # num_relations is used to concatenate, see paper <Relation Networks for Object Detection> Figure 2(Left)
    def __init__(self, in_size, out_size, add_self, num_relations=16, d_k=64, d_g=64, frame_node_num=None):
        assert out_size % num_relations == 0
        super(ObjectRelationModule, self).__init__()
        self.add_self = add_self
        self.num_relations = num_relations
        self.w_v_outsize = out_size // num_relations
        self.d_g = d_g
        self.d_k = d_k
        self.out_size = out_size
        self.W_K_lst = []
        self.W_Q_lst = []
        self.W_V_lst = []
        self.W_G_lst = []
        self.use_spatio_temporal_graph = False
        if frame_node_num is not None:
            self.frame_node_num = frame_node_num
            self.use_spatio_temporal_graph = True

        with self.init_scope():
            for i in range(num_relations):
                self.add_link("W_K_{}".format(i), L.Linear(in_size, d_k))
                self.W_K_lst.append("W_K_{}".format(i))
                self.add_link("W_Q_{}".format(i), L.Linear(in_size, d_k))
                self.W_Q_lst.append("W_Q_{}".format(i))
                self.add_link("W_V_{}".format(i), L.Linear(in_size, out_size//num_relations))
                self.W_V_lst.append("W_V_{}".format(i))
                self.add_link("W_G_{}".format(i), L.Linear(d_g, 1)) # after position embedding is d_g, then transform to scalar weight
                self.W_G_lst.append("W_G_{}".format(i))


    def position_encoding(self, low_dim_data, out_channel):
        # e.g. low_dim_data = N x 4 where 4 is coordinates number of box
        pieces = low_dim_data.shape[1]
        assert (out_channel % (2*pieces) == 0)
        xp = self.xp
        num_timescales = out_channel // (2 * pieces)
        log_timescale_increment = (
                xp.log(10000. / 1.) / (float(num_timescales) - 1))  # float(num_timescales) - 1 = paper's d_model
        inv_timescales = 1. * xp.exp(
            xp.arange(num_timescales).astype('f') * -log_timescale_increment)  # shape= (num_timescales,)
        signal = []
        for piece in range(pieces):
            scaled_time = \
                xp.expand_dims(low_dim_data[:, piece], 1) * xp.expand_dims(inv_timescales, 0) # shape = (N, num_timescales)
            signal_piece = xp.concatenate(
                [xp.sin(scaled_time), xp.cos(scaled_time)], axis=1) # shape = (N, 2 * num_timescales) = (N, out_channel//pieces)
            signal.append(signal_piece)
        signal = xp.concatenate(signal, axis=1)  # shape = (N, out_channel) = (N, d_g)
        return signal


    # About how to use the following code. First: create [1, out_channel, length] shape tensor, then
    # e.g. we have 4 dimension input. Just slice the tensor as we need in each dimension.
    def initial_position_encoding(self, out_channel, length):
        assert(out_channel % 2 == 0)
        """
        Gets a bunch of sinusoids of different frequencies.
        Each channel of the input Tensor is incremented by a sinusoid of a different
        frequency and phase.
        This allows attention to learn to use absolute and relative positions.
        Timing signals should be added to some precursors of both the query and the
        memory inputs to attention.
        The use of relative position is possible because sin(x+y)=sin(x)cos(y)+cos(x)sin(y) and
         cos(x+y)= cos(x)·cos(y)-sin(x)·sin(y) 
        can be experessed in terms of y, sin(x) and cos(x).  
        In particular, we use a geometric sequence of timescales starting with
        min_timescale and ending with max_timescale.  The number of different
        timescales is equal to channels / 2. For each timescale, we
        generate the two sinusoidal signals sin(timestep/timescale) and
        cos(timestep/timescale).  All of these sinusoids are concatenated in
        the channels dimension.
    
        Args:
        length: scalar, length of timing signal sequence.
        channels: scalar, size of timing embeddings to create. The number of
            different timescales is equal to channels / 2.
        min_timescale: a float
        max_timescale: a float
        Returns:
            a Tensor of timing signals [1, out_channel, length]
        """
        xp = self.xp

        position = xp.arange(length, dtype='f')
        num_timescales = out_channel // 2
        log_timescale_increment = (
                xp.log(10000. / 1.) / (float(num_timescales) - 1))  # float(num_timescales) - 1 = paper's d_model
        inv_timescales = 1. * xp.exp(
            xp.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = \
            xp.expand_dims(position, 1) * xp.expand_dims(inv_timescales, 0)
        signal = xp.concatenate(
            [xp.sin(scaled_time), xp.cos(scaled_time)], axis=1)
        signal = xp.reshape(signal, [1, length, out_channel])
        return xp.transpose(signal, (0,2,1))

    def encode_box_offset(self, box1, box2):
        xp = self.xp
        box1_x1y1_x2y2 = xp.reshape(box1, (-1, 2, 2))
        box1_x1y1, box1_x2y2 = xp.split(box1_x1y1_x2y2, 2, axis=1)
        box1_w_h = box1_x2y2 - box1_x1y1  # shape = (N, 2)
        box1_x_y = (box1_x2y2 + box1_x1y1) * 0.5 # shape = (N,2)

        box2_x1y1_x2y2 = xp.reshape(box2, (-1,2,2))
        box2_x1y1, box2_x2y2= xp.split(box2_x1y1_x2y2, 2, axis=1)  # shape = (N,2)
        box2_w_h = box2_x2y2 - box2_x1y1  # shape = (N,2)
        box2_x_y = (box2_x2y2 + box2_x1y1) * 0.5  # shape = (N,2)


        txty = xp.log( xp.abs(box2_x_y - box1_x_y) / box1_w_h )
        twth = xp.log( box2_w_h/ box1_w_h )
        encoded = xp.concatenate([txty, twth], axis=1) # (-1,2,2)
        return encoded.reshape(-1, 4)  # the same as paper formula

    def regular_graph_output(self, f_A, f_G):  # f_A is appearance feature shape = (N,D), f_G is geometry feature shape = (N,4)
        assert f_A.shape[0] == f_G.shape[0]
        if self.add_self:
            assert f_A.shape[1] == self.out_size
        N = f_G.shape[0]
        geo_dim = f_G.shape[1]
        f_R = []
        for nr in range(self.num_relations):
            f_G = F.tile(f_G, (1, N))  # shape = (N, 4 * N)
            f_G_1 = F.reshape(f_G, (N * N, geo_dim))  # after tile: N x (4 x N), then N^2 x 4
            f_G_2 = F.tile(f_G, (N, 1))  # shape = (N*N, 4)
            encoded_offset = self.encode_box_offset(f_G_1, f_G_2)  # shape = (N*N, 4)
            # paper formula (5), shape = (N,N)
            w_G = F.relu(getattr(self, self.W_G_lst[nr])(self.position_encoding(encoded_offset, self.d_g)))
            w_G = F.reshape(w_G, shape=(N,N))
            # paper formula (4), shape = (N,N)
            w_K_result = getattr(self, self.W_K_lst[nr])(f_A)  # shape = (N, d_k)
            w_Q_transpose_result = F.transpose(getattr(self, self.W_Q_lst[nr])(f_A)) # shape = (d_k, N)
            w_A = F.matmul(w_K_result, w_Q_transpose_result)  # shape = (N,N)
            # paper formula (3), shape = (N,N)
            w_A = w_A + F.log(w_G)
            w = F.softmax(w_A, axis=1)
            # w = w_G * F.exp(w_A) / F.sum(w_G * F.exp(w_A), axis=1) # denominator shape = (N,1) numerator shape = (N,N)
            # paper formula (2), weight sum = matmul:(N,N) x (N, out_size//nr) = (N, out_size//nr)
            f_R_nr = F.matmul(w, getattr(self, self.W_V_lst[nr])(f_A))
            f_R.append(f_R_nr)
        if self.add_self:
            return f_A + F.concat(f_R, axis=1)
        return F.concat(f_R, axis=1)

    def st_graph_output(self, f_A, f_G):  # f_A shape = (N,D), f_G shape = (N,4)
        assert f_A.shape[0] == f_G.shape[0]
        if self.add_self:
            assert f_A.shape[1] == self.out_size
        N = f_G.shape[0]
        assert N % self.frame_node_num == 0
        T = N//self.frame_node_num
        geo_dim = f_G.shape[1]
        f_A_orig = f_A
        f_G = F.reshape(f_G, (T, self.frame_node_num, geo_dim))
        f_A = F.reshape(f_A, (T, self.frame_node_num, f_A.shape[-1]))
        assert f_A_orig.ndim == 2, f_A_orig.ndim
        f_R = []
        for nr in range(self.num_relations):
            f_G_1 = F.tile(f_G, (1, 1, F))  # shape = (T, F, 4 * F)
            f_G_1 = F.reshape(f_G_1, (T, self.frame_node_num ** 2, geo_dim))  # after tile: (T, F, (4 x F)) then (T,F^2,4)
            f_G_2 = F.tile(f_G, (1, F, 1))  # shape = (T, F*F, 4)
            encoded_offset = self.encode_box_offset(f_G_1.reshape(-1, geo_dim), f_G_2.reshape(-1,geo_dim))  # shape = (TxFxF, 4)
            # paper formula (5), shape = (T,F,F)
            w_G = F.relu(getattr(self, self.W_G_lst[nr])(self.position_encoding(encoded_offset, self.d_g))) # TxFxF,1
            w_G = F.reshape(w_G, shape=(T, self.frame_node_num,self.frame_node_num)) # shape = (T,F,F)
            # paper formula (4), shape = (N,N)
            w_K_result = getattr(self, self.W_K_lst[nr])(f_A_orig).reshape(T, self.frame_node_num, self.d_k)  # shape = (T, F, d_k)
            w_Q_transpose_result = F.transpose(getattr(self, self.W_Q_lst[nr])(f_A_orig).
                                               reshape(T, self.frame_node_num, self.d_k), axes=(0,2,1))  # shape = (T, d_k, F)
            w_A = F.matmul(w_K_result, w_Q_transpose_result)  # shape = (T,F,F)
            w_A = w_A + F.log(w_G)
            # paper formula (3), shape = (T,F,F)
            w = F.softmax(w_A, axis=2) # original paper formula (3) is weighted softmax, because chainer does not provide such weighed-softmax
            # we instead only element-wise dot here, then softmax
            # w = w_G * F.exp(w_A) / F.sum(w_G * F.exp(w_A), axis=1)  # denominator shape = (N,1) numerator shape = (N,N)
            # paper formula (2), weight sum = matmul:(T,F,F) x (T, F, out_size//nr) = (T, F, out_size//nr)
            f_R_nr = F.matmul(w, getattr(self, self.W_V_lst[nr])(f_A_orig).reshape(T,
                                                                            self.frame_node_num, self.w_v_outsize))
            f_R.append(f_R_nr)
        if self.add_self:
            return f_A + F.concat(f_R, axis=2).reshape(N, self.out_size)
        return F.concat(f_R, axis=2).reshape(N, self.out_size)

    def __call__(self, f_A, f_G):
        if self.use_spatio_temporal_graph:
            return self.st_graph_output(f_A, f_G)
        return self.regular_graph_output(f_A, f_G)