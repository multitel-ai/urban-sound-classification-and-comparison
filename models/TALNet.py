import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from models.DCASE_baseline import AutoPool
from activation.mish import Mish
from torchlibrosa.augmentation import SpecAugmentation


class ConvBlock(nn.Module):
    """Convolutionnal Block from the original TALNet implementation
    See https://github.com/MaigoAkisame/cmu-thesis
    """
    def __init__(
        self, n_input_feature_maps, n_output_feature_maps, kernel_size, batch_norm=False, pool_stride=None,
    ):
        super(ConvBlock, self).__init__()
        assert all(int(x) % 2 == 1 for x in kernel_size)
        self.n_input = n_input_feature_maps
        self.n_output = n_output_feature_maps
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.pool_stride = pool_stride
        self.conv = nn.Conv2d(
            int(self.n_input),
            int(self.n_output),
            int(self.kernel_size),
            padding=tuple(int(int(x) / 2) for x in self.kernel_size),
            bias=~batch_norm,
        )
        if batch_norm:
            self.bn = nn.BatchNorm2d(int(self.n_output))
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        if self.pool_stride is not None:
            x = F.max_pool2d(x, self.pool_stride)
        return x


class TALNet(nn.Module):
    """Original TALNet from Polyphonic Sound Event Detection with Weak Labeling
    See https://github.com/MaigoAkisame/cmu-thesis
    """
    def __init__(self, args, num_mels, num_classes):
        super(TALNet, self).__init__()
        self.__dict__.update(args.__dict__)  # Instill all args into self
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = num_mels
        self.output_size = num_classes
        self.conv = []
        pool_interval = self.n_conv_layers / self.n_pool_layers
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:  # this layer has pooling
                n_freq_bins /= 2
                n_output = self.embedding_size / n_freq_bins
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = self.embedding_size * 2 / n_freq_bins
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm=self.batch_norm, pool_stride=pool_stride,)
            self.conv.append(layer)
            self.__setattr__("conv" + str(i + 1), layer)
            n_input = n_output
        self.gru = nn.GRU(
            int(self.embedding_size), int(self.embedding_size / 2), 1, batch_first=True, bidirectional=True,
        )
        self.fc_prob = nn.Linear(self.embedding_size, self.output_size)
        if self.pooling == "att":
            self.fc_att = nn.Linear(self.embedding_size, self.output_size)
        # Better initialization
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.constant_(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal_(self.gru.weight_ih_l0_reverse)
        nn.init.constant_(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0_reverse)
        nn.init.constant_(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform_(self.fc_prob.weight)
        nn.init.constant_(self.fc_prob.bias, 0)
        if self.pooling == "att":
            nn.init.xavier_uniform_(self.fc_att.weight)
            nn.init.constant_(self.fc_att.bias, 0)
        if self.pooling == "auto":
            self.autopool = AutoPool(self.output_size)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))  # x becomes (batch, channel, time, freq)
        for i in range(len(self.conv)):
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv[i](x)  # x becomes (batch, channel, time, freq)
        x = x.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))  # x becomes (batch, time, embedding_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.gru(x)  # x becomes (batch, time, embedding_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))  # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)
        if self.pooling == "max":
            global_prob, _ = frame_prob.max(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "ave":
            global_prob = frame_prob.mean(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "lin":
            global_prob = (frame_prob * frame_prob).sum(dim=1) / frame_prob.sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "exp":
            global_prob = (frame_prob * frame_prob.exp()).sum(dim=1) / frame_prob.exp().sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "att":
            frame_att = F.softmax(self.fc_att(x), dim=1)
            global_prob = (frame_prob * frame_att).sum(dim=1)
            return global_prob, frame_prob, frame_att
        elif self.pooling == "auto":
            global_prob = self.autopool(frame_prob)
            return global_prob, frame_prob

    def predict(self, x, verbose=True, batch_size=100):
        # Predict in batches. Both input and output are numpy arrays.
        # If verbose == True, return all of global_prob, frame_prob and att
        # If verbose == False, only return global_prob
        result = []
        for i in range(0, len(x), batch_size):
            with torch.no_grad():
                input = Variable(torch.from_numpy(x[i : i + batch_size])).cuda()
                output = self.forward(input)
                if not verbose:
                    output = output[:1]
                result.append([var.data.cpu().numpy() for var in output])
        result = tuple(numpy.concatenate(items) for items in zip(*result))
        return result if verbose else result[0]


class PositionalEncoding(nn.Module):
    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim
        self.matrice_computed = False

        self.alpha = nn.Parameter(torch.FloatTensor([1.0]))

    def positionalencoding2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with " "odd dimension (got dim={:d})".format(d_model)
            )
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0.0, width).unsqueeze(1)
        pos_h = torch.arange(0.0, height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe

    def positionalencoding1d(self, d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

    def forward(self, x):
        if self.matrice_computed:
            return x + self.pe
        else:
            if self.ndim == 1:
                _, l, d = x.size()
                pe = self.positionalencoding1d(d, l).cuda()
            elif self.ndim == 2:
                _, d, h, w = x.size()
                pe = self.positionalencoding2d(d, h, w).cuda()
            self.register_buffer("pe", pe)
            self.matrice_computed = True
            return x + self.alpha * self.pe


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHead(nn.Module):
    """Multi-Head Attention module."""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        self.w_qs.bias.data.fill_(0)
        self.w_ks.bias.data.fill_(0)
        self.w_vs.bias.data.fill_(0)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.fill_(0)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()  # (batch_size, 80, 512)
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # (batch_size, T, 8, 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk, (batch_size*8, T, 64)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        # mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)  # (n_head * batch_size, T, 64), (n_head * batch_size, T, T)

        output = output.view(n_head, sz_b, len_q, d_v)  # (n_head, batch_size, T, 64)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv), (batch_size, T, 512)
        output = self.fc(output)
        output = self.layer_norm(output)
        output = F.relu_(self.dropout(output))
        return output


class Normed_Linear(nn.Linear):
    """ Linear Layer with weight and input L2 normalized
    Could lead to better 'geometric' space and could deal with imbalance dataset issues
    Args:
        in_features (int) : size of each input sample
        out_features (int) : size of each output sample
        bias (bool) : If False, the layer will not learn an additive bias.
    Shape:
        Input: (N, *, in_features)
        Output: (N, *, out_features)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        weight = self.weight / (torch.norm(((self.weight)), 2, 0) + 1e-5)
        x = x / (torch.norm(((x)), 2, -1) + 1e-5).unsqueeze(-1)
        return F.linear(x, weight, self.bias)


class AvgMaxPool2d(nn.Module):
    """ Average + Max Pooling layer
    Average Pooling added to Max Pooling
    Args:
        pool_stride (int, tuple) : controls the pooling stride
    """

    def __init__(self, pool_stride):
        super().__init__()
        self.pool_stride = pool_stride
        self.avgpool = nn.MaxPool2d(self.pool_stride)
        self.maxpool = nn.AvgPool2d(self.pool_stride)

    def forward(self, x):
        x1 = self.avgpool(x)
        x2 = self.maxpool(x)
        return x1 + x2


class Pooling_Head(nn.Module):
    """ Pooling layer for MIL
    Coda adapted from 'Polyphonic Sound Event Detection with Weak Labeling' Yun Wang github
    Link : https://github.com/MaigoAkisame/cmu-thesis
    Args:
        in_features (int) : size of each input sample
        out_features (int) : size of each output sample
        pooling (str) : pooling strategie, can be max, ave, lin, exp, att, auto
    """

    def __init__(self, in_features, out_features, pooling):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pooling = pooling

        if self.pooling == "att":
            self.fc_att = nn.Linear(self.in_features, self.out_features)
            nn.init.xavier_uniform_(self.fc_att.weight)
            nn.init.constant_(self.fc_att.bias, 0)
        elif self.pooling == "auto":
            self.autopool = AutoPool(self.out_features)

    def forward(self, frame_prob, x):
        if self.pooling == "max":
            global_prob, _ = frame_prob.max(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "ave":
            global_prob = frame_prob.mean(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "lin":
            global_prob = (frame_prob * frame_prob).sum(dim=1) / frame_prob.sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "exp":
            global_prob = (frame_prob * frame_prob.exp()).sum(dim=1) / frame_prob.exp().sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "att":
            frame_att = F.softmax(self.fc_att(x), dim=1)
            global_prob = (frame_prob * frame_att).sum(dim=1)
            return global_prob, frame_prob  # , frame_att
        elif self.pooling == "auto":
            global_prob = self.autopool(frame_prob)
            return global_prob, frame_prob


class ConvBlockTALNet(nn.Conv2d):
    """ TALNet ConvBlock with Weight Standardization (WS)
    Link to WS : https://arxiv.org/abs/1903.10520
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        norm=None,
        activation="relu",
        pool_stride=None,
        pool_strat="max",
    ):
        # Use padding depending on kernel size by default
        if padding == None:
            padding = tuple(int(int(x) / 2) for x in kernel_size)

        # Call __init__ of nn.Conv2d
        super(ConvBlockTALNet, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
        )

        # Initialize norm if needed (support None, Batch Norm, Group Norm)
        if norm == "GN":
            self.norm = True
            self.norm_layer = nn.GroupNorm(num_channels=self.out_channels, num_groups=32)
        elif norm == "BN":
            self.norm = True
            self.norm_layer = nn.BatchNorm2d(self.n_output)
        else:
            self.norm = False

        # Initialize activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "mish":
            self.activation = Mish()
        else:
            raise Exception("Incorrect argument!")

        # Initialize pooling if needed (support max and avg pooling)
        self.pool_stride = pool_stride
        if pool_strat == "max":
            self.pooling = nn.MaxPool2d(self.pool_stride)
        elif pool_strat == "avg":
            self.pooling = nn.AvgPool2d(self.pool_stride)
        elif pool_strat == "avg_max":
            self.pooling = AvgMaxPool2d(self.pool_stride)

        # Better Initialization
        nn.init.orthogonal_(self.weight)

    def forward(self, x):
        # Z-Norm weights (Weight Standardization)
        weight = self.weight - self.weight.view(self.weight.size(0), -1, 1, 1).mean(1, keepdim=True)
        std = weight.view(weight.size(0), -1, 1, 1).std(dim=1, keepdim=True) + 1e-5  # Avoid 0 div
        weight = weight / std.expand_as(weight)

        # Compute conv2D with Z-Normed weights
        x = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Apply norm if needed
        if self.norm:
            x = self.norm_layer(x)

        # Apply activation function
        x = self.activation(x)

        # Apply pooling if needed
        if self.pool_stride:
            x = self.pooling(x)
        return x


class ConvBlockTALNet2(nn.Conv2d):
    """ TALNet ConvBlock with Weight Standardization (WS)
    Link to WS : https://arxiv.org/abs/1903.10520
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        norm=None,
        activation="relu",
        pool_stride=None,
        pool_strat="max",
    ):
        # Use padding depending on kernel size by default
        if padding == None:
            padding = tuple(int(int(x) / 2) for x in kernel_size)

        # Call __init__ of nn.Conv2d
        super(ConvBlockTALNet2, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode,
        )

        # Initialize norm if needed (support None, Batch Norm, Group Norm)
        if norm == "GN":
            self.norm = True
            self.norm_layer = nn.GroupNorm(num_channels=self.out_channels, num_groups=32)
        elif norm == "BN":
            self.norm = True
            self.norm_layer = nn.BatchNorm2d(self.n_output)
        else:
            self.norm = False

        # Initialize activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "mish":
            self.activation = Mish()
        else:
            raise Exception("Incorrect argument!")

        # Initialize pooling if needed (support max and avg pooling)
        self.pool_stride = pool_stride
        if pool_strat == "max":
            self.pooling = nn.MaxPool2d(self.pool_stride)
        elif pool_strat == "avg":
            self.pooling = nn.AvgPool2d(self.pool_stride)
        elif pool_strat == "avg_max":
            self.pooling = AvgMaxPool2d(self.pool_stride)

        # Better Initialization
        nn.init.orthogonal_(self.weight)

    def forward(self, x):

        # Compute conv2D with Z-Normed weights
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Apply norm if needed
        if self.norm:
            x = self.norm_layer(x)

        # Apply activation function
        x = self.activation(x)

        # Apply pooling if needed
        if self.pool_stride:
            x = self.pooling(x)
        return x


class TALNetV2(nn.Module):
    """ Improved TALNet architecture
    Coda adapted from 'Polyphonic Sound Event Detection with Weak Labeling' Yun Wang github
    Link : https://github.com/MaigoAkisame/cmu-thesis
    Args:
        num_mels (int) : number of mels of spectrogram
        num_classes (int) : number of classes
        n_conv_layers (int) : number of ConvBlockTALNet
        kernel_size (tuple) : kernel size of ConvBlockTALNet
        n_pool_layers (int) : number of pooling layers in the ConvBlock part
        embedding_size (int) : size of embeddings
        norm (str) : norm to use in ConvBlockTALNet
        pooling (str) : pooling strategie of the head
        dropout (float) : dropout applied before ConvBlockTALNet
        conv_pool_strat (str) : pooling strategie used in ConvBlockTALNet
        conv_activation (str) : activation used in ConvBlockTALNet
        n_head (int) : number of head inside attention block
        d_kv (int) : size of key and values of attention block
        dropout_transfo (float) : dropout applied inside attention block
    """

    def __init__(
        self,
        num_mels,
        num_classes,
        n_conv_layers=10,
        kernel_size=(3, 3),
        n_pool_layers=5,
        embedding_size=1024,
        norm="GN",
        pooling="att",
        dropout=0.0,
        conv_pool_strat="max",
        conv_activation="relu",
        n_head=8,
        d_kv=128,
        dropout_transfo=0,
    ):
        super(TALNetV2, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.n_pool_layers = n_pool_layers
        self.embedding_size = embedding_size
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.conv_pool_strat = conv_pool_strat
        self.conv_activation = conv_activation
        self.n_head = n_head
        self.d_k = self.d_v = d_kv
        self.dropout_transfo = dropout_transfo
        self.input_n_freq_bins = n_freq_bins = num_mels
        self.output_size = num_classes
        assert self.n_conv_layers % self.n_pool_layers == 0

        pool_interval = self.n_conv_layers / self.n_pool_layers
        n_input = 1
        self.conv = []
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:  # this layer has pooling
                n_freq_bins /= 2
                n_output = self.embedding_size / n_freq_bins
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = self.embedding_size * 2 / n_freq_bins
                pool_stride = None
            layer = ConvBlockTALNet2(  # WARNING : PUT IT BACK TO ConvBlockTALNet
                int(n_input),
                int(n_output),
                self.kernel_size,
                norm=self.norm,
                pool_stride=pool_stride,
                pool_strat=self.conv_pool_strat,
                activation=self.conv_activation,
            )
            self.conv.append(layer)
            self.__setattr__("conv" + str(i + 1), layer)
            n_input = n_output

        # self.positional_enc = PositionalEncoding()
        self.self_attention_1 = MultiHead(self.n_head, self.embedding_size, self.d_k, self.d_v, self.dropout_transfo)
        self.fc_prob = Normed_Linear(self.embedding_size, self.output_size)
        self.pooling_head = Pooling_Head(self.embedding_size, self.output_size, self.pooling)

    def forward(self, x):
        # x is (bs, time, freq)
        # x = x.unsqueeze(1)  # x becomes (batch, channel, time, freq)
        for conv_layer in self.conv:
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv_layer(x)  # x becomes (batch, channel, time, freq)
        x = x.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))  # x becomes (batch, time, embedding_size)
        # x = self.positional_enc(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.self_attention_1(x, x, x)  # x becomes (batch, time, embedding_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))  # shape of frame_prob: (batch, time, output_size)
        return self.pooling_head(frame_prob, x)

class TALNetV3NoMeta(nn.Module):
    def __init__(self, args, num_mels, num_classes):
        super(TALNetV3NoMeta, self).__init__()
        self.__dict__.update(args)  # Install all args into self
        assert self.n_conv_layers % self.n_pool_layers == 0
        self.input_n_freq_bins = n_freq_bins = num_mels
        self.output_size = num_classes
        self.n_head = self.transfo_head
        self.d_k = self.d_v = 128
        # Conv
        self.conv = []
        self.conv_v2 = []
        pool_interval = self.n_conv_layers / self.n_pool_layers
        n_input = 1
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:  # this layer has pooling
                n_freq_bins /= 2
                n_output = self.embedding_size / n_freq_bins
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = self.embedding_size * 2 / n_freq_bins
                pool_stride = None
            layer = ConvBlock(n_input, n_output, self.kernel_size, batch_norm=self.batch_norm, pool_stride=pool_stride,)
            self.conv.append(layer)
            self.__setattr__("conv" + str(i + 1), layer)
            layer_v2 = ConvBlockTALNet2(
                int(n_input),
                int(n_output),
                (int(self.kernel_size), int(self.kernel_size)),
                norm="GN",
                pool_stride=pool_stride,
                pool_strat="max",
                activation="mish",
            )
            self.conv_v2.append(layer_v2)

            self.__setattr__("conv_v2" + str(i + 1), layer_v2)
            n_input = n_output
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2, freq_drop_width=num_mels//8, freq_stripes_num=2,
        )
        self.bn0 = nn.BatchNorm2d(num_mels)

        # Temp (Transfo + GRU)
        self.gru = nn.GRU(
            int(self.embedding_size), int(self.embedding_size / 2), 1, batch_first=True, bidirectional=True,
        )
        self.multihead_v2 = MultiHead(self.n_head, self.embedding_size, self.d_k, self.d_v, self.dropout_transfo)
        # FC
        # self.att_block = AttBlock(n_in=(self.embedding_size * 2 + self.meta_emb * self.num_meta), n_out=self.output_size, activation='sigmoid')
        self.fc_prob = nn.Linear(self.embedding_size * 2, self.output_size)
        if self.pooling == "att":
            self.fc_att = nn.Linear(self.embedding_size * 2, self.output_size,)

        # Better initialization
        nn.init.orthogonal_(self.gru.weight_ih_l0)
        nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0)
        nn.init.constant_(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal_(self.gru.weight_ih_l0_reverse)
        nn.init.constant_(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0_reverse)
        nn.init.constant_(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform_(self.fc_prob.weight)
        nn.init.constant_(self.fc_prob.bias, 0)
        if self.pooling == "att":
            nn.init.xavier_uniform_(self.fc_att.weight)
            nn.init.constant_(self.fc_att.bias, 0)
        if self.pooling == "auto":
            self.autopool = AutoPool(self.output_size)

    def forward(self, x):
        x = x.view((-1, 1, x.size(1), x.size(2)))  # x becomes (batch, channel, time, freq)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        x_v2 = x
        for i in range(len(self.conv)):
            if self.dropout_AS > 0:
                x = F.dropout(x, p=self.dropout_AS, training=self.training)
            x = self.conv[i](x)  # x becomes (batch, channel, time, freq)
        x = x.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))  # x becomes (batch, time, embedding_size)
        if self.dropout_AS > 0:
            x = F.dropout(x, p=self.dropout_AS, training=self.training)
        x, _ = self.gru(x)

        for i in range(len(self.conv_v2)):
            if self.dropout > 0:
                x_v2 = F.dropout(x_v2, p=self.dropout, training=self.training)
            x_v2 = self.conv_v2[i](x_v2)  # x becomes (batch, channel, time, freq)
        x_v2 = x_v2.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, freq)
        x_v2 = x_v2.view((-1, x_v2.size(1), x_v2.size(2) * x_v2.size(3)))  # x becomes (batch, time, embedding_size)
        if self.dropout > 0:
            x_v2 = F.dropout(x_v2, p=self.dropout, training=self.training)
        x_v2 = self.multihead_v2(x_v2, x_v2, x_v2)

        x = torch.cat([x, x_v2], 2)

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        fcprob = self.fc_prob(x)
        frame_prob = torch.sigmoid(fcprob)  # shape of frame_prob: (batch, time, output_size)
        frame_prob = torch.clamp(frame_prob, 1e-7, 1 - 1e-7)

        if self.pooling == "max":
            global_prob, _ = frame_prob.max(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "ave":
            global_prob = frame_prob.mean(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "lin":
            global_prob = (frame_prob * frame_prob).sum(dim=1) / frame_prob.sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "exp":
            global_prob = (frame_prob * frame_prob.exp()).sum(dim=1) / frame_prob.exp().sum(dim=1)
            return global_prob, frame_prob
        elif self.pooling == "att":
            frame_att = F.softmax(self.fc_att(x), dim=1)
            global_prob = (frame_prob * frame_att).sum(dim=1)
            return torch.clamp(global_prob, 0, 1), frame_prob, frame_att, fcprob
        elif self.pooling == "auto":
            global_prob = self.autopool(frame_prob)
            return global_prob, frame_prob


class TALNetDWC(nn.Module):
    """ Improved TALNet architecture
    Coda adapted from 'Polyphonic Sound Event Detection with Weak Labeling' Yun Wang github
    Link : https://github.com/MaigoAkisame/cmu-thesis
    Args:
        num_mels (int) : number of mels of spectrogram
        num_classes (int) : number of classes
        n_conv_layers (int) : number of ConvBlockTALNet
        kernel_size (tuple) : kernel size of ConvBlockTALNet
        n_pool_layers (int) : number of pooling layers in the ConvBlock part
        embedding_size (int) : size of embeddings
        norm (str) : norm to use in ConvBlockTALNet
        pooling (str) : pooling strategie of the head
        dropout (float) : dropout applied before ConvBlockTALNet
        conv_pool_strat (str) : pooling strategie used in ConvBlockTALNet
        conv_activation (str) : activation used in ConvBlockTALNet
        n_head (int) : number of head inside attention block
        d_kv (int) : size of key and values of attention block
        dropout_transfo (float) : dropout applied inside attention block
    """

    def __init__(
        self,
        num_mels,
        num_classes,
        n_conv_layers=10,
        kernel_size=(3, 3),
        n_pool_layers=5,
        embedding_size=1024,
        norm="GN",
        pooling="att",
        dropout=0.0,
        conv_pool_strat="max",
        conv_activation="relu",
        n_head=8,
        d_kv=128,
        dropout_transfo=0,
    ):
        super(TALNetDWC, self).__init__()
        self.n_conv_layers = n_conv_layers
        self.kernel_size = kernel_size
        self.n_pool_layers = n_pool_layers
        self.embedding_size = embedding_size
        self.norm = norm
        self.pooling = pooling
        self.dropout = dropout
        self.conv_pool_strat = conv_pool_strat
        self.conv_activation = conv_activation
        self.n_head = n_head
        self.d_k = self.d_v = d_kv
        self.dropout_transfo = dropout_transfo
        self.input_n_freq_bins = n_freq_bins = num_mels
        self.output_size = num_classes
        assert self.n_conv_layers % self.n_pool_layers == 0

        pool_interval = self.n_conv_layers / self.n_pool_layers
        n_input = 1
        self.conv = []
        for i in range(self.n_conv_layers):
            if (i + 1) % pool_interval == 0:  # this layer has pooling
                n_freq_bins /= 2
                n_output = self.embedding_size / n_freq_bins
                pool_stride = (2, 2) if i < pool_interval * 2 else (1, 2)
            else:
                n_output = self.embedding_size * 2 / n_freq_bins
                pool_stride = None
            layer = DepthWiseSeparableConvBlock(  # WARNING : PUT IT BACK TO ConvBlockTALNet
                int(n_input), int(n_output), kernel_size, pool_stride=pool_stride, pool_strat=self.conv_pool_strat,
            )
            self.conv.append(layer)
            self.__setattr__("conv" + str(i + 1), layer)
            n_input = n_output

        self.self_attention_1 = MultiHead(self.n_head, self.embedding_size, self.d_k, self.d_v, self.dropout_transfo)
        self.fc_prob = Normed_Linear(self.embedding_size, self.output_size)
        self.pooling_head = Pooling_Head(self.embedding_size, self.output_size, self.pooling)

    def forward(self, x):
        # x is (bs, time, freq)
        # x = x.unsqueeze(1)  # x becomes (batch, channel, time, freq)
        for conv_layer in self.conv:
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv_layer(x)  # x becomes (batch, channel, time, freq)
        x = x.permute(0, 2, 1, 3).contiguous()  # x becomes (batch, time, channel, freq)
        x = x.view((-1, x.size(1), x.size(2) * x.size(3)))  # x becomes (batch, time, embedding_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.self_attention_1(x, x, x)  # x becomes (batch, time, embedding_size)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        frame_prob = torch.sigmoid(self.fc_prob(x))  # shape of frame_prob: (batch, time, output_size)
        return self.pooling_head(frame_prob, x)
