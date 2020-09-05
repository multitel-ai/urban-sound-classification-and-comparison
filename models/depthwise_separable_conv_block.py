# CREDITS : https://github.com/dr-costas/dnd-sed

from functools import reduce
from typing import Union, Tuple, MutableSequence, List, Optional

from torch import Tensor, sigmoid, clamp
from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU, MaxPool2d, AvgPool2d, Linear, Sequential, Dropout2d, ReLU
from models.TALNet import Pooling_Head

__author__ = "Konstantinos Drossos -- Tampere University"
__docformat__ = "reStructuredText"
__all__ = ["DepthWiseSeparableConvBlock"]


def apply_layer(layer_input: Tensor, layer: Module) -> Tensor:
    """Small aux function to speed up reduce operation.
    :param layer_input: Input to the layer.
    :type layer_input: torch.Tensor
    :param layer: Layer.
    :type layer: torch.nn.Module
    :return: Output of the layer.
    :rtype: torch.Tensor
    """
    return layer(layer_input)


class DepthWiseSeparableConvBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int], MutableSequence[int]],
        stride: Optional[int] = 1,
        padding: Optional[int] = 0,
        dilation: Optional[int] = 1,
        bias: Optional[bool] = True,
        padding_mode: Optional[str] = "zeros",
        inner_kernel_size: Optional[Union[int, Tuple[int, int], MutableSequence[int]]] = 1,
        inner_stride: Optional[int] = 1,
        inner_padding: Optional[int] = 0,
    ) -> None:
        """Depthwise separable 2D Convolution.

        :param in_channels: Input channels.
        :type in_channels: int
        :param out_channels: Output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape/size.
        :type kernel_size: int|tuple|list
        :param stride: Stride.
        :type stride: int|tuple|list
        :param padding: Padding.
        :type padding: int|tuple|list
        :param dilation: Dilation.
        :type dilation: int
        :param bias: Bias.
        :type bias: bool
        :param padding_mode: Padding mode.
        :type padding_mode: str
        :param inner_kernel_size: Kernel shape/size of the second convolution.
        :type inner_kernel_size: int|tuple|list
        :param inner_stride: Inner stride.
        :type inner_stride: int|tuple|list
        :param inner_padding: Inner padding.
        :type inner_padding: int|tuple|list
        """
        super(DepthWiseSeparableConvBlock, self).__init__()

        self.depth_wise_conv: Module = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.non_linearity: Module = LeakyReLU()

        self.batch_norm: Module = BatchNorm2d(out_channels)

        self.point_wise: Module = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=inner_kernel_size,
            stride=inner_stride,
            padding=inner_padding,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.layers: List[Module] = [self.depth_wise_conv, self.non_linearity, self.batch_norm, self.point_wise]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return reduce(apply_layer, self.layers, x)


class DepthWiseSeparableDNN(Module):
    def __init__(
        self,
        cnn_channels: int,
        cnn_dropout: float,
        inner_kernel_size: Union[int, Tuple[int, int]],
        inner_padding: Union[int, Tuple[int, int]],
    ) -> None:
        """Depthwise separable blocks.

        :param cnn_channels: Amount of output CNN channels. For first\
                             CNN in the block is considered equal to 1.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to apply.
        :type cnn_dropout: float
        :param inner_kernel_size: Kernel shape to use.
        :type inner_kernel_size: (int, int)|int
        :param inner_padding: Padding to use.
        :type inner_padding: (int, int)|int
        """
        super().__init__()

        self.layer_1: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=1,
                out_channels=cnn_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding,
            ),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 5), stride=(1, 5)),
            Dropout2d(cnn_dropout),
        )

        self.layer_2: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding,
            ),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            Dropout2d(cnn_dropout),
        )

        self.layer_3: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding,
            ),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            Dropout2d(cnn_dropout),
        )

        self.layers: List[Module] = [self.layer_1, self.layer_2, self.layer_3]

    def forward(self, x):
        """The forward pass of the DepthWiseSeparableDNN.

        :param x: Input audio features.
        :type x: torch.Tensor
        :return: Learned representation\
                 by the DepthWiseSeparableDNN.
        :rtype: torch.Tensor
        """
        return reduce(apply_layer, self.layers, x.unsqueeze(1))


class DilatedConvBLock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
        dilation: Union[int, Tuple[int, int]],
    ) -> None:
        """Dilated convolution block.
        :param in_channels: Amount of input channels.
        :type in_channels: int
        :param out_channels: Amount of output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape.
        :type kernel_size: int|(int, int)
        :param stride: Stride shape.
        :type stride: int|(int, int)
        :param padding: Padding shape.
        :type padding: int|(int, int)
        :param dilation: Dilation shape.
        :type dilation: int|(int, int)
        """
        super().__init__()

        self.cnn = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        )

        self.batch_norm = BatchNorm2d(num_features=out_channels)

        self.non_linearity = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the dilated\
        convolution block.
        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        return self.batch_norm(self.non_linearity(self.cnn(x)))


class DESSEDDilated(Module):
    def __init__(
        self,
        cnn_channels: int,
        cnn_dropout: float,
        dilated_output_channels: int,
        dilated_kernel_size: Union[int, Tuple[int, int], List[int]],
        dilated_stride: Union[int, Tuple[int, int], List[int]],
        dilated_padding: Union[int, Tuple[int, int], List[int]],
        dilation_shape: Union[int, Tuple[int, int], List[int]],
        dilated_nb_features: int,
        nb_classes: int,
        inner_kernel_size: Optional[Union[int, Tuple[int, int], MutableSequence[int]]] = 1,
        inner_padding: Optional[int] = 0,
    ) -> None:
        """The DESSEDDilated model.

        :param cnn_channels: Amount of CNN channels.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to be applied to the CNNs.
        :type cnn_dropout: float
        :param dilated_output_channels: Amount of channels for the\
                                        dilated CNN.
        :type dilated_output_channels: int
        :param dilated_kernel_size: Kernel shape of the dilated CNN.
        :type dilated_kernel_size: int|(int, int)|list[int]
        :param dilated_stride: Stride shape of the dilated CNN.
        :type dilated_stride: int|(int, int)|list[int]
        :param dilated_padding: Padding shape of the dilated CNN.
        :type dilated_padding: int|(int, int)|list[int]
        :param dilation_shape: Dilation shape of the dilated CNN.
        :type dilation_shape: int|(int, int)|list[int]
        :param dilated_nb_features: Amount of features for the batch\
                                    norm after the dilated CNN.
        :type dilated_nb_features: int
        :param nb_classes: Amount of classes to be predicted.
        :type nb_classes: int
        """
        super().__init__()

        self.p_1: List[int] = [0, 3, 2, 1]
        self.p_2: List[int] = [0, 2, 1, 3]

        self.dnn: Module = DepthWiseSeparableDNN(
            cnn_channels=cnn_channels,
            cnn_dropout=cnn_dropout,
            inner_kernel_size=inner_kernel_size,
            inner_padding=inner_padding,
        )

        self.dilated_cnn: Module = DilatedConvBLock(
            in_channels=1,
            out_channels=dilated_output_channels,
            kernel_size=dilated_kernel_size,
            stride=dilated_stride,
            padding=dilated_padding,
            dilation=dilation_shape,
        )

        self.classifier: Module = Linear(
            in_features=dilated_nb_features * dilated_output_channels, out_features=nb_classes, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the DESSEDDilated model.

        :param x: Input to the DESSEDDilated.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        r_s: List[int] = list(x.size()[:-1]) + [-1]

        out: Tensor = self.dnn(x).permute(*self.p_1).contiguous()

        out: Tensor = self.dilated_cnn(out).permute(*self.p_2).reshape(*r_s)

        return self.classifier(out)


class DESSEDDilatedTag(Module):
    def __init__(
        self,
        cnn_channels: int,
        cnn_dropout: float,
        dilated_output_channels: int,
        dilated_kernel_size: Union[int, Tuple[int, int], List[int]],
        dilated_stride: Union[int, Tuple[int, int], List[int]],
        dilated_padding: Union[int, Tuple[int, int], List[int]],
        dilation_shape: Union[int, Tuple[int, int], List[int]],
        dilated_nb_features: int,
        nb_classes: int,
        inner_kernel_size: Optional[Union[int, Tuple[int, int], MutableSequence[int]]] = 1,
        inner_padding: Optional[int] = 0,
    ) -> None:
        """The DESSEDDilated model.

        :param cnn_channels: Amount of CNN channels.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to be applied to the CNNs.
        :type cnn_dropout: float
        :param dilated_output_channels: Amount of channels for the\
                                        dilated CNN.
        :type dilated_output_channels: int
        :param dilated_kernel_size: Kernel shape of the dilated CNN.
        :type dilated_kernel_size: int|(int, int)|list[int]
        :param dilated_stride: Stride shape of the dilated CNN.
        :type dilated_stride: int|(int, int)|list[int]
        :param dilated_padding: Padding shape of the dilated CNN.
        :type dilated_padding: int|(int, int)|list[int]
        :param dilation_shape: Dilation shape of the dilated CNN.
        :type dilation_shape: int|(int, int)|list[int]
        :param dilated_nb_features: Amount of features for the batch\
                                    norm after the dilated CNN.
        :type dilated_nb_features: int
        :param nb_classes: Amount of classes to be predicted.
        :type nb_classes: int
        """
        super().__init__()

        self.p_1: List[int] = [0, 3, 2, 1]
        self.p_2: List[int] = [0, 2, 1, 3]

        self.dnn: Module = DepthWiseSeparableDNN(
            cnn_channels=cnn_channels,
            cnn_dropout=cnn_dropout,
            inner_kernel_size=inner_kernel_size,
            inner_padding=inner_padding,
        )

        self.dilated_cnn: Module = DilatedConvBLock(
            in_channels=1,
            out_channels=dilated_output_channels,
            kernel_size=dilated_kernel_size,
            stride=dilated_stride,
            padding=dilated_padding,
            dilation=dilation_shape,
        )

        self.classifier: Module = Linear(
            in_features=dilated_nb_features * dilated_output_channels, out_features=nb_classes, bias=True
        )

        # self.lin = Linear(10, 10)

        pooling_param = {
            "in_features": nb_classes,
            "out_features": nb_classes,
            "pooling": "auto",
        }
        self.pooling_head = Pooling_Head(**pooling_param)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the DESSEDDilated model.

        :param x: Input to the DESSEDDilated.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        r_s: List[int] = list(x.size()[:-1]) + [-1]

        out: Tensor = self.dnn(x).permute(*self.p_1).contiguous()

        out: Tensor = self.dilated_cnn(out).permute(*self.p_2).reshape(*r_s)

        out: Tensor = self.classifier(out)

        frame_prob: Tensor = sigmoid(out)
        # return clamp(self.pooling_head(frame_prob,out)[0], 0, 1)
        return clamp(self.pooling_head(frame_prob, out)[0], 0, 1)


# EOF
