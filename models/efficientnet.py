from timm.models import register_model

import copy
import math
from functools import partial
from typing import Any, Callable, Optional, List, Sequence

import torch
from torch import nn, Tensor
from models.stochastic_depth import StochasticDepth


def load_state_dict_from_url(*args, **kwargs):
    return None


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from models.misc import ConvNormActivation, SqueezeExcitation

model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
        allways_skip: bool = False,
        downsample_kernel_size: Optional[int] = None
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        self.allways_skip = allways_skip
        self.downsample_kernel_size = downsample_kernel_size

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f", allways_skip={self.allways_skip}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,

        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.allways_skip or (cnf.stride == 1 and cnf.input_channels == cnf.out_channels)
        self.conv_skip = None
        if cnf.allways_skip and (cnf.stride != 1 or cnf.input_channels != cnf.out_channels):
            # add a 1x1 convolution with stride 1 which is used in allways_skip mode
            self.conv_skip = nn.Conv2d(
                kernel_size=1, stride=cnf.stride,
                in_channels=cnf.input_channels,
                out_channels=cnf.out_channels
            )

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.conv_skip is not None:
            input = self.conv_skip(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()
        #_log_api_usage_once(self)
        self.default_cfg = {}

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                if not stage and cnf.downsample_kernel_size is not None:
                    block_cnf.kernel = cnf.downsample_kernel_size

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.SiLU,
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet(
    arch: str,
    width_mult: float,
    depth_mult: float,
    dropout: float,
    pretrained: bool,
    progress: bool,
    inverted_residual_setting: Optional[List[MBConvConfig]] = None,
    allways_skip: bool = False,
    **kwargs: Any,
) -> EfficientNet:
    if "layer_scale_init_value" in kwargs:
        kwargs.pop("layer_scale_init_value")
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult, allways_skip=allways_skip)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ] if inverted_residual_setting is None else inverted_residual_setting
    # this one has no unproductive layers and is thus likely more efficient
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


@register_model
def _efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, pretrained, progress=progress, **kwargs)


@register_model
def _efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b1", 1.0, 1.1, 0.2, pretrained, progress=progress, **kwargs)

@register_model
def _efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b2", 1.1, 1.2, 0.3, pretrained, progress, **kwargs)


def _efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b3", 1.2, 1.4, 0.3, pretrained, progress, **kwargs)


def _efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet("efficientnet_b4", 1.4, 1.8, 0.4, pretrained, progress, **kwargs)


def _efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b5",
        1.6,
        2.2,
        0.4,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def _efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b6",
        1.8,
        2.6,
        0.5,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


def _efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _efficientnet(
        "efficientnet_b7",
        2.0,
        3.1,
        0.5,
        pretrained,
        progress,
        norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        **kwargs,
    )


@register_model
def efficientnet_b0(*args, **kwargs):
    model = _efficientnet_b0(**kwargs)
    model.name = "EfficentNetB0"
    model.num_classes = ...
    return model


@register_model
def efficientnet_b1(*args, **kwargs):
    model = _efficientnet_b1(**kwargs)
    model.name = "EfficentNetB1"
    model.num_classes = ...
    return model


@register_model
def efficientnet_b2(*args, **kwargs):
    model = _efficientnet_b2(**kwargs)
    model.name = "EfficentNetB2"
    model.num_classes = ...
    return model


@register_model
def efficientnet_b3(*args, **kwargs):
    model = _efficientnet_b3(**kwargs)
    model.name = "EfficentNetB3"
    model.num_classes = ...
    return model


@register_model
def efficientnet_b4(*args, **kwargs):
    model = _efficientnet_b4(**kwargs)
    model.name = "EfficentNetB4"
    model.num_classes = ...
    return model


@register_model
def efficientnet_b5(*args, **kwargs):
    model = _efficientnet_b5(**kwargs)
    model.name = "EfficentNetB5"
    model.num_classes = ...
    return model

@register_model
def efficientnet_b6(*args, **kwargs):
    model = _efficientnet_b6(**kwargs)
    model.name = "EfficentNetB6"
    model.num_classes = ...
    return model

@register_model
def efficientnet_b7(*args, **kwargs):
    model = _efficientnet_b7(**kwargs)
    model.name = "EfficentNetB7"
    model.num_classes = ...
    return model



@register_model
def efficientnet_b0_eff(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Efficiency"
    return model


@register_model
def efficientnet_b0_perf(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 7),
        bneck_conf(6, 5, 2, 112, 192, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance"
    return model


@register_model
def efficientnet_b0_perf2(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 3),
        bneck_conf(6, 5, 2, 24, 40, 3),
        bneck_conf(6, 3, 2, 40, 80, 4),
        bneck_conf(6, 5, 1, 80, 112, 4),
        bneck_conf(6, 5, 2, 112, 192, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance2"
    return model

@register_model
def efficientnet_b0_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance3"
    return model


@register_model
def efficientnet_b1_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.1)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.0, depth_mult=1.1, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB1_Performance3"
    return model


@register_model
def efficientnet_b2_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.1, depth_mult=1.2)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.1, depth_mult=1.2, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB2_Performance3"
    return model


@register_model
def efficientnet_b3_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.2, depth_mult=1.4)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.2, depth_mult=1.4, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB3_Performance3"
    return model


@register_model
def efficientnet_b4_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.4, depth_mult=1.8)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b4", width_mult=1.4, depth_mult=1.8, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB4_Performance3"
    return model


@register_model
def efficientnet_b5_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.6, depth_mult=2.2)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b5", width_mult=1.6, depth_mult=2.2, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                          **kwargs)
    model.name = "EfficentNetB5_Performance3"
    return model


@register_model
def efficientnet_b6_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.8, depth_mult=2.6)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b6", width_mult=1.8, depth_mult=2.6, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),

                          **kwargs)
    model.name = "EfficentNetB6_Performance3"
    return model


@register_model
def efficientnet_b7_perf3(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=2.0, depth_mult=3.1)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 3, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b7", width_mult=2.0, depth_mult=3.1, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                          **kwargs)
    model.name = "EfficentNetB7_Performance3"
    return model



@register_model
def efficientnet_b0_perf32(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 3, 2, 24, 40, 2),
        bneck_conf(6, 5, 2, 40, 80, 3, downsample_kernel_size=3),
        bneck_conf(6, 3, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4, downsample_kernel_size=3),
        bneck_conf(6, 5, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance3"
    return model


@register_model
def efficientnet_b0_perf33(*args, **kwargs):
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 7, 1, 32, 16, 1),
        bneck_conf(6, 7, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 117, 3),
        bneck_conf(6, 3, 2, 117, 195, 4),
        bneck_conf(6, 3, 1, 195, 320, 1),
    ]
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance33"
    return model


@register_model
def efficientnet_b0_perf4(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 240, 7),
        bneck_conf(6, 5, 2, 240, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance4"
    return model


@register_model
def efficientnet_b1_perf4(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 240, 7),
        bneck_conf(6, 5, 2, 240, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.0, depth_mult=1.1, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB1_Performance4"
    return model


@register_model
def efficientnet_b0_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance5"
    return model


@register_model
def efficientnet_b1_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.0, depth_mult=1.1, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB1_Performance5"
    return model


@register_model
def efficientnet_b2_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.1, depth_mult=1.2)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.1, depth_mult=1.2, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB2_Performance5"
    return model


@register_model
def efficientnet_b3_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.2, depth_mult=1.4)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b3", width_mult=1.2, depth_mult=1.4, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB3_Performance5"
    return model


@register_model
def efficientnet_b4_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.4, depth_mult=1.8)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b4", width_mult=1.4, depth_mult=1.8, dropout=0.4, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB4_Performance5"
    return model


@register_model
def efficientnet_b5_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.6, depth_mult=2.2)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b5", width_mult=1.6, depth_mult=2.2, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB5_Performance5"
    return model


@register_model
def efficientnet_b5_perf52(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.6, depth_mult=2.2)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b5", width_mult=1.6, depth_mult=2.2, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB5_Performance52"
    return model


@register_model
def efficientnet_b6_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.8, depth_mult=2.6)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b5", width_mult=1.8, depth_mult=2.6, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB5_Performance5"
    return model


@register_model
def efficientnet_b7_perf5(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=2.0, depth_mult=3.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 1, 40, 80, 3),
        bneck_conf(6, 5, 2, 80, 112, 3),
        bneck_conf(6, 5, 1, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b7", width_mult=2.0, depth_mult=3.1, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,

                          **kwargs)
    model.name = "EfficentNetB7_Performance5"
    return model


@register_model
def efficientnet_b0_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance42"
    return model

@register_model
def efficientnet_b0_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b0", width_mult=1.0, depth_mult=1.0, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB0_Performance43"
    return model


@register_model
def efficientnet_b1_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.0, depth_mult=1.1, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB1_Performance42"
    return model

@register_model
def efficientnet_b1_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b1", width_mult=1.0, depth_mult=1.1, dropout=0.2, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB1_Performance43"
    return model


@register_model
def efficientnet_b2_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.1, depth_mult=1.2)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b2", width_mult=1.1, depth_mult=1.2, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB2_Performance42"
    return model

@register_model
def efficientnet_b2_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.1, depth_mult=1.2)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b2", width_mult=1.1, depth_mult=1.2, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB2_Performance42"
    return model

@register_model
def efficientnet_b3_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.2, depth_mult=1.4)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b3", width_mult=1.2, depth_mult=1.4, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB2_Performance42"
    return model


@register_model
def efficientnet_b3_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.2, depth_mult=1.4)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b3", width_mult=1.2, depth_mult=1.4, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB3_Performance43"
    return model


@register_model
def efficientnet_b4_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.4, depth_mult=1.8)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b4", width_mult=1.4, depth_mult=1.8, dropout=0.4, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB4_Performance42"
    return model


@register_model
def efficientnet_b4_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.4, depth_mult=1.8)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b4", width_mult=1.4, depth_mult=1.8, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB4_Performance43"
    return model


@register_model
def efficientnet_b5_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.6, depth_mult=2.2)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b5", width_mult=1.6, depth_mult=2.2, dropout=0.4, pretrained=False,
                          norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB5_Performance42"
    return model


@register_model
def efficientnet_b5_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.6, depth_mult=2.2)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b5", width_mult=1.6, depth_mult=2.2, dropout=0.4, pretrained=False,
                          norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB5_Performance43"
    return model


@register_model
def efficientnet_b6_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.8, depth_mult=2.6)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b3", width_mult=1.8, depth_mult=2.6, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB2_Performance42"
    return model


@register_model
def efficientnet_b6_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.8, depth_mult=2.6)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b6", width_mult=1.8, depth_mult=2.6, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB6_Performance43"
    return model


@register_model
def efficientnet_b7(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=2.0, depth_mult=3.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b7", width_mult=2.0, depth_mult=3.1, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB7"
    return model


@register_model
def efficientnet_b7_perf42(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=2.0, depth_mult=3.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 2, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b7", width_mult=2.0, depth_mult=3.1, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB7_Performance42"
    return model


@register_model
def efficientnet_b7_perf43(*args, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=2.0, depth_mult=3.1)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 165, 7),
        bneck_conf(6, 5, 1, 165, 320, 1),
        #bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model = _efficientnet("efficientnet_b7", width_mult=2.0, depth_mult=3.1, dropout=0.3, pretrained=False,
                          progress=True, inverted_residual_setting=inverted_residual_setting,
                          **kwargs)
    model.name = "EfficentNetB7_Performance43"
    return model


@register_model
def efficientnet_b0_perf6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if "pretrained" in kwargs:
        kwargs.pop("pretrained")
    model =  _efficientnet("efficientnet_b0", 1.0, 1.0, 0.2, pretrained, progress, allways_skip=True, **kwargs)
    model.name = "EfficientNetB0_Performance6"
    return model


if __name__ == '__main__':
    from rfa_toolbox import create_graph_from_pytorch_model, visualize_architecture, input_resolution_range
    from torchvision.models import resnet18
    models = [
        efficientnet_b0_perf43,
        efficientnet_b1_perf43,
        efficientnet_b2_perf43,
        efficientnet_b3_perf43,
        efficientnet_b4_perf43,
        efficientnet_b5_perf43,
        efficientnet_b6_perf43,
        efficientnet_b7_perf43,
        #efficientnet_b0_perf5,
        #efficientnet_b1_perf4,
        #efficientnet_b1_perf5,
        #efficientnet_b2_perf42,
        #efficientnet_b2_perf5,
        #efficientnet_b3_perf42,
        #efficientnet_b3_perf5,
        #efficientnet_b4_perf42,
        #efficientnet_b4_perf5,
        #efficientnet_b5_perf42,
        #efficientnet_b5_perf5,
        #efficientnet_b6_perf42,
        #efficientnet_b6_perf5,
        #efficientnet_b7_perf42,
        #efficientnet_b7_perf5,
    ]
    for model  in models:
        model = model().cpu()
        name = model.name
        graph = create_graph_from_pytorch_model(model, input_res=(1, 3, 224, 224),
                                                custom_layers=["SqueezeExcitation", "ConvNormActivation"])
        imin, imax = input_resolution_range(graph)
        print(name, ":", *imin)
        visualize_architecture(graph, "EfficientNet", input_res=224).view()
