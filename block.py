# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
Block modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = ('DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3', 'C2f', 'C3x', 'C3TR', 'C3Ghost',
           'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'RepC3', 'BasicStage', 'PatchEmbed_FasterNet',
           'PatchMerging_FasterNet','VanillaBlock','PatchMerging', 'PatchEmbed', 'SwinStage','GhostV2','DepthSepConv','SGBlock',
           'CBRM', 'Shuffle_Block','stem', 'MBConvBlock','Conv_BN_HSwish', 'MobileNetV3_InvertedResidual',
           'ConvNeXt_Stem', 'ConvNeXt_Block', 'ConvNeXt_Downsample','MV2Block', 'MobileViTBlock', 'ResNetLayer','StarBlock','Silence',
           'MobileNetV3_BLOCK', 'InvertedBottleneck', 'mn_conv','SE','SimAM', 'ECA', 'SpatialGroupEnhance', 'TripletAttention', 'CoordAtt', 'GAMAttention',
           'SE', 'ShuffleAttention', 'SKAttention', 'DoubleAttention', 'CoTAttention', 'EffectiveSEModule',
           'GlobalContext', 'GatherExcite', 'MHSA', 'S2Attention', 'NAMAttention', 'CrissCrossAttention',
           'SequentialPolarizedSelfAttention', 'ParallelPolarizedSelfAttention','C2f_CA', 'C2f_SE', 'C2f_ECA', 'C2f_CBAM',
           'ParNetAttention','C2f_EffectiveSE', 'C2f_GlobalContext', 'C2f_GatherExcite', 'C2f_MHSA','C2f_SimAM', 'C2f_CoT', 'C2f_Double', 'C2f_SK',
           'BiLevelRoutingAttention','C2f_Triplet', 'C2f_SpatialGroupEnhance', 'C2f_S2', 'C2f_NAM','C2f_ParNet', 'C2f_GAM', 'C2f_CrissCross', 'C2f_ParallelPolarized',
           'C2f_SequentialPolarized','BiFPN_Add2', 'BiFPN_Add3','DRAMiTransformer','G_bneck','MEEM','CondConv','MogaBlock','SCSA','AxialImageTransformer','WTConv2d',
           'PPA','SAFM'



























          )




class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """HG_Block of PPHGNetV2 with 2 convolutions and LightConv.
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):  # ch_in, ch_out, number
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))
"""
https://arxiv.org/abs/2303.03667
<<Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks>>
"""
# --------------------------FasterNet----------------------------
from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):
    def __init__(self,
                 dim,
                 depth=1,
                 n_div=4,
                 mlp_ratio=2,
                 layer_scale_init_value=0,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        dpr = [x.item()
               for x in torch.linspace(0, 0.0, sum([1, 2, 8, 2]))]
        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.blocks(x)
        return x


class PatchEmbed_FasterNet(nn.Module):

    def __init__(self, in_chans, embed_dim, patch_size, patch_stride, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.proj(x))
        return x

    def fuseforward(self, x):
        x = self.proj(x)
        return x


class PatchMerging_FasterNet(nn.Module):

    def __init__(self, dim, out_dim, k, patch_stride2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.reduction = nn.Conv2d(dim, out_dim, kernel_size=k, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(out_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.reduction(x))
        return x

    def fuseforward(self, x):
        x = self.reduction(x)
        return x

# -------------Vanillanet----------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import weight_init, DropPath
from timm.models.registry import register_model


# Series informed activation function. Implemented by conv.
class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.act_num = act_num
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num * 2 + 1, act_num * 2 + 1))
        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, self.bias, padding=self.act_num, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class VanillaBlock(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=2, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num, deploy=self.deploy)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)

            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x, self.act_learn)

            x = self.conv2(x)

        x = self.pool(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1, 3),
                                             self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1, 3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True

# -------------Swin Transformer----------------------
""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        #x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = (attn.to(v.dtype) @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把 feature map 给 pad 到 window size 的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SwinStage(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, c2, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        assert dim == c2, r"no. in/out channel should be same"
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """

    def __init__(self, in_c=3, embed_dim=96, patch_size=4, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        B, C, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        # view: [B, HW, C] -> [B, H, W, C]
        # permute: [B, H, W, C] -> [B, C, H, W]
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, c2, norm_layer=nn.LayerNorm):
        super().__init__()
        assert c2 == (2 * dim), r"no. out channel should be 2 * no. in channel "
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        x = x.permute(0, 2, 3, 1).contiguous()
        # x = x.view(B, H*W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]
        x = x.view(B, int(H / 2), int(W / 2), C * 2)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x



        #pp-LCNet


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module('FC1', nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1))  # 1*1卷积与此效果相同
        self.Excitation.add_module('ReLU', nn.ReLU())
        self.Excitation.add_module('FC2', nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1))
        self.Excitation.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


class DepthSepConv(nn.Module):
    def __init__(self, inp, oup, dw_size, stride, use_se):
        super(DepthSepConv, self).__init__()
        self.stride = stride
        self.inp = inp
        self.oup = oup
        self.dw_size = dw_size
        self.dw_sp = nn.Sequential(
            nn.Conv2d(self.inp, self.inp, kernel_size=self.dw_size, stride=self.stride, padding=(dw_size - 1) // 2,
                      groups=self.inp, bias=False),
            nn.BatchNorm2d(self.inp),
            nn.Hardswish(),

            SeBlock(self.inp, reduction=16) if use_se else nn.Sequential(),

            nn.Conv2d(self.inp, self.oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.oup),
            nn.Hardswish())

    def forward(self, x):
        y = self.dw_sp(x)
        return y


# ---------------------------MobileNext Begin---------------------------

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False, initialize_weights=True):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        # self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.initialize_weights = initialize_weights
        self.expand_ratio = expand_ratio

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif self.initialize_weights:
            self._initialize_weights()
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            return out + x
        else:
            return out

# ---------------------------MobileNext End---------------------------



# ---------------------------Shufflenetv2---------------------------
import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class CBRM(nn.Module):  # Conv BN ReLU Maxpool2d
    def __init__(self, c1, c2):
        super(CBRM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

    def forward(self, x):
        return self.maxpool(self.conv(x))


class Shuffle_Block(nn.Module):
    def __init__(self, ch_in, ch_out, stride):
        super(Shuffle_Block, self).__init__()

        if not (1 <= stride <= 2):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = ch_out // 2
        assert (self.stride != 1) or (ch_in == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(ch_in, ch_in, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(ch_in),

                nn.Conv2d(ch_in, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(ch_in if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

# EfficientNetLite
class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module('FC1', nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1))  # 1*1卷积与此效果相同
        self.Excitation.add_module('ReLU', nn.ReLU())
        self.Excitation.add_module('FC2', nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1))
        self.Excitation.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


class drop_connect:
    def __init__(self, drop_connect_rate):
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x, training):
        if not training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)  # 1
        x = (x / keep_prob) * binary_mask
        return x


class stem(nn.Module):
    def __init__(self, c1, c2, act='ReLU6'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(num_features=c2)
        if act == 'ReLU6':
            self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, drop_connect_rate, has_se=False):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect
        se_ratio = 0.25

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self.se = SeBlock(oup, 4)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU6(inplace=True)

        self.drop_connect = drop_connect(drop_connect_rate)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x = self.se(x)

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = self.drop_connect(x, training=self.training)
            x += identity  # skip connection
        return x




#................ Mobilenetv3Small
class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module('FC1', nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1))  # 1*1卷积与此效果相同
        self.Excitation.add_module('ReLU', nn.ReLU())
        self.Excitation.add_module('FC2', nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1))
        self.Excitation.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))


class Conv_BN_HSwish(nn.Module):
    """
    This equals to
    def conv_3x3_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            h_swish()
        )
    """

    def __init__(self, c1, c2, stride):
        super(Conv_BN_HSwish, self).__init__()
        self.conv = nn.Conv2d(c1, c2, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MobileNetV3_InvertedResidual(nn.Module):
    def __init__(self, inp, oup, hidden_dim, kernel_size, stride, use_se, use_hs):
        super(MobileNetV3_InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(),
                # Squeeze-and-Excite
                SeBlock(hidden_dim) if use_se else nn.Sequential(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.Hardswish() if use_hs else nn.ReLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SeBlock(hidden_dim) if use_se else nn.Sequential(),
                nn.Hardswish() if use_hs else nn.ReLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y




#-------------------------------------------ConvNeXt-------------------------------------------------
import math

from torch import nn
import torch
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F


class ConvNeXt_Stem(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, groups=g, dilation=d)
        self.ln = LayerNorm(c2, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return self.ln(self.conv(x))


class ConvNeXt_Downsample(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, groups=g, dilation=d)
        self.ln = LayerNorm(c1, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        return self.conv(self.ln(x))


class ConvNeXt_Inside_Block(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, drop_path=0.):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv1(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt_Block(nn.Module):
    def __init__(self, c1, c2, n=1, layer_scale_init_value=1e-6):
        super().__init__()
        self.m = nn.Sequential(*(ConvNeXt_Inside_Block(c2, layer_scale_init_value) for _ in range(n)))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.m(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


from einops import rearrange


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    # CSDN 迪菲赫尔曼
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    # CSDN 迪菲赫尔曼
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph = patch_size
        self.pw = patch_size
        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)
        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        x = self.conv1(x)
        x = self.conv2(x)
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x



class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(Conv(c1, c2, k=7, s=2, p=3, act=True),
                                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   StarNet.py
@Time      :   2024/06/12 19:43:33
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
import torch.nn as nn
from timm.models.layers import DropPath


# 定义 ConvBN 模块
class ConvBN(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        with_bn=True,
    ):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=not with_bn,
        )
        self.with_bn = with_bn
        if with_bn:
            self.bn = nn.BatchNorm2d(out_planes)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        return x


# 定义 Block 模块
class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, n=1):
        super(StarBlock, self).__init__()
        self.n = n
        drop_path = 0.0
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, kernel_size=1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, kernel_size=1)
        self.dwconv2 = ConvBN(
            dim, dim, kernel_size=7, padding=3, groups=dim, with_bn=False
        )
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        for _ in range(self.n):  # 使用循环来重复特定的操作n次
            x = self.dwconv(x)
            x1, x2 = self.f1(x), self.f2(x)
            x = self.act(x1) * x2
            x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)  # 确保残差连接仅应用一次，而不是n次
        return x


# 定义 StarNet 模块
class StarNet(nn.Module):
    def __init__(
        self,
        base_dim=32,
        depths=[3, 3, 12, 5],
        mlp_ratio=4,
        drop_path_rate=0.0,
        num_classes=1000,
    ):
        super(StarNet, self).__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6()
        )

        # 构建阶段
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # 第一阶段
        embed_dim = base_dim * 2**0
        down_sampler1 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks1 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[0])
        ]
        cur += depths[0]
        stage1 = nn.Sequential(down_sampler1, *blocks1)

        # 第二阶段
        embed_dim = base_dim * 2**1
        down_sampler2 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks2 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[1])
        ]
        cur += depths[1]
        stage2 = nn.Sequential(down_sampler2, *blocks2)

        # 第三阶段
        embed_dim = base_dim * 2**2
        down_sampler3 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks3 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[2])
        ]
        cur += depths[2]
        stage3 = nn.Sequential(down_sampler3, *blocks3)

        # 第四阶段
        embed_dim = base_dim * 2**3
        down_sampler4 = ConvBN(
            self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.in_channel = embed_dim
        blocks4 = [
            StarBlock(self.in_channel, mlp_ratio, dpr[cur + i])
            for i in range(depths[3])
        ]
        cur += depths[3]
        stage4 = nn.Sequential(down_sampler4, *blocks4)

        self.stages = nn.ModuleList([stage1, stage2, stage3, stage4])
        # 分类头
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(self.norm(x))
        x = torch.flatten(x, 1)
        return self.head(x)


# 使用示例
if __name__ == "__main__":
    model = StarNet(
        base_dim=32,
        depths=[3, 3, 12, 5],
        mlp_ratio=4,
        drop_path_rate=0.1,
        num_classes=1000,
    )
    input_tensor = torch.randn(1, 3, 224, 224)  # 创建一个随机输入张量
    output = model(input_tensor)  # 前向传播
    print(output.shape)  # 输出分类结果

# 在 ultralytics/nn/modules.py 或其他自定义模块文件中添加
import torch.nn as nn


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x




import math
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(x, divisor=4):
    return int(np.ceil(x * 1. / divisor) * divisor)


class MyHSigmoid(nn.Module):
    """
    Hard Sigmoid definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> MyHSigmoid()
    """

    def __init__(self):
        super(MyHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        """ forward """
        return self.relu6(x + 3.) * 0.16666667


class Activation(nn.Module):
    """
    Activation definition.

    Args:
        act_func(string): activation name.

    Returns:
         Tensor, output tensor.
    """

    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = MyHSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.Hardswish()
        else:
            raise NotImplementedError

    def forward(self, x):
        """ forward """
        return self.act(x)


class GlobalAvgPooling(nn.Module):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """ forward """
        x = self.mean(x)
        return x


class SE_Ghost(nn.Module):
    """
    SE warpper definition.

    Args:
        num_out (int): Output channel.
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.

    """

    def __init__(self, num_out, ratio=4):
        super(SE_Ghost, self).__init__()
        num_mid = _make_divisible(num_out // ratio)
        self.pool = GlobalAvgPooling()
        self.conv_reduce = nn.Conv2d(in_channels=num_out, out_channels=num_mid,
                                     kernel_size=1, bias=True, padding_mode='zeros')
        self.act1 = Activation('relu')
        self.conv_expand = nn.Conv2d(in_channels=num_mid, out_channels=num_out,
                                     kernel_size=1, bias=True, padding_mode='zeros')
        self.act2 = Activation('hsigmoid')

    def forward(self, x):
        """ forward of SE module """
        out = self.pool(x)
        out = self.conv_reduce(out)
        out = self.act1(out)
        out = self.conv_expand(out)
        out = self.act2(out)
        out = x * out
        return out


class ConvUnit(nn.Module):
    """
    ConvUnit warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (Union[int, tuple[int]]): Input kernel size.
        stride (int): Stride size.
        padding (Union[int, tuple[int]]): Padding number.
        num_groups (int): Output num group.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, num_groups=1,
                 use_act=True, act_type='relu'):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_in,
                              out_channels=num_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=num_groups,
                              bias=False,
                              padding_mode='zeros')
        self.bn = nn.BatchNorm2d(num_out)
        self.use_act = use_act
        self.act = Activation(act_type) if use_act else None

    def forward(self, x):
        """ forward of conv unit """
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out


class GhostModule(nn.Module):
    """
    GhostModule warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        ratio (int): Reduction ratio.
        dw_size (int): kernel size of cheap operation.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size // 2, num_groups=1, use_act=use_act, act_type=act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                        padding=dw_size // 2, num_groups=init_channels,
                                        use_act=use_act, act_type=act_type)

    def forward(self, x):
        """ ghost module forward """
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostModuleMul(nn.Module):
    """
    GhostModuleMul warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        ratio (int): Reduction ratio.
        dw_size (int): kernel size of cheap operation.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModuleMul, self).__init__()
        self.avgpool2d = nn.AvgPool2d(kernel_size=1, stride=1)
        self.gate_fn = Activation('sigmoid')
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvUnit(num_in, init_channels, kernel_size=kernel_size, stride=stride,
                                     padding=kernel_size // 2, num_groups=1, use_act=use_act, act_type=act_type)
        self.cheap_operation = ConvUnit(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                        padding=dw_size // 2, num_groups=init_channels,
                                        use_act=use_act, act_type=act_type)
        self.short_conv = nn.Sequential(
            ConvUnit(num_in, num_out, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, num_groups=1, use_act=False),
            ConvUnit(num_out, num_out, kernel_size=(1, 5), stride=1,
                     padding=(0, 2), num_groups=num_out, use_act=False),
            ConvUnit(num_out, num_out, kernel_size=(5, 1), stride=1,
                     padding=(2, 0), num_groups=num_out, use_act=False),
        )

    def forward(self, x):
        """ ghost module forward """
        res = self.avgpool2d(x)
        res = self.short_conv(res)
        res = self.gate_fn(res)

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        out = out * F.interpolate(res, size=out.shape[-2:], mode="bilinear", align_corners=True)
        return out


class GhostV2(nn.Module):
    """
    GhostBottleneck warpper definition.

    Args:
        num_in (int): Input channel.
        num_mid (int): Middle channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        act_type (str): Activation type.
        use_se (bool): Use SE warpper or not.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, num_in, num_out, num_mid, kernel_size, stride=1, act_type='relu', use_se=False, layer_id=None):
        super(GhostV2, self).__init__()
        self.use_ori_module = layer_id <= 1
        if self.use_ori_module:
            self.ghost1 = GhostModule(num_in, num_mid, kernel_size=1,
                                      stride=1, padding=0, act_type=act_type)
        else:
            self.ghost1 = GhostModuleMul(num_in, num_mid, kernel_size=1,
                                         stride=1, padding=0, act_type=act_type)

        self.use_dw = stride > 1
        self.dw = None
        if self.use_dw:
            self.dw = ConvUnit(num_mid, num_mid, kernel_size=kernel_size, stride=stride,
                               padding=self._get_pad(kernel_size), act_type=act_type, num_groups=num_mid, use_act=False)

        self.use_se = use_se
        if use_se:
            self.se = SE_Ghost(num_mid)

        self.ghost2 = GhostModule(num_mid, num_out, kernel_size=1, stride=1,
                                  padding=0, act_type=act_type, use_act=False)

        self.down_sample = False
        if num_in != num_out or stride != 1:
            self.down_sample = True
        self.shortcut = None
        if self.down_sample:
            self.shortcut = nn.Sequential(
                ConvUnit(num_in, num_in, kernel_size=kernel_size, stride=stride,
                         padding=self._get_pad(kernel_size), num_groups=num_in, use_act=False),
                ConvUnit(num_in, num_out, kernel_size=1, stride=1,
                         padding=0, num_groups=1, use_act=False),
            )

    def forward(self, x):
        """ forward of ghostnet """
        shortcut = x
        out = self.ghost1(x)
        if self.use_dw:
            out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        if self.down_sample:
            shortcut = self.shortcut(shortcut)
        out = shortcut + out
        return out

    def _get_pad(self, kernel_size):
        """set the padding number"""
        pad = 0
        if kernel_size == 1:
            pad = 0
        elif kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        elif kernel_size == 7:
            pad = 3
        else:
            raise NotImplementedError
        return pad


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   LeYOLO.py
@Time      :   2024/06/21 15:32:34
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
import torch.nn as nn

__all__ = ("mn_conv", "MobileNetV3_BLOCK")


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def activation_function(act="RE"):
    res = nn.Hardswish()
    if act == "RE":
        res = nn.ReLU6(inplace=True)
    elif act == "GE":
        res = nn.GELU()
    elif act == "SI":
        res = nn.SiLU()
    elif act == "EL":
        res = nn.ELU()
    else:
        res = nn.Hardswish()
    return res


class mn_conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, act="RE", p=None, g=1, d=1):
        super().__init__()
        padding = 0 if k == s else autopad(k, p, d)
        self.c = nn.Conv2d(c1, c2, k, s, padding, groups=g)
        self.bn = nn.BatchNorm2d(c2)
        self.act = activation_function(
            act
        )  # nn.ReLU6(inplace=True) if act=="RE" else nn.Hardswish()

    def forward(self, x):
        return self.act(self.bn(self.c(x)))


class InvertedBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        # input_channels, output_channels, repetition, stride, expension ratio
        super().__init__()
        # act = nn.ReLU6(inplace=True) if NL=="RE" else nn.Hardswish()
        c_mid = e if e != None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else []  # if c_mid != c1 else []
        features.extend(
            [
                mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                # attn,
                nn.Conv2d(c_mid, c2, 1),
                nn.BatchNorm2d(c2),
                # nn.SiLU(),
            ]
        )
        self.layers = nn.Sequential(*features)

    def forward(self, x):
        # print(x.shape)
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV3_BLOCK(nn.Module):
    def __init__(self, c1, c2, k=3, e=None, sa="None", act="RE", stride=1, pw=True):
        # input_channels, output_channels, repetition, stride, expension ratio
        super().__init__()
        # act = nn.ReLU6(inplace=True) if NL=="RE" else nn.Hardswish()
        c_mid = e if e != None else c1
        self.residual = c1 == c2 and stride == 1

        features = [mn_conv(c1, c_mid, act=act)] if pw else []  # if c_mid != c1 else []
        features.extend(
            [
                mn_conv(c_mid, c_mid, k, stride, g=c_mid, act=act),
                # attn,
                nn.Conv2d(c_mid, c2, 1),
                nn.BatchNorm2d(c2),
                # nn.SiLU(),
            ]
        )
        self.layers = nn.Sequential(*features)

    def forward(self, x):
        # print(x.shape)
        if self.residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


# SE
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.nn import Softmax, init


# ---------------------------SE Begin---------------------------
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        # c*1*1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------------------SE End---------------------------


# ---------------------------ECA Begin---------------------------
class ECA(nn.Module):

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# ---------------------------ECA End---------------------------


# ---------------------------CA Begin---------------------------
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


# ---------------------------CA End---------------------------


# ---------------------------CBAM Begin---------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


# ---------------------------CBAM End---------------------------
# ---------------------------SimAM Begin---------------------------
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)


# ---------------------------SimAM End---------------------------


# ---------------------------S2-MLPv2 Begin---------------------------
def spatial_shift1(x):
    b, w, h, c = x.size()
    x[:, 1:, :, : c // 4] = x[:, : w - 1, :, : c // 4]
    x[:, : w - 1, :, c // 4 : c // 2] = x[:, 1:, :, c // 4 : c // 2]
    x[:, :, 1:, c // 2 : c * 3 // 4] = x[:, :, : h - 1, c // 2 : c * 3 // 4]
    x[:, :, : h - 1, 3 * c // 4 :] = x[:, :, 1:, 3 * c // 4 :]
    return x


def spatial_shift2(x):
    b, w, h, c = x.size()
    x[:, :, 1:, : c // 4] = x[:, :, : h - 1, : c // 4]
    x[:, :, : h - 1, c // 4 : c // 2] = x[:, :, 1:, c // 4 : c // 2]
    x[:, 1:, :, c // 2 : c * 3 // 4] = x[:, : w - 1, :, c // 2 : c * 3 // 4]
    x[:, : w - 1, :, 3 * c // 4 :] = x[:, 1:, :, 3 * c // 4 :]
    return x


class SplitAttention(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.channel = channels
        self.k = k
        self.mlp1 = nn.Linear(channels, channels)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channels, channels * k)  # 保持k倍输出
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        assert c == self.channel, f"Input channels {c} != expected {self.channel}"

        # 展平空间维度
        x_all = x_all.reshape(b, k, -1, c)  # (b, k, h*w, c)

        # 全局池化：对空间和k维度求和
        a = torch.sum(x_all, dim=(1, 2))  # (b, c)

        # 通过MLP
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # (b, c*k)
        hat_a = hat_a.reshape(b, self.k, c)  # (b, k, c)

        # 计算注意力权重
        bar_a = self.softmax(hat_a)  # (b, k, c)
        attention = bar_a.unsqueeze(-2)  # (b, k, 1, c)

        # 应用注意力
        out = attention * x_all  # (b, k, h*w, c)
        out = torch.sum(out, dim=1).reshape(b, h, w, c)  # (b, h, w, c)
        return out


class S2Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 调整mlp1维度，确保与后续操作匹配
        self.mlp1 = nn.Linear(channels, channels)  # 原为channels*3
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b, c, w, h = x.size()
        assert c == self.channels, f"Input channels {c} != expected {self.channels}"

        # 调整维度顺序为(b, w, h, c)
        x = x.permute(0, 2, 3, 1)

        # 关键修正：展平空间维度后再通过线性层
        batch_wh = b * w * h
        x_flat = x.reshape(batch_wh, c)
        x_flat = self.mlp1(x_flat)  # 现在输入是(b*w*h, c)，输出也是(b*w*h, c)
        x = x_flat.reshape(b, w, h, c)  # 恢复形状

        # 空间移位操作
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c: c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)  # (b, 3, w, h, c)

        # 通过SplitAttention
        a = self.split_attention(x_all)  # (b, w, h, c)

        # 关键修正：展平后通过mlp2
        a_flat = a.reshape(batch_wh, c)
        x = self.mlp2(a_flat)  # (b*w*h, c)
        x = x.reshape(b, w, h, c)  # 恢复形状

        # 恢复原始维度顺序(b, c, w, h)
        x = x.permute(0, 3, 1, 2)
        return x


# ---------------------------S2-MLPv2 End---------------------------


# ---------------------------NAMAttention Begin---------------------------
class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1


# ---------------------------NAMAttention End---------------------------


# ---------------------------Criss-CrossAttention Begin---------------------------
from torch.nn import Softmax


def INF(B, H, W, dtype=torch.float32):
    """创建与输入数据类型匹配的无穷大张量"""
    # 使用传入的数据类型创建无穷大张量
    return (
        -torch.diag(torch.tensor(float("inf"), dtype=dtype).repeat(H), 0)
        .unsqueeze(0)
        .repeat(B * W, 1, 1)
    )


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    import torch
    import torch.nn as nn
    from torch.nn import Softmax

    def INF(B, H, W, dtype=torch.float32, device=torch.device("cpu")):
        """创建与输入数据类型、设备匹配的无穷大张量"""
        return (
            -torch.diag(torch.tensor(float("inf"), dtype=dtype, device=device).repeat(H), 0)
            .unsqueeze(0)
            .repeat(B * W, 1, 1)
        )

    class CrissCrossAttention(nn.Module):
        """Criss-Cross Attention Module（终极修复版）"""

        def __init__(self, in_dim):
            super(CrissCrossAttention, self).__init__()
            self.query_conv = nn.Conv2d(
                in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
            )
            self.key_conv = nn.Conv2d(
                in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
            )
            self.value_conv = nn.Conv2d(
                in_channels=in_dim, out_channels=in_dim, kernel_size=1
            )
            self.softmax = Softmax(dim=3)
            self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))  # 强制gamma为Float32

        def forward(self, x):
            # 1. 固定为Float32（因AMP=False，无需半精度），同时记录设备
            target_dtype = torch.float32
            device = x.device
            m_batchsize, _, height, width = x.size()

            # 2. 卷积输出强制转Float32+GPU
            proj_query = self.query_conv(x).to(dtype=target_dtype, device=device)
            proj_key = self.key_conv(x).to(dtype=target_dtype, device=device)
            proj_value = self.value_conv(x).to(dtype=target_dtype, device=device)

            # 3. 处理H方向的注意力：全程强制Float32
            proj_query_H = (
                proj_query.permute(0, 3, 1, 2)
                .contiguous()
                .view(m_batchsize * width, -1, height)
                .permute(0, 2, 1)
                .to(dtype=target_dtype, device=device)
            )
            proj_key_H = (
                proj_key.permute(0, 3, 1, 2)
                .contiguous()
                .view(m_batchsize * width, -1, height)
                .to(dtype=target_dtype, device=device)
            )
            proj_value_H = (
                proj_value.permute(0, 3, 1, 2)
                .contiguous()
                .view(m_batchsize * width, -1, height)
                .to(dtype=target_dtype, device=device)
            )

            # 4. 处理W方向的注意力：全程强制Float32
            proj_query_W = (
                proj_query.permute(0, 2, 1, 3)
                .contiguous()
                .view(m_batchsize * height, -1, width)
                .permute(0, 2, 1)
                .to(dtype=target_dtype, device=device)
            )
            proj_key_W = (
                proj_key.permute(0, 2, 1, 3)
                .contiguous()
                .view(m_batchsize * height, -1, width)
                .to(dtype=target_dtype, device=device)
            )
            proj_value_W = (
                proj_value.permute(0, 2, 1, 3)
                .contiguous()
                .view(m_batchsize * height, -1, width)
                .to(dtype=target_dtype, device=device)
            )

            # 5. 创建Float32的INF张量（GPU）
            inf_tensor = INF(m_batchsize, height, width, dtype=target_dtype, device=device)

            # 6. 计算注意力能量：强制Float32
            energy_H = (
                (torch.bmm(proj_query_H, proj_key_H) + inf_tensor)
                .view(m_batchsize, width, height, height)
                .permute(0, 2, 1, 3)
                .to(dtype=target_dtype, device=device)
            )
            energy_W = (
                torch.bmm(proj_query_W, proj_key_W)
                .view(m_batchsize, height, width, width)
                .to(dtype=target_dtype, device=device)
            )

            # 7. 拼接+Softmax：强制Float32
            concate = self.softmax(torch.cat([energy_H, energy_W], 3)).to(dtype=target_dtype, device=device)

            # 8. 拆分注意力权重：强制Float32
            att_H = (
                concate[:, :, :, 0:height]
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(m_batchsize * width, height, height)
                .to(dtype=target_dtype, device=device)
            )
            att_W = (
                concate[:, :, :, height: height + width]
                .contiguous()
                .view(m_batchsize * height, width, width)
                .to(dtype=target_dtype, device=device)
            )

            # ===================== 核心修复：bmm前强制转Float32 =====================
            proj_value_H = proj_value_H.to(dtype=target_dtype, device=device)
            att_H_T = att_H.permute(0, 2, 1).to(dtype=target_dtype, device=device)
            proj_value_W = proj_value_W.to(dtype=target_dtype, device=device)
            att_W_T = att_W.permute(0, 2, 1).to(dtype=target_dtype, device=device)

            # 可选：打印dtype，验证是否为Float32（调试用）
            # print(f"proj_value_H dtype: {proj_value_H.dtype}, att_H_T dtype: {att_H_T.dtype}")

            # 9. 计算输出：使用强制转换后的张量做bmm
            out_H = (
                torch.bmm(proj_value_H, att_H_T)  # 修复后的bmm
                .view(m_batchsize, width, -1, height)
                .permute(0, 2, 3, 1)
                .to(dtype=target_dtype, device=device)
            )
            out_W = (
                torch.bmm(proj_value_W, att_W_T)  # 修复后的bmm
                .view(m_batchsize, height, -1, width)
                .permute(0, 2, 1, 3)
                .to(dtype=target_dtype, device=device)
            )

            # 10. 最终结果：gamma强制转Float32，与输入x（Float32）相加
            output = self.gamma.to(dtype=target_dtype, device=device) * (out_H + out_W) + x.to(dtype=target_dtype,
                                                                                               device=device)
            return output


# ---------------------------Criss-CrossAttention End---------------------------


# ---------------------------GAMAttention Begin---------------------------
class GAMAttention(nn.Module):

    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1),
        )
        self.spatial_attention = nn.Sequential(
            (
                nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            (
                nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        return out


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


# ---------------------------GAMAttention End---------------------------


# ---------------------------Selective Kernel Attention Begin-------------------------
class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "conv",
                                nn.Conv2d(
                                    channel,
                                    channel,
                                    kernel_size=k,
                                    padding=k // 2,
                                    groups=group,
                                ),
                            ),
                            ("bn", nn.BatchNorm2d(channel)),
                            ("relu", nn.ReLU()),
                        ]
                    )
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


# ---------------------------Selective Kernel Attention End---------------------------


# ---------------------------ShuffleAttention Begin---------------------------
from torch.nn.parameter import Parameter


class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


# ---------------------------ShuffleAttention End---------------------------


# ---------------------------A2-Net  Begin---------------------------
class DoubleAttention(nn.Module):

    def __init__(self, in_channels, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        c__ = int(in_channels * 0.5 * 0.5)  # hidden channels
        self.reconstruct = reconstruct
        self.c_m = c_m = c__
        self.c_n = c_n = c__
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=1)
        # step 1: feature gating
        global_descriptors = torch.bmm(
            tmpA, attention_maps.permute(0, 2, 1)
        )  # b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


# ---------------------------A2-Net  End---------------------------


# ---------------------------RFB  Begin---------------------------
class BasicConv(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        scale=0.1,
        map_reduce=8,
        vision=1,
        groups=1,
    ):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                2 * inter_planes,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 1,
                dilation=vision + 1,
                relu=False,
                groups=groups,
            ),
        )
        self.branch1 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                2 * inter_planes,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 2,
                dilation=vision + 2,
                relu=False,
                groups=groups,
            ),
        )
        self.branch2 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                (inter_planes // 2) * 3,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                (inter_planes // 2) * 3,
                2 * inter_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 4,
                dilation=vision + 4,
                relu=False,
                groups=groups,
            ),
        )

        self.ConvLinear = BasicConv(
            6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False
        )
        self.shortcut = BasicConv(
            in_planes, out_planes, kernel_size=1, stride=stride, relu=False
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


# ---------------------------RFB  End---------------------------


# ---------------------------CoTAttention Begin---------------------------
class CoTAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=4,
                bias=False,
            ),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1),
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w

        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


# ---------------------------CoTAttention End---------------------------


# ---------------------------EffectiveSEModule Begin---------------------------
from timm.layers.create_act import create_act_layer


class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False, gate_layer="hard_sigmoid"):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


# ---------------------------EffectiveSEModule End---------------------------


# ---------------------------GlobalContext Begin---------------------------
import torch
from torch import nn as nn
import torch.nn.functional as F
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers import make_divisible


class GlobalContext(nn.Module):
    def __init__(
            self,
            channels,
            use_attn=True,
            fuse_add=False,
            fuse_scale=True,
            init_last_zero=False,
            rd_ratio=1.0 / 8,
            rd_channels=None,
            rd_divisor=1,
            act_layer=nn.ReLU,
            gate_layer="sigmoid",
    ):
        super(GlobalContext, self).__init__()

        # 检查 channels 是否有效
        assert isinstance(channels, int) and channels > 0, f"channels 必须是正整数，当前值为: {channels}"

        # 获取激活层
        act_layer = get_act_layer(act_layer)

        # 生成 self.conv_attn 层，如果 use_attn 为 True，则初始化 conv_attn
        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )

        # 计算 rd_channels，如果为 None，则计算默认值
        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )

        # 检查 rd_channels 的有效性
        assert isinstance(rd_channels, int) and rd_channels > 0, f"rd_channels 必须是正整数，当前值为: {rd_channels}"

        # 选择性创建 mlp_add 层 - 使用简单的卷积层代替ConvMlp
        if fuse_add:
            # 使用简单的卷积层组合，避免timm的ConvMlp问题
            self.mlp_add = nn.Sequential(
                nn.Conv2d(channels, rd_channels, 1),
                nn.BatchNorm2d(rd_channels),  # 使用BatchNorm2d而不是LayerNorm
                create_act_layer(act_layer),
                nn.Conv2d(rd_channels, channels, 1),
                nn.BatchNorm2d(channels)
            )
        else:
            self.mlp_add = None

        # 选择性创建 mlp_scale 层 - 使用简单的卷积层代替ConvMlp
        if fuse_scale:
            # 使用简单的卷积层组合，避免timm的ConvMlp问题
            self.mlp_scale = nn.Sequential(
                nn.Conv2d(channels, rd_channels, 1),
                nn.BatchNorm2d(rd_channels),  # 使用BatchNorm2d而不是LayerNorm
                create_act_layer(act_layer),
                nn.Conv2d(rd_channels, channels, 1),
                nn.BatchNorm2d(channels)
            )
        else:
            self.mlp_scale = None

        # 创建激活层
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(
                self.conv_attn.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add[3].weight)  # 初始化最后一个卷积层权重为0

    def forward(self, x):
        B, C, H, W = x.shape

        # 检查输入是否符合期望的形状
        assert isinstance(x, torch.Tensor), f"x 必须是 torch.Tensor 类型，当前类型为: {type(x)}"
        assert x.dim() == 4, f"x 的维度必须为 4，当前维度为: {x.dim()}"

        # 计算注意力分数
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        # 对输入应用缩放和加法
        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x

# ---------------------------GlobalContext End---------------------------

# ---------------------------GatherExcite Begin---------------------------

# ---------------------------GatherExcite End---------------------------


# ---------------------------MHSA Begin---------------------------
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                requires_grad=True,
            )
            self.rel_w_weight = nn.Parameter(
                torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                requires_grad=True,
            )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (
                (self.rel_h_weight + self.rel_w_weight)
                .view(1, self.heads, C // self.heads, -1)
                .permute(0, 1, 3, 2)
            )  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = (
                content_position
                if (content_content.shape == content_position.shape)
                else content_position[
                    :,
                    :,
                    :c3,
                ]
            )
            assert content_content.shape == content_position.shape
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


# ---------------------------MHSA End---------------------------


# ---------------------------ParNetAttention Begin---------------------------
class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid(),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1), nn.BatchNorm2d(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


# ---------------------------ParNetAttention End---------------------------


# ---------------------------ParallelPolarizedSelfAttention Begin---------------------------
class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = (
            self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            )
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out


# ---------------------------ParallelPolarizedSelfAttention End---------------------------


# ---------------------------SpatialGroupEnhance Begin---------------------------
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


# ---------------------------SpatialGroupEnhance End---------------------------


# ---------------------------SequentialPolarizedSelfAttention Begin---------------------------
class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = (
            self.sigmoid(
                self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))
            )
            .permute(0, 2, 1)
            .reshape(b, c, 1, 1)
        )  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * channel_out
        return spatial_out


# ---------------------------SequentialPolarizedSelfAttention End---------------------------


# ---------------------------TripletAttention Begin---------------------------
class BasicConv_T(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv_T, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv_T(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


# ---------------------------TripletAttention End---------------------------

# ----------------------SE------------------
class SE(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


# ------------------------------------------以下是追加的部分代码------------------------------------------
class SE_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.se = SE(c2, 16)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.se(self.cv2(self.cv1(x))) if self.add else self.se(self.cv2(self.cv1(x)))


class C2f_SE(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 SE. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(SE_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# ------------------------------------------以下是追加的部分代码------------------------------------------
class CBAM_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cbam = CBAM(c2, 7)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cbam(self.cv2(self.cv1(x))) if self.add else self.cbam(self.cv2(self.cv1(x)))


class C2f_CBAM(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 CBAM. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CBAM_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


# ------------------------------------------以下是追加的部分代码------------------------------------------
class CA_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.ca = CoordAtt(c2, c2, 32)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.ca(self.cv2(self.cv1(x))) if self.add else self.ca(self.cv2(self.cv1(x)))


class C2f_CA(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 CoordAtt. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CA_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# ECA
class ECA(nn.Module):

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ------------------------------------------以下是追加的部分代码------------------------------------------
class ECA_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.eca = ECA(3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.eca(self.cv2(self.cv1(x))) if self.add else self.eca(self.cv2(self.cv1(x)))


class C2f_ECA(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 ECA. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(ECA_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------

# EffectiveSE
class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False, gate_layer='hard_sigmoid'):
        super(EffectiveSEModule, self).__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


# ------------------------------------------以下是追加的部分代码------------------------------------------
class EffectiveSE_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.effse = EffectiveSEModule(c2, add_maxpool=False, gate_layer='hard_sigmoid')
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.effse(self.cv2(self.cv1(x))) if self.add else self.effse(self.cv2(self.cv1(x)))


class C2f_EffectiveSE(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 EffectiveSE. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            EffectiveSE_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# GlobalContext
import torch
from torch import nn as nn
import torch.nn.functional as F
from timm.layers.create_act import create_act_layer, get_act_layer
from timm.layers import make_divisible
from timm.layers.mlp import ConvMlp
from timm.layers.norm import LayerNorm2d


class GlobalContext(nn.Module):
    def __init__(
            self,
            channels,
            use_attn=True,
            fuse_add=False,
            fuse_scale=True,
            init_last_zero=False,
            rd_ratio=1.0 / 8,
            rd_channels=None,
            rd_divisor=1,
            act_layer=nn.ReLU,
            gate_layer="sigmoid",
    ):
        super(GlobalContext, self).__init__()

        # 检查 channels 是否有效
        assert isinstance(channels, int) and channels > 0, f"channels 必须是正整数，当前值为: {channels}"

        # 获取激活层
        act_layer = get_act_layer(act_layer)

        # 生成 self.conv_attn 层，如果 use_attn 为 True，则初始化 conv_attn
        self.conv_attn = (
            nn.Conv2d(channels, 1, kernel_size=1, bias=True) if use_attn else None
        )

        # 计算 rd_channels，如果为 None，则计算默认值
        if rd_channels is None:
            rd_channels = make_divisible(
                channels * rd_ratio, rd_divisor, round_limit=0.0
            )

        # 检查 rd_channels 的有效性
        assert isinstance(rd_channels, int) and rd_channels > 0, f"rd_channels 必须是正整数，当前值为: {rd_channels}"

        # 选择性创建 mlp_add 层
        if fuse_add:
            self.mlp_add = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_add = None

        # 选择性创建 mlp_scale 层
        if fuse_scale:
            self.mlp_scale = ConvMlp(
                channels, rd_channels, act_layer=act_layer, norm_layer=LayerNorm2d
            )
        else:
            self.mlp_scale = None

        # 创建激活层
        self.gate = create_act_layer(gate_layer)
        self.init_last_zero = init_last_zero

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        if self.conv_attn is not None:
            nn.init.kaiming_normal_(
                self.conv_attn.weight, mode="fan_in", nonlinearity="relu"
            )
        if self.mlp_add is not None:
            nn.init.zeros_(self.mlp_add.fc2.weight)

    def forward(self, x):
        B, C, H, W = x.shape

        # 检查输入是否符合期望的形状
        assert isinstance(x, torch.Tensor), f"x 必须是 torch.Tensor 类型，当前类型为: {type(x)}"
        assert x.dim() == 4, f"x 的维度必须为 4，当前维度为: {x.dim()}"

        # 计算注意力分数
        if self.conv_attn is not None:
            attn = self.conv_attn(x).reshape(B, 1, H * W)  # (B, 1, H * W)
            attn = F.softmax(attn, dim=-1).unsqueeze(3)  # (B, 1, H * W, 1)
            context = x.reshape(B, C, H * W).unsqueeze(1) @ attn
            context = context.view(B, C, 1, 1)
        else:
            context = x.mean(dim=(2, 3), keepdim=True)

        # 对输入应用缩放和加法
        if self.mlp_scale is not None:
            mlp_x = self.mlp_scale(context)
            x = x * self.gate(mlp_x)
        if self.mlp_add is not None:
            mlp_x = self.mlp_add(context)
            x = x + mlp_x

        return x


# ------------------------------------------以下是追加的部分代码------------------------------------------
class GlobalContext_Bottleneck_(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)

        # 替换原来的GlobalContext为简化的注意力模块
        # self.effse = GlobalContext(c2)  # 注释掉这一行
        self.effse = nn.Sequential(  # 使用简化的注意力模块
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 16, 8), 1, bias=False),
            nn.BatchNorm2d(max(c2 // 16, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(c2 // 16, 8), c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        return (
            x + self.effse(self.cv2(self.cv1(x)))
            if self.add
            else self.effse(self.cv2(self.cv1(x)))
        )


class C2f_GlobalContext(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 GlobalContext. by csdn迪菲赫尔曼"""

    def __init__(
            self, c1, c2, n=2, shortcut=False, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        assert n > 0, "n 必须大于 0"
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            GlobalContext_Bottleneck_(
                self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0
            )
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        for idx, m in enumerate(self.m):
            y.append(m(y[-1]))
        concatenated_shape = torch.cat(y, 1).shape
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# ------------------------------------------以上是追加的部分代码------------------------------------------
# GatherExcite


class GatherExcite(nn.Module):
    def __init__(
            self, channels, feat_size=None, extra_params=False, extent=0, use_mlp=True,
            rd_ratio=1. / 16, rd_channels=None, rd_divisor=1, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, gate_layer='sigmoid'):
        super(GatherExcite, self).__init__()
        self.add_maxpool = add_maxpool
        act_layer = get_act_layer(act_layer)
        self.extent = extent
        if extra_params:
            self.gather = nn.Sequential()
            if extent == 0:
                assert feat_size is not None, 'spatial feature size must be specified for global extent w/ params'
                self.gather.add_module(
                    'conv1', create_conv2d(channels, channels, kernel_size=feat_size, stride=1, depthwise=True))
                if norm_layer:
                    self.gather.add_module(f'norm1', nn.BatchNorm2d(channels))
            else:
                assert extent % 2 == 0
                num_conv = int(math.log2(extent))
                for i in range(num_conv):
                    self.gather.add_module(
                        f'conv{i + 1}',
                        create_conv2d(channels, channels, kernel_size=3, stride=2, depthwise=True))
                    if norm_layer:
                        self.gather.add_module(f'norm{i + 1}', nn.BatchNorm2d(channels))
                    if i != num_conv - 1:
                        self.gather.add_module(f'act{i + 1}', act_layer(inplace=True))
        else:
            self.gather = None
            if self.extent == 0:
                self.gk = 0
                self.gs = 0
            else:
                assert extent % 2 == 0
                self.gk = self.extent * 2 - 1
                self.gs = self.extent

        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.mlp = ConvMlp(channels, rd_channels, act_layer=act_layer) if use_mlp else nn.Identity()
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        size = x.shape[-2:]
        if self.gather is not None:
            x_ge = self.gather(x)
        else:
            if self.extent == 0:
                # global extent
                x_ge = x.mean(dim=(2, 3), keepdims=True)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * x.amax((2, 3), keepdim=True)
            else:
                x_ge = F.avg_pool2d(
                    x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2, count_include_pad=False)
                if self.add_maxpool:
                    # experimental codepath, may remove or change
                    x_ge = 0.5 * x_ge + 0.5 * F.max_pool2d(x, kernel_size=self.gk, stride=self.gs, padding=self.gk // 2)
        x_ge = self.mlp(x_ge)
        if x_ge.shape[-1] != 1 or x_ge.shape[-2] != 1:
            x_ge = F.interpolate(x_ge, size=size)
        return x * self.gate(x_ge)


# ------------------------------------------以下是追加的部分代码------------------------------------------
class GatherExcite_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.effse = GatherExcite(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.effse(self.cv2(self.cv1(x))) if self.add else self.effse(self.cv2(self.cv1(x)))


class C2f_GatherExcite(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 GatherExcite. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            GatherExcite_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# MHSA
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.pos = pos_emb
        if self.pos:
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # 1,C,h*w,h*w
        c1, c2, c3, c4 = content_content.size()
        if self.pos:
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # 1,4,1024,64

            content_position = torch.matmul(content_position, q)  # ([1, 4, 1024, 256])
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]
            assert (content_content.shape == content_position.shape)
            energy = content_content + content_position
        else:
            energy = content_content
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # 1,4,256,64
        out = out.view(n_batch, C, width, height)
        return out


# ------------------------------------------以下是追加的部分代码------------------------------------------
class MHSA_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.mhsa = MHSA(c2, 14, 14 ,4 , False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.mhsa(self.cv2(self.cv1(x))) if self.add else self.mhsa(self.cv2(self.cv1(x)))


class C2f_MHSA(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 MHSA. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            MHSA_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# SimAM
class SimAM(torch.nn.Module):

    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# ------------------------------------------以下是追加的部分代码------------------------------------------
class SimAM_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.simam = SimAM(e_lambda=1e-4)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.simam(self.cv2(self.cv1(x))) if self.add else self.simam(self.cv2(self.cv1(x)))


class C2f_SimAM(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 SimAM. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            SimAM_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# CoTAttention
class CoTAttention(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  # bs,c,h,w
        v = self.value_embed(x).view(bs, c, -1)  # bs,c,h,w
        y = torch.cat([k1, x], dim=1)  # bs,2c,h,w
        att = self.attention_embed(y)  # bs,c*k*k,h,w
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # bs,c,h*w
        k2 = F.softmax(att, dim=-1) * v
        k2 = k2.view(bs, c, h, w)

        return k1 + k2


# ------------------------------------------以下是追加的部分代码------------------------------------------
class CoT_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cot = CoTAttention(c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cot(self.cv2(self.cv1(x))) if self.add else self.cot(self.cv2(self.cv1(x)))


class C2f_CoT(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 CoTAttention. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(CoT_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# SK
class SKAttention(nn.Module):

    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


# ------------------------------------------以下是追加的部分代码------------------------------------------
class SK_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.sk = SKAttention(c2, reduction=16)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.sk(self.cv2(self.cv1(x))) if self.add else self.sk(self.cv2(self.cv1(x)))


class C2f_SK(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 SKAttention. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(SK_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# DoubleAttention
class DoubleAttention(nn.Module):

    def __init__(self, in_channels, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels
        c__ = int(in_channels * 0.5 * 0.5)  # hidden channels
        self.reconstruct = reconstruct
        self.c_m = c_m = c__
        self.c_n = c_n = c__
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels
        A = self.convA(x)  # b,c_m,h,w
        B = self.convB(x)  # b,c_n,h,w
        V = self.convV(x)  # b,c_n,h,w
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1), dim=1)
        attention_vectors = F.softmax(V.view(b, self.c_n, -1), dim=1)
        # step 1: feature gating
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)

        return tmpZ


# ------------------------------------------以下是追加的部分代码------------------------------------------
class Double_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.doubles = DoubleAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.doubles(self.cv2(self.cv1(x))) if self.add else self.doubles(self.cv2(self.cv1(x)))


class C2f_Double(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 DoubleAttention. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Double_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# ------------------------------------------------
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from einops import rearrange


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)

        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights
    """

    def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3,
                 auto_pad=True):
        super().__init__()
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity':  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor
        Return:
            NHWC tensor
        """
        x = rearrange(x, "n c h w -> n h w c")
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0,  # dim=-1
                          pad_l, pad_r,  # dim=-2
                          pad_t, pad_b))  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            N, H, W, C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0  #
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)
        q, kv = self.qkv(x)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean(
            [2, 3])  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)
        lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win,
                                   i=self.n_win).contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)
        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c',
                              m=self.num_heads)  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c',
                          m=self.num_heads)  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)
        attn_weight = (
                                  q_pix * self.scale) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)
        out = out + lepe
        out = self.wo(out)
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")


# TripletAttention
class BasicConv_T(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv_T, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv_T(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


# ------------------------------------------以下是追加的部分代码------------------------------------------
class Triplet_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.triplet = TripletAttention(False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.triplet(self.cv2(self.cv1(x))) if self.add else self.triplet(self.cv2(self.cv1(x)))


class C2f_Triplet(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 TripletAttention. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            Triplet_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# SpatialGroupEnhance
class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=8):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


# ------------------------------------------以下是追加的部分代码------------------------------------------
class SpatialGroupEnhance_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.sg = SpatialGroupEnhance(groups=8)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.sg(self.cv2(self.cv1(x))) if self.add else self.sg(self.cv2(self.cv1(x)))


class C2f_SpatialGroupEnhance(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 SpatialGroupEnhance. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            SpatialGroupEnhance_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# C2f_NAM
class Channel_Att(nn.Module):
    def __init__(self, channels):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1

# ------------------------------------------以下是追加的部分代码------------------------------------------
class NAM_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.nam = NAMAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.nam(self.cv2(self.cv1(x))) if self.add else self.nam(self.cv2(self.cv1(x)))


class C2f_NAM(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 SpatialGroupEnhance. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            NAM_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# S2
import torch
import torch.nn as nn
import torch.nn.functional as F

# 补充缺失的spatial_shift1和spatial_shift2函数（原代码中被调用但未定义）
def spatial_shift1(x):
    b, h, w, c = x.size()
    # 右移：第一列补0，其余列左移
    x[:, :, 1:, :c//4] = x[:, :, :-1, :c//4].clone()
    x[:, :, 0, :c//4] = 0
    # 下移：第一行补0，其余行上移
    x[:, 1:, :, c//4:c//2] = x[:, :-1, :, c//4:c//2].clone()
    x[:, 0, :, c//4:c//2] = 0
    return x

def spatial_shift2(x):
    b, h, w, c = x.size()
    # 左移：最后一列补0，其余列右移
    x[:, :, :-1, :c//4] = x[:, :, 1:, :c//4].clone()
    x[:, :, -1, :c//4] = 0
    # 上移：最后一行补0，其余行下移
    x[:, :-1, :, c//4:c//2] = x[:, 1:, :, c//4:c//2].clone()
    x[:, -1, :, c//4:c//2] = 0
    return x

# 修正SplitAttention：保留默认值，但支持外部传递channel和k
class SplitAttention(nn.Module):
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel  # 由外部动态传递，不再固定
        self.k = k
        # 线性层维度匹配传入的channel
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        b, k, h, w, c = x_all.shape  # 输入维度：(b,3,h,w,c)
        x_all = x_all.reshape(b, k, -1, c)  # (b,3,h*w,c)
        a = torch.sum(torch.sum(x_all, 1), 1)  # 压缩维度：(b,c)
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # 现在c=256，与mlp1输入匹配
        hat_a = hat_a.reshape(b, self.k, c)  # (b,3,c)
        bar_a = self.softmax(hat_a)  # 注意力权重归一化
        attention = bar_a.unsqueeze(-2)  # (b,3,1,c)
        out = attention * x_all  # 加权：(b,3,h*w,c)
        out = torch.sum(out, 1).reshape(b, h, w, c)  # 融合3路：(b,h,w,c)
        return out

# 修正S2Attention：动态接收通道数，并传递给SplitAttention
class S2Attention(nn.Module):
    def __init__(self, channels=256):  # 默认值改为256（适配你的yaml），支持外部传入
        super().__init__()
        self.channels = channels  # 动态通道数
        # MLP1输出channels*3，匹配后续x1/x2/x3的分割逻辑
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        # 关键：将动态通道数传递给SplitAttention，不再使用默认的512
        self.split_attention = SplitAttention(channel=channels, k=3)

    def forward(self, x):
        b, c, h, w = x.size()  # 输入维度：(b,256,h,w)（c=channels=256）
        x = x.permute(0, 2, 3, 1)  # 维度变换：(b,h,w,c)
        x = self.mlp1(x)  # 输出：(b,h,w,3c) → 匹配x1/x2/x3的分割
        # 特征分割：3c拆分为3个c，与原逻辑一致
        x1 = spatial_shift1(x[:, :, :, :c])
        x2 = spatial_shift2(x[:, :, :, c:c * 2])
        x3 = x[:, :, :, c * 2:]
        x_all = torch.stack([x1, x2, x3], 1)  # 堆叠为3路：(b,3,h,w,c)
        a = self.split_attention(x_all)  # 调用修正后的SplitAttention
        x = self.mlp2(a)  # 输出：(b,h,w,c)
        x = x.permute(0, 3, 1, 2)  # 还原维度：(b,c,h,w)
        return x

# 以下代码无需修改，仅需确保Conv类已在ultralytics中定义
class S2_Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.s2 = S2Attention(c2)  # 传递c2给S2Attention，动态适配通道数
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.s2(self.cv2(self.cv1(x))) if self.add else self.s2(self.cv2(self.cv1(x)))

class C2f_S2(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 SpatialGroupEnhance."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            S2_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------

# ParNet
class ParNetAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel)
        )
        self.silu = nn.SiLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.sse(x) * x
        y = self.silu(x1 + x2 + x3)
        return y


# ------------------------------------------以下是追加的部分代码------------------------------------------
class ParNet_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.parnet = ParNetAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.parnet(self.cv2(self.cv1(x))) if self.add else self.parnet(self.cv2(self.cv1(x)))


class C2f_ParNet(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 ParNet. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            ParNet_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# CrissCross
from torch.nn import Softmax


def INF(B, H, W, device):
    # Create an infinite diagonal tensor on the specified device
    return (
        -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0)
        .unsqueeze(0)
        .repeat(B * W, 1, 1)
    )


class CrissCrossAttention(nn.Module):
    """Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        device = x.device
        self.to(device)
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = (
            proj_query.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
            .permute(0, 2, 1)
        )
        proj_query_W = (
            proj_query.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
            .permute(0, 2, 1)
        )
        proj_key = self.key_conv(x)
        proj_key_H = (
            proj_key.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_key_W = (
            proj_key.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        proj_value = self.value_conv(x)
        proj_value_H = (
            proj_value.permute(0, 3, 1, 2)
            .contiguous()
            .view(m_batchsize * width, -1, height)
        )
        proj_value_W = (
            proj_value.permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * height, -1, width)
        )
        energy_H = (
            (
                torch.bmm(proj_query_H, proj_key_H)
                + self.INF(m_batchsize, height, width, device)
            )
            .view(m_batchsize, width, height, height)
            .permute(0, 2, 1, 3)
        )
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(
            m_batchsize, height, width, width
        )
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = (
            concate[:, :, :, 0:height]
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(m_batchsize * width, height, height)
        )
        # print(concate)
        # print(att_H)
        att_W = (
            concate[:, :, :, height : height + width]
            .contiguous()
            .view(m_batchsize * height, width, width)
        )
        out_H = (
            torch.bmm(proj_value_H, att_H.permute(0, 2, 1))
            .view(m_batchsize, width, -1, height)
            .permute(0, 2, 3, 1)
        )
        out_W = (
            torch.bmm(proj_value_W, att_W.permute(0, 2, 1))
            .view(m_batchsize, height, -1, width)
            .permute(0, 2, 1, 3)
        )
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

# ------------------------------------------以下是追加的部分代码------------------------------------------
class CrissCross_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cca = CrissCrossAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cca(self.cv2(self.cv1(x))) if self.add else self.cca(self.cv2(self.cv1(x)))


class C2f_CrissCross(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 CrissCross. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            CrissCross_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# GAM
class GAMAttention(nn.Module):

    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        return out


def channel_shuffle(x, groups=2):  ##shuffle channel
    # RESHAPE----->transpose------->Flatten
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


# ------------------------------------------以下是追加的部分代码------------------------------------------
class GAM_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.gam = GAMAttention(c2, c2, True, 4)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.gam(self.cv2(self.cv1(x))) if self.add else self.gam(self.cv2(self.cv1(x)))


class C2f_GAM(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 GAM. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            GAM_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# ParallelPolarized
class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2,
                                                                                                                 1).reshape(
            b, c, 1, 1)  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(x)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(x)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out


# ------------------------------------------以下是追加的部分代码------------------------------------------
class ParallelPolarized_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.papo = ParallelPolarizedSelfAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.papo(self.cv2(self.cv1(x))) if self.add else self.papo(self.cv2(self.cv1(x)))


class C2f_ParallelPolarized(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 ParallelPolarized. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            ParallelPolarized_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# SequentialPolarizedSelfAttention
class SequentialPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel-only Self-Attention
        channel_wv = self.ch_wv(x)  # bs,c//2,h,w
        channel_wq = self.ch_wq(x)  # bs,1,h,w
        channel_wv = channel_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        channel_wq = channel_wq.reshape(b, -1, 1)  # bs,h*w,1
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)  # bs,c//2,1,1
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2,
                                                                                                                 1).reshape(
            b, c, 1, 1)  # bs,c,1,1
        channel_out = channel_weight * x

        # Spatial-only Self-Attention
        spatial_wv = self.sp_wv(channel_out)  # bs,c//2,h,w
        spatial_wq = self.sp_wq(channel_out)  # bs,c//2,h,w
        spatial_wq = self.agp(spatial_wq)  # bs,c//2,1,1
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)  # bs,c//2,h*w
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)  # bs,1,c//2
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)  # bs,1,h*w
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))  # bs,1,h,w
        spatial_out = spatial_weight * channel_out
        return spatial_out


# ------------------------------------------以下是追加的部分代码------------------------------------------
class SequentialPolarized_Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.sep = SequentialPolarizedSelfAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.sep(self.cv2(self.cv1(x))) if self.add else self.sep(self.cv2(self.cv1(x)))


class C2f_SequentialPolarized(nn.Module):
    """CSP Bottleneck with 2 convolutions and 1 SequentialPolarized. by csdn迪菲赫尔曼"""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            SequentialPolarized_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass of a YOLOv5 CSPDarknet backbone layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Applies spatial attention to module's input."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ------------------------------------------以上是追加的部分代码------------------------------------------
# BiFPN
# 两个特征图add操作
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))


# 三个特征图add操作
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))


# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   RAMiT.py
@Time      :   2024/09/11 20:40:14
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""

import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch
import math


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class QKVProjection(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=True):
        super(QKVProjection, self).__init__()
        self.dim = dim
        self.num_head = num_head

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b (l c) h w -> b l c h w", l=self.num_head)
        return qkv

    def flops(self, resolutions):
        return resolutions[0] * resolutions[1] * 1 * 1 * self.dim * 3 * self.dim


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(
        torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing="ij")
    )  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # 2, Wh*Ww, Wh*Ww (xaxis matrix & yaxis matrix)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, window_size=4, shift=0,
                 attn_drop=0.0, proj_drop=0.0, helper=True):
        super(SpatialSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.window_size = window_size
        self.window_area = window_size ** 2
        self.shift = shift
        self.helper = helper

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head)
        )

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_area, self.window_area, -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, qkv, ch=None):
        B, L, C, H, W = qkv.size()
        H0, W0 = H, W  # Store original dimensions

        # Calculate padding if needed
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        qkv = F.pad(qkv, (0, pad_w, 0, pad_h))

        # Update dimensions after padding
        _, _, _, H, W = qkv.size()

        if self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2, -1))

        q, k, v = rearrange(
            qkv,
            "b l c (h wh) (w ww) -> (b h w) l (wh ww) c",
            wh=self.window_size,
            ww=self.window_size,
        ).chunk(3, dim=-1)

        if ch is not None and self.helper:
            if self.shift > 0:
                ch = torch.roll(ch, shifts=(-self.shift, -self.shift), dims=(-2, -1))
            ch = F.pad(ch, (0, pad_w, 0, pad_h))  # Pad ch to match qkv dimensions
            ch = rearrange(
                ch,
                "b (l c) (h wh) (w ww) -> (b h w) l (wh ww) c",
                l=self.total_head - self.num_head,
                wh=self.window_size,
                ww=self.window_size,
            )
            ch = torch.mean(ch, dim=1, keepdim=True)
            v = v * ch

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        attn = attn + self._get_rel_pos_bias()
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        x = attn @ v

        # Window unpartition + head merge
        x = window_unpartition(x, (H, W), self.window_size)
        x = self.proj_drop(self.proj(x))

        # Reverse shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(-2, -1))

        # Unpad the output to original dimensions
        x = x[:, :, :H0, :W0]

        return x

    def flops(self, resolutions):
        H, W = resolutions
        num_wins = H // self.window_size * W // self.window_size
        flops = self.num_head * H * W * self.dim if self.helper else 0  # v = v*ch
        flops += (
                num_wins * self.num_head * self.window_area * self.dim * self.window_area
        )  # attn = Q@K^T
        flops += (
                num_wins * self.num_head * self.window_area * self.window_area * self.dim
        )  # attn@V
        flops += (
                H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim
        )  # self.proj
        return flops


def window_unpartition(x, resolutions, window_size):
    return rearrange(
        x,
        "(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)",
        h=resolutions[0] // window_size,
        w=resolutions[1] // window_size,
        wh=window_size,
    )


class ChannelSelfAttention(nn.Module):
    def __init__(
            self, dim, num_head, total_head, attn_drop=0.0, proj_drop=0.0, helper=True
    ):
        super(ChannelSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.helper = helper

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True
        )

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, qkv, sp=None):
        B, L, C, H, W = qkv.size()

        q, k, v = rearrange(qkv, "b l c h w -> b l c (h w)").chunk(
            3, dim=-2
        )  # [B, L2, C/L, HW]
        if sp is not None and self.helper:
            sp = torch.mean(sp, dim=1, keepdim=True)  # channel squeeze # [B, 1, H, W]
            sp = rearrange(sp, "b (l c) h w -> b l c (h w)", l=1)  # [B, 1, 1, HW]
            v = v * sp  # [B, L2, C/L, HW]

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(
            2, -1
        )  # [B, L2, C/L, C/L]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, L2, C/L, HW]

        # head merge
        x = rearrange(x, "b l c (h w) -> b (l c) h w", h=H)  # [B, L2*C/L, H, W]
        x = self.proj_drop(self.proj(x))  # [B, L2*C/L, H, W]

        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = self.num_head * self.dim * H * W if self.helper else 0  # v = v*sp
        flops += self.num_head * self.dim * H * W * self.dim  # attn = Q@K^T
        flops += self.num_head * self.dim * self.dim * H * W  # attn@V
        flops += (
                H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim
        )  # self.proj
        return flops


class ReshapeLayerNorm(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(ReshapeLayerNorm, self).__init__()

        self.dim = dim
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H)
        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += H * W * self.dim
        return flops


class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(
            dim, dim, kernel_size, stride, kernel_size // 2, groups=dim
        )
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = (
                H * W * self.kernel_size * self.kernel_size * self.dim
                + H * W * 1 * 1 * self.dim * self.out_dim
        )  # self.dw_conv + self.pw_conv
        return flops


class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(
            self,
            dim,
            kernel_size,
            stride,
            act=nn.LeakyReLU,
            out_dim=None,
            exp_factor=1.2,
            expand_groups=4,
    ):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(
            expand_dim,
            expand_dim,
            kernel_size,
            stride,
            kernel_size // 2,
            groups=expand_dim,
        )
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = (
                H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim
        )  # self.exp_conv
        flops += (
                H * W * self.kernel_size * self.kernel_size * self.expand_dim
        )  # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim  # self.pw_conv
        return flops


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_ratio, act_layer=nn.GELU, bias=True, drop=0.0):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.hidden_ratio = hidden_ratio

        self.hidden = nn.Conv2d(dim, int(dim * hidden_ratio), 1, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.out = nn.Conv2d(int(dim * hidden_ratio), dim, 1, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.act = act_layer()

    def forward(self, x):
        return self.drop2(self.out(self.drop1(self.act(self.hidden(x)))))

    def flops(self, resolutions):
        H, W = resolutions
        flops = (
                2 * H * W * 1 * 1 * self.dim * self.dim * self.hidden_ratio
        )  # self.hidden + self.out
        return flops


class NoLayer(nn.Identity):
    def __init__(self):
        super(NoLayer, self).__init__()

    def flops(self, resolutions):
        return 0

    def forward(self, x, **kwargs):
        return x.flatten(1, 2)


class DRAMiTransformer(nn.Module):  # Reciprocal Attention Transformer Block
    def __init__(
            self,
            dim,
            num_head=4,
            chsa_head_ratio=0.25,
            window_size=4,
            shift=0,
            head_dim=None,
            qkv_bias=True,
            mv_ver=1,
            hidden_ratio=2.0,
            act_layer=nn.GELU,
            norm_layer=ReshapeLayerNorm,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,
            helper=True,
            mv_act=nn.LeakyReLU,
            exp_factor=1.2,
            expand_groups=4,
    ):
        super(DRAMiTransformer, self).__init__()

        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.chsa_head = int(num_head * chsa_head_ratio)
        self.shift = shift
        self.helper = helper

        self.qkv_proj = QKVProjection(dim, num_head, qkv_bias=qkv_bias)
        self.sp_attn = (
            SpatialSelfAttention(
                dim // num_head,
                num_head - self.chsa_head,
                num_head,
                window_size,
                shift,
                attn_drop,
                proj_drop,
                helper,
            )
            if num_head - self.chsa_head != 0
            else NoLayer()
        )
        self.ch_attn = (
            ChannelSelfAttention(
                dim // num_head, self.chsa_head, num_head, attn_drop, proj_drop, helper
            )
            if self.chsa_head != 0
            else NoLayer()
        )
        if mv_ver == 1:
            self.mobivari = MobiVari1(dim, 3, 1, act=mv_act)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(
                dim,
                3,
                1,
                act=mv_act,
                out_dim=None,
                exp_factor=exp_factor,
                expand_groups=expand_groups,
            )

        self.norm1 = norm_layer(dim)

        self.ffn = FeedForward(dim, hidden_ratio, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, sp_=None, ch_=None):
        B, C, H, W = x.size()

        # QKV projection + head split
        qkv = self.qkv_proj(x)  # [B, L, C, H, W]

        # SP-SA / CH-SA
        sp = self.sp_attn(qkv[:, :self.num_head - self.chsa_head], ch=ch_)  # [B, L1*C/L, H, W]
        ch = self.ch_attn(qkv[:, self.num_head - self.chsa_head:], sp=sp_)  # [B, L2*C/L, H, W]

        # Ensure spatial dimensions match before concatenation
        if sp.size(2) != ch.size(2) or sp.size(3) != ch.size(3):
            ch = F.interpolate(ch, size=(sp.size(2), sp.size(3)), mode='bilinear', align_corners=False)

        attn0 = self.mobivari(torch.cat([sp, ch], dim=1))  # merge [B, C, H, W]
        attn = self.drop_path(self.norm1(attn0)) + x  # LN, skip connection [B, C, H, W]

        # FFN
        out = self.drop_path(self.norm2(self.ffn(attn))) + attn  # FFN, LN, skip connection [B, C, H, W]

        return out

    def flops(self, resolutions):
        flops = self.qkv_proj.flops(resolutions)
        flops += self.sp_attn.flops(resolutions)
        flops += self.ch_attn.flops(resolutions)
        flops += self.mobivari.flops(resolutions)
        flops += self.norm1.flops(resolutions)
        flops += self.ffn.flops(resolutions)
        flops += self.norm2.flops(resolutions)
        params = sum([p.numel() for n, p in self.named_parameters()])
        return flops


if __name__ == "__main__":
    block = DRAMiTransformer(dim=64)
    input = torch.randn(1, 64, 32, 32)
    output = block(input)
    print(input.size())
    print(output.size())


class SeBlock(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super().__init__()
        self.Squeeze = nn.AdaptiveAvgPool2d(1)

        self.Excitation = nn.Sequential()
        self.Excitation.add_module('FC1', nn.Conv2d(in_channel, in_channel // reduction, kernel_size=1))  # 1*1卷积与此效果相同
        self.Excitation.add_module('ReLU', nn.ReLU())
        self.Excitation.add_module('FC2', nn.Conv2d(in_channel // reduction, in_channel, kernel_size=1))
        self.Excitation.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        y = self.Squeeze(x)
        ouput = self.Excitation(y)
        return x * (ouput.expand_as(x))

class G_bneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, midc, k=5, s=1, use_se = False):  # ch_in, ch_mid, ch_out, kernel, stride, use_se
        super().__init__()
        assert s in [1, 2]
        c_ = midc
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),              # Expansion
                                  Conv(c_, c_, 3, s=2, p=1, g=c_, act=False) if s == 2 else nn.Identity(),  # dw
                                  # Squeeze-and-Excite
                                  SeBlock(c_) if use_se else nn.Sequential(),
                                  GhostConv(c_, c2, 1, 1, act=False))   # Squeeze pw-linear

        self.shortcut = nn.Identity() if (c1 == c2 and s == 1) else \
                                                nn.Sequential(Conv(c1, c1, 3, s=s, p=1, g=c1, act=False), \
                                                Conv(c1, c2, 1, 1, act=False)) # 避免stride=2时 通道数改变的情况

    def forward(self, x):
        # print(self.conv(x).shape)
        # print(self.shortcut(x).shape)
        return self.conv(x) + self.shortcut(x)


# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace
    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)
def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()
    if isinstance(name, str):
        if name == 'silu':
            module = nn.SiLU(inplace=inplace)
        elif name == 'relu':
            module = nn.ReLU(inplace=inplace)
        elif name == 'lrelu':
            module = nn.LeakyReLU(0.1, inplace=inplace)
        elif name == 'swish':
            module = Swish(inplace=inplace)
        elif name == 'hardsigmoid':
            module = nn.Hardsigmoid(inplace=inplace)
        elif name == 'identity':
            module = nn.Identity()
        else:
            raise AttributeError('Unsupported act type: {}'.format(name))
        return module
    elif isinstance(name, nn.Module):
        return name
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
def get_norm(name, out_channels, inplace=True):
    if name == 'bn':
        module = nn.BatchNorm2d(out_channels)
    else:
        raise NotImplementedError
    return module
class ConvBNAct(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize,
        stride=1,
        groups=1,
        bias=False,
        act='silu',
        norm='bn',
        reparam=False,
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        if norm is not None:
            self.bn = get_norm(norm, out_channels, inplace=True)
        if act is not None:
            self.act = get_activation(act, inplace=True)
        self.with_norm = norm is not None
        self.with_act = act is not None
    def forward(self, x):
        x = self.conv(x)
        if self.with_norm:
            x = self.bn(x)
        if self.with_act:
            x = self.act(x)
        return x
    def fuseforward(self, x):
        return self.act(self.conv(x))
class RepConv(nn.Module):
    '''RepConv is a basic rep-style block, including training and deploy status
    Code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act='relu',
                 norm=None):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        if isinstance(act, str):
            self.nonlinearity = get_activation(act)
        else:
            self.nonlinearity = act
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=True,
                                         padding_mode=padding_mode)
        else:
            self.rbr_identity = None
            self.rbr_dense = conv_bn(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=padding_11,
                                   groups=groups)
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True
class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels,
                               hidden_channels,
                               1,
                               stride=1,
                               act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = ConvBNAct(conv2_channels,
                               out_channels,
                               1,
                               stride=1,
                               act=activation)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x
class Focus(nn.Module):
    """Focus width and height information into channel space."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super().__init__()
        self.conv = ConvBNAct(in_channels * 4,
                              out_channels,
                              ksize,
                              stride,
                              act=act)
    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)
class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_hidden_ratio,
                 ch_out,
                 act='relu',
                 shortcut=True):
        super(BasicBlock_3x3_Reverse, self).__init__()
        assert ch_in == ch_out
        ch_hidden = int(ch_in * ch_hidden_ratio)
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, stride=1, act=act)
        self.conv2 = RepConv(ch_in, ch_hidden, 3, stride=1, act=act)
        self.shortcut = shortcut
    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        if self.shortcut:
            return x + y
        else:
            return y
class SPP(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        k,
        pool_size,
        act='swish',
    ):
        super(SPP, self).__init__()
        self.pool = []
        for i, size in enumerate(pool_size):
            pool = nn.MaxPool2d(kernel_size=size,
                                stride=1,
                                padding=size // 2,
                                ceil_mode=False)
            self.add_module('pool{}'.format(i), pool)
            self.pool.append(pool)
        self.conv = ConvBNAct(ch_in, ch_out, k, act=act)
    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)
        y = self.conv(y)
        return y
class CReToNeXt(nn.Module):
    # CSPStage BLock
    def __init__(self,
                 ch_in=768,
                 ch_out=512,
                 ch_hidden_ratio=1.0,
                 n=round(3 * 1.0),
                 block_fn='BasicBlock_3x3_Reverse',
                 act='swish',
                 spp=False):
        super(CReToNeXt, self).__init__()
        split_ratio = 2
        ch_first = int(ch_out // split_ratio)
        ch_mid = int(ch_out - ch_first)
        self.conv1 = ConvBNAct(ch_in, ch_first, 1, act=act)
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock_3x3_Reverse':
                self.convs.add_module(
                    str(i),
                    BasicBlock_3x3_Reverse(next_ch_in,
                                           ch_hidden_ratio,
                                           ch_mid,
                                           act=act,
                                           shortcut=True))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNAct(ch_mid * n + ch_first, ch_out, 1, act=act)
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        mid_out = [y1]
        for conv in self.convs:
            y2 = conv(y2)
            mid_out.append(y2)
        y = torch.cat(mid_out, axis=1)
        y = self.conv3(y)
        return y
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=groups,
                  bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result





# 论文地址：https://arxiv.org/html/2408.04326v1
# 代码地址：https://github.com/BellyBeauty/MDSAM

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width=4, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False), norm(hidden_dim), nn.Sigmoid()
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                    norm(hidden_dim),
                    nn.Sigmoid(),
                )
            )
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False), norm(in_dim), act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False), norm(in_dim), nn.Sigmoid()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class DetailEnhancement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm, act):
        super().__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding=1, bias=False), norm(img_dim), act()
        )
        self.img_er = MEEM(img_dim, img_dim // 2, 4, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim + img_dim, 32, 3, padding=1, bias=False),
            norm(32),
            act(),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            norm(16),
            act(),
        )

        self.out_conv = nn.Conv2d(16, 1, 1)

        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
        )

    def forward(self, img, feature, b_feature):

        feature = torch.cat([feature, b_feature], dim=1)
        feature = self.feature_upsample(feature)

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature

        out_feature = torch.cat([feature, img_feature], dim=1)
        out_feature = self.fusion_conv(out_feature)
        out = self.out_conv(out_feature)

        return out


class MLFusion(nn.Module):
    def __init__(self, norm, act):
        super().__init__()
        self.fusi_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False), norm(256), act(),
        )

        self.attn_conv = nn.ModuleList()
        for i in range(4):
            self.attn_conv.append(
                nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), norm(256), act(),)
            )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        fusi_feature = torch.cat(feature_list, dim=1).contiguous()
        fusi_feature = self.fusi_conv(fusi_feature)

        for i in range(4):
            x = feature_list[i]
            attn = self.attn_conv[i](x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)

            x = attn * x + x
            feature_list[i] = x

        return feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]


class ModifyPPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(ModifyPPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_dim, reduction_dim, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(
                        reduction_dim,
                        reduction_dim,
                        kernel_size=3,
                        bias=False,
                        groups=reduction_dim,
                    ),
                    nn.GELU(),
                )
            )
        self.features = nn.ModuleList(self.features)
        self.local_conv = nn.Sequential(
            nn.Conv2d(
                in_dim, in_dim, kernel_size=3, padding=1, bias=False, groups=in_dim
            ),
            nn.GELU(),
        )

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for f in self.features:
            out.append(
                F.interpolate(f(x), x_size[2:], mode="bilinear", align_corners=True)
            )
        return torch.cat(out, 1)


class LMSA(nn.Module):
    def __init__(self, in_dim, hidden_dim, patch_num):
        super().__init__()
        self.down_project = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.mppm = ModifyPPM(hidden_dim, hidden_dim // 4, [3, 6, 9, 12])
        self.patch_num = patch_num
        self.up_project = nn.Linear(hidden_dim, in_dim)
        self.down_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1), nn.GELU()
        )

    def forward(self, x):
        down_x = self.down_project(x)
        down_x = self.act(down_x)

        down_x = down_x.permute(0, 3, 1, 2).contiguous()
        down_x = self.mppm(down_x).contiguous()
        down_x = self.down_conv(down_x)
        down_x = down_x.permute(0, 2, 3, 1).contiguous()

        up_x = self.up_project(down_x)
        return x + up_x


if __name__ == "__main__":

    MEEM = MEEM(in_dim=64, hidden_dim=32)
    input = torch.randn(1, 64, 128, 128)
    output = MEEM(input)
    print(input.size())
    print(output.size())

import torch
from torch import nn
from torch.nn import functional as F

# -----------------------------CondConv Begin-----------------------------
class CondAttention(nn.Module):
    def __init__(self, in_planes, K, init_weight=True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Conv2d(in_planes, K, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        if (init_weight):
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.avgpool(x)  # bs,dim,1,1
        att = self.net(att).view(x.shape[0], -1)  # bs,K
        return self.sigmoid(att)


class CondConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, dilation=1, grounps=1, bias=True, K=4,
                 init_weight=True):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = grounps
        self.bias = bias
        self.K = K
        self.init_weight = init_weight
        self.attention = CondAttention(in_planes=in_planes, K=K, init_weight=init_weight)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes // grounps, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_planes), requires_grad=True)
        else:
            self.bias = None

        if self.init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        bs, in_planels, h, w = x.shape
        softmax_att = self.attention(x)  # bs,K
        x = x.view(1, -1, h, w)
        weight = self.weight.view(self.K, -1)  # K,-1
        aggregate_weight = torch.mm(softmax_att, weight).view(bs * self.out_planes, self.in_planes // self.groups,
                                                              self.kernel_size, self.kernel_size)  # bs*out_p,in_p,k,k

        if self.bias is not None:
            bias = self.bias.view(self.K, -1)  # K,out_p
            aggregate_bias = torch.mm(softmax_att, bias).view(-1)  # bs,out_p
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              groups=self.groups * bs, dilation=self.dilation)

        output = output.view(bs, self.out_planes, h, w)
        return output
# -----------------------------CondConv End-----------------------------

if __name__ == '__main__':
    input = torch.randn(2, 32, 64, 64)
    m = CondConv(in_planes=32, out_planes=64, kernel_size=3, stride=1, padding=1, bias=False)
    out = m(input)
    print(out.shape)




# 论文：MogaNet: Multi-order Gated Aggregation Network (ICLR 2024)
# 论文地址：https://arxiv.org/pdf/2211.03295
# Github地址：https://github.com/Westlake-AI/MogaNet
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ["GELU", "ReLU", "SiLU"]
    if act_type == "SiLU":
        return nn.SiLU()
    elif act_type == "ReLU":
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ["BN", "GN", "LN2d", "SyncBN"]
    if norm_type == "GN":
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == "LN2d":
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == "SyncBN":
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm2d(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0.0, requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        kernel_size=3,
        act_type="GELU",
        ffn_drop=0.0,
    ):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1,
        )
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels,
        )
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels, out_channels=embed_dims, kernel_size=1
        )
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1,
            kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True
        )
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(
        self,
        embed_dims,
        dw_dilation=[
            1,
            2,
            3,
        ],
        channel_split=[
            1,
            3,
            4,
        ],
    ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1,
            dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1,
            dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1,
            dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0 : self.embed_dims_0 + self.embed_dims_1, ...]
        )
        x_2 = self.DW_conv2(x_0[:, self.embed_dims - self.embed_dims_2 :, ...])
        x = torch.cat([x_0[:, : self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(
        self,
        embed_dims,
        attn_dw_dilation=[1, 2, 3],
        attn_channel_split=[1, 3, 4],
        attn_act_type="SiLU",
        attn_force_fp32=False,
    ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1
        )

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type="cuda", enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(
        self,
        embed_dims,
        ffn_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        act_type="GELU",
        norm_type="BN",
        init_value=1e-5,
        attn_dw_dilation=[1, 2, 3],
        attn_channel_split=[1, 3, 4],
        attn_act_type="SiLU",
        attn_force_fp32=False,
    ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True
        )
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True
        )

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x

class ConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of PatchEmbed.
            Defaults to 3.
        stride (int): The conv stride of PatchEmbed. Defaults to 2.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class StackConvPatchEmbed(nn.Module):
    """An implementation of Stack Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_type='GELU',
                 norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


if __name__ == "__main__":
    input = torch.randn(3, 64, 32, 32).cuda()
    model = MogaBlock(64).cuda()
    output = model(input)
    print(output.shape)




# 论文名称：SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention
# 论文地址：https://arxiv.org/pdf/2407.05128
# 代码地址：https://github.com/HZAI-ZJNU/SCSA
# 博客地址：TODO


import typing as t
import torch
import torch.nn as nn
from einops import rearrange


class SCSA(nn.Module):
    def __init__(
        self,
        dim: int,
        head_num: int = 8,
        window_size: int = 7,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        down_sample_mode: str = "avg_pool",
        attn_drop_ratio: float = 0.0,
        gate_layer: str = "sigmoid",
    ):
        super(SCSA, self).__init__()  # 调用 nn.Module 的构造函数
        self.dim = dim  # 特征维度
        self.head_num = head_num  # 注意力头数
        self.head_dim = dim // head_num  # 每个头的维度
        self.scaler = self.head_dim**-0.5  # 缩放因子
        self.group_kernel_sizes = group_kernel_sizes  # 分组卷积核大小
        self.window_size = window_size  # 窗口大小
        self.qkv_bias = qkv_bias  # 是否使用偏置
        self.fuse_bn = fuse_bn  # 是否融合批归一化
        self.down_sample_mode = down_sample_mode  # 下采样模式

        assert (
            self.dim % 4 == 0
        ), "The dimension of input feature should be divisible by 4."  # 确保维度可被4整除
        self.group_chans = group_chans = self.dim // 4  # 分组通道数

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2,
            groups=group_chans,
        )
        self.global_dwc_s = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=group_chans,
        )
        self.global_dwc_m = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=group_chans,
        )
        self.global_dwc_l = nn.Conv1d(
            group_chans,
            group_chans,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=group_chans,
        )

        # 注意力门控层
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)  # 水平方向的归一化
        self.norm_w = nn.GroupNorm(4, dim)  # 垂直方向的归一化

        self.conv_d = nn.Identity()  # 直接连接
        self.norm = nn.GroupNorm(1, dim)  # 通道归一化
        # 定义查询、键和值的卷积层
        self.q = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.k = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.v = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim
        )
        self.attn_drop = nn.Dropout(attn_drop_ratio)  # 注意力丢弃层
        self.ca_gate = (
            nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()
        )  # 通道注意力门控

        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化
        else:
            if down_sample_mode == "recombination":
                self.down_func = self.space_to_chans  # 重组合下采样
                # 维度降低
                self.conv_d = nn.Conv2d(
                    in_channels=dim * window_size**2,
                    out_channels=dim,
                    kernel_size=1,
                    bias=False,
                )
            elif down_sample_mode == "avg_pool":
                self.down_func = nn.AvgPool2d(
                    kernel_size=(window_size, window_size), stride=window_size
                )  # 平均池化
            elif down_sample_mode == "max_pool":
                self.down_func = nn.MaxPool2d(
                    kernel_size=(window_size, window_size), stride=window_size
                )  # 最大池化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        """
        # 计算空间注意力优先级
        b, c, h_, w_ = x.size()  # 获取输入的形状
        # (B, C, H)
        x_h = x.mean(dim=3)  # 沿着宽度维度求平均
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(
            x_h, self.group_chans, dim=1
        )  # 拆分通道
        # (B, C, W)
        x_w = x.mean(dim=2)  # 沿着高度维度求平均
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(
            x_w, self.group_chans, dim=1
        )  # 拆分通道

        # 计算水平注意力
        x_h_attn = self.sa_gate(
            self.norm_h(
                torch.cat(
                    (
                        self.local_dwc(l_x_h),
                        self.global_dwc_s(g_x_h_s),
                        self.global_dwc_m(g_x_h_m),
                        self.global_dwc_l(g_x_h_l),
                    ),
                    dim=1,
                )
            )
        )
        x_h_attn = x_h_attn.view(b, c, h_, 1)  # 调整形状

        # 计算垂直注意力
        x_w_attn = self.sa_gate(
            self.norm_w(
                torch.cat(
                    (
                        self.local_dwc(l_x_w),
                        self.global_dwc_s(g_x_w_s),
                        self.global_dwc_m(g_x_w_m),
                        self.global_dwc_l(g_x_w_l),
                    ),
                    dim=1,
                )
            )
        )
        x_w_attn = x_w_attn.view(b, c, 1, w_)  # 调整形状

        # 计算最终的注意力加权
        x = x * x_h_attn * x_w_attn

        # 基于自注意力的通道注意力
        # 减少计算量
        y = self.down_func(x)  # 下采样
        y = self.conv_d(y)  # 维度转换
        _, _, h_, w_ = y.size()  # 获取形状

        # 先归一化，然后重塑 -> (B, H, W, C) -> (B, C, H * W)，并生成 q, k 和 v
        y = self.norm(y)  # 归一化
        q = self.q(y)  # 计算查询
        k = self.k(y)  # 计算键
        v = self.v(y)  # 计算值
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(
            q,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        k = rearrange(
            k,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )
        v = rearrange(
            v,
            "b (head_num head_dim) h w -> b head_num head_dim (h w)",
            head_num=int(self.head_num),
            head_dim=int(self.head_dim),
        )

        # 计算注意力
        attn = q @ k.transpose(-2, -1) * self.scaler  # 点积注意力计算
        attn = self.attn_drop(attn.softmax(dim=-1))  # 应用注意力丢弃
        # (B, head_num, head_dim, N)
        attn = attn @ v  # 加权值
        # (B, C, H_, W_)
        attn = rearrange(
            attn,
            "b head_num head_dim (h w) -> b (head_num head_dim) h w",
            h=int(h_),
            w=int(w_),
        )
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)  # 求平均
        attn = self.ca_gate(attn)  # 应用通道注意力门控
        return attn * x  # 返回加权后的输入


if __name__ == "__main__":
    scsa = SCSA(dim=32, head_num=8, window_size=7)
    input_tensor = torch.rand(1, 32, 256, 256)
    print(input_tensor.shape)
    output_tensor = scsa(input_tensor)
    print(output_tensor.shape)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   Axial.py
@Time      :   2024/05/11 20:23:22
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import torch
from torch import nn
from operator import itemgetter
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states


class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1, y2 = None, None

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=1)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=1)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=1)
            dx = torch.cat([dx1, dx2], dim=1)

        return x, dx


class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=1)


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(
        self,
        blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for (f, g) in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {"f_args": f_args, "g_args": g_args}
        x = torch.cat((x, x), dim=1)
        x = _ReversibleFunction.apply(x, self.blocks, block_kwargs)
        return torch.stack(x.chunk(2, dim=1)).mean(dim=0)


# helper functions


def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for axial_dim in axial_dims:
        last_two_dims = [axial_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


# helper classes


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x)
            x = x + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, **kwargs)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


# axial pos emb


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        self.num_axials = len(shape)

        for i, (axial_dim, axial_dim_index) in enumerate(zip(shape, ax_dim_indexes)):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[axial_dim_index] = axial_dim
            parameter = nn.Parameter(torch.randn(*shape))
            setattr(self, f"param_{i}", parameter)

    def forward(self, x):
        for i in range(self.num_axials):
            x = x + getattr(self, f"param_{i}")
        return x


# attention


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = (
            lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        )
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum("bie,bje->bij", q, k) * (e**-0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum("bij,bje->bie", dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


# axial attention class


class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_dimensions=2,
        heads=8,
        dim_heads=None,
        dim_index=-1,
        sum_axial_out=True,
    ):
        assert (
            dim % heads
        ) == 0, "hidden dimension must be divisible by number of heads"
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = (
            dim_index if dim_index > 0 else (dim_index + self.total_dimensions)
        )

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(
                PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads))
            )

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x):
        assert (
            len(x.shape) == self.total_dimensions
        ), "input tensor does not have the correct number of dimensions"
        assert (
            x.shape[self.dim_index] == self.dim
        ), "input tensor does not have the correct input dimension"

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out


# axial image transformer


class AxialImageTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        dim_heads=None,
        dim_index=1,
        reversible=True,
        axial_pos_emb_shape=None,
    ):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = lambda: nn.Sequential(
            ChanLayerNorm(dim),
            nn.Conv2d(dim, dim * 4, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim * 4, dim, 3, padding=1),
        )

        self.pos_emb = (
            AxialPositionalEmbedding(dim, axial_pos_emb_shape, dim_index)
            if exists(axial_pos_emb_shape)
            else nn.Identity()
        )

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList(
                [
                    PermuteToFrom(
                        permutation, PreNorm(dim, SelfAttention(dim, heads, dim_heads))
                    )
                    for permutation in permutations
                ]
            )
            conv_functions = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attn_functions)
            layers.append(conv_functions)

        execute_type = ReversibleSequence if reversible else Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        return self.layers(x)


if __name__ == "__main__":
    input = torch.rand(3, 64, 32, 32).cuda()
    model = AxialImageTransformer(dim=64, depth=12, reversible=True).cuda()
    output = model(input)
    print(input.size(), output.size())



from functools import partial
import pywt
import pywt.data
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WTConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type="db1",
    ):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [
                _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
                for _ in range(self.wt_levels)
            ]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(
                torch.ones(in_channels, 1, 1, 1), requires_grad=False
            )
            self.do_stride = lambda x_in: F.conv2d(
                x_in,
                self.stride_filter,
                bias=None,
                stride=self.stride,
                groups=in_channels,
            )
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(
                shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4]
            )
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


if __name__ == "__main__":
    input = torch.randn(64, 256, 8, 8).cuda()
    model = WTConv2d(256, 256).cuda()
    output = model(input)
    print(output.shape)





#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File      :   PPA.py
@Time      :   2024/09/11 19:35:07
@Author    :   CSDN迪菲赫尔曼 
@Version   :   1.0
@Reference :   https://blog.csdn.net/weixin_43694096
@Desc      :   None
"""


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(
            torch.randn(output_dim, requires_grad=True)
        )
        self.top_down_transform = torch.nn.parameter.Parameter(
            torch.eye(output_dim), requires_grad=True
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(
            self.prompt[None, ..., None], dim=1
        )  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(
            B, H // P, W // P, self.output_dim
        )  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(
            local_out, size=(H, W), mode="bilinear", align_corners=False
        )
        output = self.conv(local_out)

        return output


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class conv_block(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        norm_type="bn",
        activation=True,
        use_bias=True,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=use_bias,
            groups=groups,
        )

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == "gn":
            self.norm = nn.GroupNorm(
                32 if out_features >= 32 else out_features, out_features
            )
        if self.norm_type == "bn":
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
        super().__init__()

        self.skip = conv_block(
            in_features=in_features,
            out_features=filters,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type="bn",
            activation=False,
        )
        self.c1 = conv_block(
            in_features=in_features,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type="bn",
            activation=True,
        )
        self.c2 = conv_block(
            in_features=filters,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type="bn",
            activation=True,
        )
        self.c3 = conv_block(
            in_features=filters,
            out_features=filters,
            kernel_size=(3, 3),
            padding=(1, 1),
            norm_type="bn",
            activation=True,
        )
        self.sa = SpatialAttentionModule()
        self.cn = ECA(filters)
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        self.bn1 = nn.BatchNorm2d(filters)
        self.drop = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

        self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


if __name__ == "__main__":
    block = PPA(64, 64)
    input = torch.rand(3, 64, 128, 128)
    output = block(input)
    print(input.size())
    print(output.size())


# 代码地址: https://github.com/sunny2109/SAFMN
# 论文地址：https://arxiv.org/pdf/2302.13800
# 博客地址：TODO
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [
                nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
                for i in range(self.n_levels)
            ]
        )

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2**i, w // 2**i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode="nearest")
            else:
                s = self.mfr[i](xc[i])
            out.append(s)

        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out


if __name__ == "__main__":
    input = torch.randn(3, 128, 64, 64)  # 输入b c h w
    block = SAFM(dim=128)
    output = block(input)
    print(output.size())

