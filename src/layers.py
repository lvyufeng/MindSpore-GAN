import math
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import initializer, HeUniform, Uniform, Normal, _calculate_fan_in_and_fan_out
from mindspore.ops.operations.image_ops import ResizeBilinearV2, ResizeLinear1D
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops import constexpr

class _Dropout2d(nn.Cell):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.dropout2d = ops.Dropout2D(keep_prob)

    def construct(self, x):
        return self.dropout2d(x)

    def bprop(self, x, out, dout):
        _, mask = out
        dy, _ = dout
        if self.keep_prob != 0:
            dy = dy * (1 / self.keep_prob)
        dy = mask.astype(mindspore.float32) * dy
        return (dy.astype(x.dtype),)


class Dropout2d(nn.Cell):
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"For '{self.cls_name}', the 'p' must be a number in range [0, 1], "
                             f"but got {p}.")
        self.keep_prob = 1 - p
        self.dropout2d = _Dropout2d(self.keep_prob)

    def construct(self, x):
        if not self.training:
            return x

        if self.keep_prob == 1:
            return x

        out, _ = self.dropout2d(x)
        return out


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, pad_mode, padding, dilation, group, has_bias, weight_init='normal', bias_init='zeros')
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        #self.weight = Parameter(initializer(HeUniform(math.sqrt(5)), self.weight.shape), name='weight')
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


class Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_size, use_one_hot=False, embedding_table='normal', dtype=mindspore.float32, padding_idx=None):
        if embedding_table == 'normal':
            embedding_table = Normal(1.0)
        super().__init__(vocab_size, embedding_size, use_one_hot, embedding_table, dtype, padding_idx)


class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=has_bias, activation=activation)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.weight.set_data(initializer(HeUniform(math.sqrt(5)), self.weight.shape))
        if self.has_bias:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight.shape)
            bound = 1 / math.sqrt(fan_in)
            self.bias.set_data(initializer(Uniform(bound), [self.out_channels]))


@constexpr
def _check_scale_factor(shape, scale_factor):
    if isinstance(scale_factor, tuple) and len(scale_factor) != len(shape[2:]):
        raise ValueError(f"the number of 'scale_fator' must match to inputs.shape[2:], "
                         f"but get scale_factor={scale_factor}, inputs.shape[2:]={shape[2:]}")


def _interpolate_output_shape(shape, scales, sizes, mode):
    """calculate output shape"""
    if sizes is not None:
        if mode == "nearest":
            return sizes
        return Tensor(sizes)

    ret = ()        
    for i in range(len(shape[2:])):
        if isinstance(scales, float):
            out_i = int(scales * shape[i+2])
        else:
            out_i = int(scales[i] * shape[i+2])
        ret = ret + (out_i,)
    if mode == "nearest":
        return ret
    return Tensor(ret)


class Upsample(nn.Cell):
    def __init__(self, size = None, scale_factor = None,
                 mode: str = 'nearest', align_corners = False):
        super().__init__()
        if mode not in ['nearest', 'linear', 'bilinear']:
            raise ValueError(f'do not support mode :{mode}.')
        if size and scale_factor:
            raise ValueError(f"can not set 'size' and 'scale_fator' at the same time.")
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def construct(self, inputs):
        inputs_shape = inputs.shape
        _check_scale_factor(inputs_shape, self.scale_factor)
        sizes = _interpolate_output_shape(inputs_shape, self.scale_factor, self.size, self.mode)
        if self.mode == 'nearest':
            interpolate = _get_cache_prim(ops.ResizeNearestNeighbor)(sizes, self.align_corners)
            return interpolate(inputs)
        elif self.mode == 'linear':
            interpolate = _get_cache_prim(ResizeLinear1D)('align_corners' if self.align_corners else 'half_pixel')
            return interpolate(inputs, sizes)
        elif self.mode == 'bilinear':
            interpolate = _get_cache_prim(ResizeBilinearV2)(self.align_corners, True if self.align_corners==False else False)
            return interpolate(inputs, sizes)
        return inputs
