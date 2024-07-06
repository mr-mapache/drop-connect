from typing import Optional
import torch
import math
from torch import Tensor
from torch import Size
from torch.nn import Parameter
from torch.nn import Module
from torch.nn import init
from torch.nn.functional import linear
from torch.distributions.normal import Normal


def drop_connect_training(input: Tensor, mask: Tensor, weight: Tensor, bias: Optional[Tensor] = None, bias_mask: Optional[Tensor] = None) -> Tensor:
    input = (input.unsqueeze(1)@torch.masked_fill(weight, mask, 0).transpose(1,2)).squeeze()
    if bias is not None:
        assert bias_mask is not None , "bias mask is required when bias is provided"
        input = input + torch.masked_fill(bias, bias_mask, 0)
    return input


def drop_connect_inference(input: Tensor, weight: Tensor, p: float, bias: Optional[Tensor] = None) -> Tensor:
    mean = (1-p)*linear(input, weight, bias)
    variance = p*(1-p)*linear(input**2, weight**2)
    return Normal(mean, variance.sqrt()).sample(Size([input.size(0)]))

class DropConnectLinear(Module):

    r"""Applies a linear transformation to the incoming data with DropConnect: :math:`y = xA^T + b`.

    WARNING: The module returns a tensor with different shape in the training and inference modes. Be sure
    of handle this with the DropConnectBatchAverage that averages the output in the batch dimension.

    DropConnect is a generalization of Dropout, where weights are randomly dropped 
    during training. This method was introduced by Wan et al. in the paper 
    "Regularization of Neural Networks using DropConnect" 
    (http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf).

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``.
        p: Probability of an element to be zeroed during training. Default: ``0.5``.
        max_batch_size: Maximum batch size for the input tensor. Default: ``512``.
        device: The desired device of the parameters and buffers.
        dtype: The desired data type of the parameters and buffers.

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: The learnable weights of the module of shape :math:`(\text{out\_features}, \text{in\_features})`. 
                The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where :math:`k = \frac{1}{\text{in\_features}}`.
        bias: The learnable bias of the module of shape :math:`(\text{out\_features})`. 
              If :attr:`bias` is ``True``, the values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{in\_features}}`.
        weight_mask: The mask for the weight tensor of shape :math:`(\text{max\_batch\_size}, \text{out\_features}, \text{in\_features})`.
        bias_mask: The mask for the bias tensor of shape :math:`(\text{max\_batch\_size}, \text{out\_features})`.
        p: Probability of an element to be zeroed during training.
        max_batch_size: Maximum batch size for the input tensor.
    """

    __constants__ = ['in_features', 'out_features', 'p', 'max_batch_size']
    in_features: int
    out_features: int
    weight: Tensor
    weight_mask: Tensor
    p: float

    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.5 , max_batch_size: int = 512, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_batch_size = max_batch_size
        self.p = p
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_mask = Parameter(
            data=torch.bernoulli(torch.zeros(max_batch_size, out_features, in_features, dtype=torch.bool, device=device), p),
            requires_grad=False)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_mask = Parameter(
                data=torch.bernoulli(torch.zeros(max_batch_size, out_features, dtype=torch.bool, device=device), p),
                requires_grad=False)

        else:
            self.register_parameter('bias', None)
            self.register_parameter('bias_mask', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return drop_connect_training(input, self.weight_mask[:input.size(0)], self.weight, self.bias, self.bias_mask[:input.size(0)])
        else:
            return drop_connect_inference(input, self.weight, self.p, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, p={self.p}, max_batch_size={self.max_batch_size}'


def batch_average(input: Tensor) -> Tensor:
    return input.mean(dim=0)

class DropConnectBatchAverage(Module):

    r"""Applies batch averaging during inference.

    This module averages the input tensor across the batch dimension during inference. 
    During training, it simply returns the input tensor as is.

    This is useful when using DropConnect to ensure that the output is averaged 
    over multiple runs, which can help in stabilizing the training process.

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size and :math:`*` represents any number of additional dimensions.
        - Output: :math:`(*)` where the output is averaged across the batch dimension.

    Examples::

        >>> m = DropConnectBatchAverage()
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([20])
    """

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return input
        else:
            return batch_average(input)