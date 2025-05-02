from torch import Tensor 
from torch.nn import Linear 
from torch.nn.functional import linear 
from dropconnect.functional import dropconnect

class Dropconnect(Linear):   
    r"""A linear transformation layer with DropConnect regularization applied during training.

    DropConnect is a form of regularization similar to Dropout, but instead of randomly zeroing 
    elements of the input tensor, it randomly drops (zeros out) elements of the weight matrix during 
    training. This helps prevent co-adaptation of features and improves generalization.

    This module inherits from `torch.nn.Linear` and overrides the `forward` method to apply 
    DropConnect only during training. During evaluation, it defaults to a standard linear transformation.

    Args:
        input_features (int): Size of each input sample.
        output_features (int): Size of each output sample.
        bias (bool, optional): If set to `True`, the layer will learn an additive bias. Default: `True`.
        p (float, optional): Probability of retaining a weight (i.e., not dropping it). Default: `0.5`.

    Shape:
        - Input: :math:`(N, \text{input\_features})`
        - Output: :math:`(N, \text{output\_features})`

    Example::
        >>> layer = Dropconnect(128, 64, p=0.5)
        >>> x = torch.randn(32, 128)
        >>> output = layer(x)

    Note:
        During evaluation (`model.eval()`), this layer behaves as a standard `nn.Linear` module.

    Reference:
        Wan, L., Zeiler, M., Zhang, S., Le Cun, Y., & Fergus, R. (2013). 
        *Regularization of Neural Networks using DropConnect*. 
        In Proceedings of the 30th International Conference on Machine Learning (pp. 1058–1066). 
        https://proceedings.mlr.press/v28/wan13.html
    """
 
    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.5):
        super().__init__(in_features, out_features, bias) 
        self.p = p

    def forward(self, features: Tensor) -> Tensor: 
        return dropconnect(features, self.weight, self.bias, self.p) if self.training else linear(features, self.weight, self.bias)