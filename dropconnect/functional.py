from torch import Tensor
from torch import masked_fill
from torch import bernoulli
from torch import full 

def dropconnect(features: Tensor, weight: Tensor, bias: Tensor | None = None, p: float = 0.5) -> Tensor:
    r"""Applies the DropConnect regularization to the input features during a linear transformation.

    DropConnect is a stochastic regularization technique where weights of a neural network
    are randomly set to zero during training, rather than the activations (as in Dropout).
    This function applies DropConnect on a per-sample basis during the matrix multiplication
    step of a linear transformation.

    The operation performed is:
        :math:`y = (x / (1 - p)) @ (W \odot M)^T + b`
    where:
        - :math:`x` is the input feature tensor
        - :math:`W` is the weight matrix
        - :math:`M` is a binary mask sampled from a Bernoulli distribution with probability `p`
        - :math:`b` is an optional bias term

    Args:
        features (Tensor): Input tensor of shape :math:`(N, \text{in\_features})`
        weight (Tensor): Weight tensor of shape :math:`(\text{out\_features}, \text{in\_features})`
        bias (Tensor or None): Optional bias tensor of shape :math:`(\text{out\_features})`
        p (float): Probability of retaining a connection (i.e., not dropping it). Default: 0.5

    Returns:
        Tensor: Output tensor of shape :math:`(N, \text{out\_features})`

    Note:
        This function is intended for use during training. For evaluation, use the standard
        linear transformation without masking.

    Reference:
        Wan, L., Zeiler, M., Zhang, S., Le Cun, Y., & Fergus, R. (2013). *Regularization of Neural Networks using DropConnect*.
        In Proceedings of the 30th International Conference on Machine Learning (pp. 1058–1066). 
        https://proceedings.mlr.press/v28/wan13.html
    """

    mask = bernoulli(full((features.shape[0], weight.shape[0], weight.shape[1]), p)).bool() 
    features = (features / (1 - p)).unsqueeze(-2) @ masked_fill(weight, mask, 0).transpose(-2, -1)
    return (features + bias).squeeze() if bias else features.squeeze()

