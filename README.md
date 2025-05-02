# Archieved

This implementation has issues. I will be creating an installable package with a better implementation and documentation soon.

# Drop Connect

The paper [DropConnectPapper](https://proceedings.mlr.press/v28/wan13.html) introduces a regularization technique that is similar to Dropout, but instead of dropping out individual units, it drops out individual connections between units. This is done by applying a mask to the weights of the network, which is sampled from a Bernoulli distribution. 

![DropConnectImage](/dropconnect.png)


### Installing

```bash
pip install dropconnect
```

### Usage

```python
from torch import Tensor
from dropconnect import Dropconnect

layer = Dropconnect(in_features=5, out_features=10, bias=True, p=0.5)
input = Tensor([[1,2,3,4,5],[2,3,4,5,6]])
output = layer(input)
print(output) # Can be used just like a drop-in replacement for linear layer.
```

### Training

Let $X \in \mathbb{R}^{n \times d}$ a tensor with $n$ examples and $d$ features a $W \in \mathbb{R}^{l \times d}$ a tensor of weights.

For training, a mask matrix $M$ is created from a Bernoulli distribution to mask elements of the weight matrix $W$ , using the Hadamard product, in order to drop neuron connections instead of turning off neurons like in dropout

For a single example, the implementation is straightforward, just apply a mask $M$ to a weight tensor $W$. However, according to the paper: "A key component to successfully training with DropConnect is the selection of a different mask for each training example. Selecting a single mask for a subset of training examples, such as a mini-batch of 128 examples, does not regularize the model enough in practice."

Therefore, a mask tensor $M \in \mathbb{R}^{n \times l \times d}$ must be chosen, so the linear layer with DropConnect should be implemented as:


$$ 
\text{DropConnect}(X, W, M) = \begin{bmatrix}
    \frac{1}{1-p}\begin{bmatrix} x^1{}_1 & x^1{}_2 & \cdots & x^1{}_d \end{bmatrix}
    \left(\begin{bmatrix}
        m^{11}{}_1 & m^{11}{}_2 & \cdots & m^{11}{}_l \\
        m^{12}{}_1 & m^{12}{}_2 & \cdots & m^{12}{}_l \\
        \vdots & \vdots & \ddots & \vdots \\
        m^{1d}{}_1 & m^{1d}{}_2 & \cdots & m^{1d}{}_l \\
    \end{bmatrix} \odot \begin{bmatrix}
        w^1{}_1 & w^1{}_2 & \cdots & w^1{}_l \\
        w^2{}_1 & w^2{}_2 & \cdots & w^2{}_l \\
        \vdots & \vdots & \ddots & \vdots \\
        w^d{}_1 & w^d{}_2 & \cdots & w^d{}_l \\
    \end{bmatrix}
    \right) \\
    \\
    \\
    \vdots
    \\
    \\
    \frac{1}{1-p}\begin{bmatrix}  x^n{}_1 & x^n{}_2 & \cdots & x^n{}_d \end{bmatrix}
    \left(\begin{bmatrix}
        m^{n1}{}_1 & m^{n1}{}_2 & \cdots & m^{n1}{}_l \\
        m^{n2}{}_1 & m^{n2}{}_2 & \cdots & m^{n2}{}_l \\
        \vdots & \vdots & \ddots & \vdots \\
        m^{nd}{}_1 & m^{nd}{}_2 & \cdots & m^{nd}{}_l \\
    \end{bmatrix} \odot \begin{bmatrix}
        w^1{}_1 & w^1{}_2 & \cdots & w^1{}_l \\
        w^2{}_1 & w^2{}_2 & \cdots & w^2{}_l \\
        \vdots & \vdots & \ddots & \vdots \\
        w^d{}_1 & w^d{}_2 & \cdots & w^d{}_l \\
    \end{bmatrix}
    \right) \\
\end{bmatrix} 
$$

#### Backpropagation

In order to update the weight matrix $W$ in a DropConnect layer, the mask is applied to the gradient to update only those elements that were active in the forward pass. but this is already done by the automatic differentiation in Pytorch, since if $J$ is the gradient coming from the linear operation, the gradient propagated by the Hadamard product with respect to $W$ will be:

$$ J \odot M $$

So there is no need to implement an additional backpropagation operation, and only the Hadamard product already provided by Pytorch is needed.  
