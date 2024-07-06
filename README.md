# drop-connect

The paper [DropConnect](http://yann.lecun.com/exdb/publis/pdf/wan-icml-13.pdf) introduces a regularization technique that is similar to Dropout, but instead of dropping out individual units, it drops out individual connections between units. This is done by applying a mask to the weights of the network, which is sampled from a Bernoulli distribution.


## Training

Let $X \in \mathbb{R}^{n \times d}$ a tensor with $n$ examples and $d$ features a $W \in \mathbb{R}^{l \times d}$ a tensor of weights.

For training, a mask matrix $M$ is created from a Bernoulli distribution to mask elements of a weight matrix $W$, using the Hadamard product.

For a single example, the implementation is straightforward, just apply a mask $M$ to a weight tensor $W$. However, according to the paper: "A key component to successfully training with DropConnect is the selection of a different mask for each training example. Selecting a single mask for a subset of training examples, such as a mini-batch of 128 examples, does not regularize the model enough in practice."

Therefore, a mask tensor $M \in \mathbb{R}^{n \times l \times d}$ must be chosen, so the linear layer with DropConnect should be implemented as:

$$ \text{DropConnect}(X, W, M) = \begin{bmatrix}
    \begin{bmatrix}  x_1^1 & x_2^1 & \cdots & x_d^1 \end{bmatrix}
    \left(\begin{bmatrix}
        m_1^{11} & m_2^{11} & \cdots & m_d^{11} \\
        m_1^{12} & m_2^{12} & \cdots & m_d^{12} \\
        \vdots & \vdots & \ddots & \vdots \\
        m_1^{1l} & m_2^{1l} & \cdots & m_d^{1l} \\
    \end{bmatrix} \odot \begin{bmatrix}
        w_1^1 & w_2^1 & \cdots & w_d^1 \\
        w_1^2 & w_2^2 & \cdots & w_d^2 \\
        \vdots & \vdots & \ddots & \vdots \\
        w_1^l & w_2^l & \cdots & w_d^l \\
    \end{bmatrix}
    \right)^T \\
    \\
    \begin{bmatrix}  x_1^2 & x_2^2 & \cdots & x_d^2 \end{bmatrix}
    \left(\begin{bmatrix}
        m_2^{21} & m_2^{21} & \cdots & m_d^{21} \\
        m_1^{22} & m_2^{22} & \cdots & m_d^{22} \\
        \vdots & \vdots & \ddots & \vdots \\
        m_1^{2l} & m_2^{2l} & \cdots & m_d^{2l} \\
    \end{bmatrix} \odot \begin{bmatrix}
        w_1^1 & w_2^1 & \cdots & w_d^1 \\
        w_1^2 & w_2^2 & \cdots & w_d^2 \\
        \vdots & \vdots & \ddots & \vdots \\
        w_1^l & w_2^l & \cdots & w_d^l \\
    \end{bmatrix}
    \right)^T \\
    \\ \vdots   \\
    \\
    \begin{bmatrix}  x_1^n & x_2^n & \cdots & x_d^n \end{bmatrix}
    \left(\begin{bmatrix}
        m_1^{n1} & m_2^{n1} & \cdots & m_d^{n1} \\
        m_1^{n2} & m_2^{n2} & \cdots & m_d^{n2} \\
        \vdots & \vdots & \ddots & \vdots \\
        m_1^{nl} & m_2^{nl} & \cdots & m_d^{nl} \\
    \end{bmatrix} \odot \begin{bmatrix}
        w_1^1 & w_2^1 & \cdots & w_d^1 \\
        w_1^2 & w_2^2 & \cdots & w_d^2 \\
        \vdots & \vdots & \ddots & \vdots \\
        w_1^l & w_2^l & \cdots & w_d^l \\
    \end{bmatrix}
    \right)^T
\end{bmatrix} $$


### Backpropagation

In order to update the weight matrix $W$ in a DropConnect layer, the mask is applied to the gradient to update only those elements that were active in the forward pass. but this is already done by the automatic differentiation in Pytorch, since if $J$ is the gradient coming from the linear operation, the gradient propagated by the Hadamard product with respect to $W$ will be:

$$ J \odot M $$

So there is no need to implement an additional backpropagation operation, and only the Hadamard product already provided by Pytorch is needed. 


## Inference

For inference, the output of the DropConnect layer should be computed as:


$$ \frac{1}{|M|} \sum_M X \cdot (M \odot W) $$

However, this is very computationally expensive, so the same paper proposes an alternative way.
Notice that the output of the linear layer applying the mask $M$ is:


$$ \sum_k X_k^i W_j^k \delta_{ii} \delta^{jj} M_i^j = \sum_k W_k^j X_i^k M_i^j \quad \text{for the $i$, $j$ element}$$

But this is a heavy sum of the elements $M_i^j$ of the Bernoulli distribution that can be approximated by a Gaussian distribution $N(\mu,\sigma^2)$ with mean:

$$ \mu_M[X] = p X W^T $$

and variance:

$$ \sigma^2_M[X] = p(1 - p) (X \odot X) (W^T \odot W^T) $$

where $p$ is the probability of the Bernoulli distribution.

Again, a single distribution is not enough, so a different distribution must be chosen for each example, and the output of the DropConnect layer should, as the paper suggests, be averaged only after the activation function.