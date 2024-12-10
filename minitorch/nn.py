from typing import Tuple, Optional

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

max_reduce = FastOps.reduce(operators.max, float("-inf"))


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    batch, channel, height, width = input.shape
    k1, k2 = kernel
    assert height % k1 == 0
    assert width % k2 == 0
    new_height = height // k1
    new_width = width // k2
    tile_size = k1 * k2
    output = input.contiguous().view(batch, channel, new_height, k1, new_width, k2)
    output = output.permute(0, 1, 2, 4, 3, 5)
    output = output.contiguous().view(batch, channel, new_height, new_width, tile_size)
    return output, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        output.mean(4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to compute the argmax.

    Returns:
    -------
        Tensor: A 1-hot tensor with the same shape as the input, where the maximum values along the specified dimension are 1 and all other values are 0.

    """
    max_values = max_reduce(input, dim)
    one_hot = input == max_values
    return one_hot


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Max forward"""
        if isinstance(dim, Tensor):
            dimI = int(dim._tensor._storage[0])
        else:
            dimI = int(dim)
        b = max_reduce(a, dimI)
        ctx.save_for_backward(a.f.eq_zip(a, b))
        return b

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Max backward"""
        (a,) = ctx.saved_values
        return a * grad_output, 0.0


def max(input: Tensor, dim: Optional[int]) -> Tensor:
    """Compute the max along a dimension.

    Args:
    ----
        input: Tensor
        dim: dimension to compute max

    Returns:
    -------
        Tensor with dimension dim reduced to 1

    """
    if dim is None:
        return Max.apply(input.contiguous().view(input.size), input._ensure_tensor(0))
    else:
        return Max.apply(input, input._ensure_tensor(dim))


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D.

    Args:
    ----
        input: Tensor of size batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    output, new_height, new_width = tile(input, kernel)
    return (
        max(output, 4)
        .contiguous()
        .view(output.shape[0], output.shape[1], new_height, new_width)
    )


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: Tensor
        dim: dimension to compute softmax

    Returns:
    -------
        Tensor with softmax applied to dimension dim

    """
    out = input.exp()
    return out / (out.sum(dim))


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: Tensor
        dim: dimension to compute softmax

    Returns:
    -------
        Tensor with log softmax applied to dimension dim

    """
    max_tensor = max(input, dim)
    shifted_input = input - max_tensor
    sum_exp = shifted_input.exp().sum(dim)
    log_sum_exp = sum_exp.log() + max_tensor
    return input - log_sum_exp


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off.

    Args:
    ----
        input: Tensor
        p: probability of dropout
        ignore: if True, do not apply dropout

    Returns:
    -------
        Tensor with dropout applied

    """
    if ignore:
        return input
    if p == 0.0:
        return input
    if p == 1.0:
        return input.zeros()
    return input * (rand(input.shape, backend=input.backend, requires_grad=False) > p)
