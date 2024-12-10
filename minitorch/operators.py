"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(a: float, b: float) -> float:
    """Multiply 'a' by 'b'."""
    return a * b


def id(x: float) -> float:
    """Return 'x'."""
    return x


def add(a: float, b: float) -> float:
    """Add 'a' and 'b'."""
    return a + b


def neg(x: float) -> float:
    """Negate 'x'."""
    return -x


def lt(a: float, b: float) -> float:
    """Check if 'a' < 'b'."""
    return 1.0 if a < b else 0.0


def eq(a: float, b: float) -> float:
    """Check if 'a' and 'b' are equal."""
    return 1.0 if a == b else 0.0


def max(a: float, b: float) -> float:
    """Return the max between 'a' and 'b'."""
    return a if a > b else b


def is_close(a: float, b: float) -> float:
    """Check if 'a' and 'b' are close in value."""
    return (a - b < 1e-2) and (b - a < 1e-2)


def sigmoid(x: float) -> float:
    """Calculate the sigmoid function for 'x'."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply relu activation function to 'x'."""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Apply log function to 'x'."""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculate the exponential function, i.e., e^x."""
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate the inverse of 'x'."""
    return 1.0 / x


def log_back(a: float, b: float) -> float:
    """Compute the derivative of ln('a') and multiply by 'b'."""
    return b / (a + EPS)


def inv_back(a: float, b: float) -> float:
    """Compute the derivative of the reciprocal of 'a' and multiply by 'b'."""
    return -(1.0 / a**2) * b


def relu_back(a: float, b: float) -> float:
    """Compute the derivative of relu('a') and multiply by 'b'."""
    return b if a > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable.

    Args:
    ----
        fn: Function from one value to another value.

    Returns:
    -------
        A function that takes a list, applies `fn` to each element and returns a new list.

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function.

    Args:
    ----
        fn: Function to apply to each element of lst.

    Returns:
    -------
        Function that takes two equally sized lists and produces a new list applying fn(x,y) on each pair of elements.

    """

    def _zipWith(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(lst1, lst2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function.

    Args:
    ----
        fn: Function to apply to elements in lst.
        start: start value $x_0$

    Returns:
    -------
        Function that takes a list `ls` of elements and computes the reduction.

    """

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list.

    Args:
    ----
        lst: List of values to negate.

    Returns:
    -------
        List of values from lst, negated.

    """
    return map(neg)(lst)


def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        lst1: First list to add.
        lst2: Second list to add.

    Returns:
    -------
        List of values that are the sum of lst1 and lst2.

    """
    return zipWith(add)(lst1, lst2)


def sum(lst: Iterable[float]) -> float:
    """Add all elements in a list.

    Args:
    ----
        lst: List of elements to add.

    Returns:
    -------
        Sum of elements in lst.

    """
    return reduce(add, 0.0)(lst)


def prod(lst: Iterable[float]) -> float:
    """Multiply all elements in a list."""
    return reduce(mul, 1.0)(lst)
