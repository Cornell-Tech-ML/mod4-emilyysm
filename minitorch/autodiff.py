from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    add_eps = [v for v in vals]
    sub_eps = [v for v in vals]
    add_eps[arg] = add_eps[arg] + epsilon
    sub_eps[arg] = sub_eps[arg] - epsilon
    delta = f(*add_eps) - f(*sub_eps)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate derivative of the output w.r.t. input variable `x`."""
        ...

    """Accumulate derivative of the output w.r.t. input variable `x`."""

    @property
    def unique_id(self) -> int:
        """Get id."""
        ...

    def is_leaf(self) -> bool:
        """Whether this is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Whether this node is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Get parent nodes."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply chain rule."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    order: List[Variable] = []
    visited = set()

    def dfs(var: Variable) -> None:
        if var.is_constant() or var.unique_id in visited:
            return
        if not var.is_leaf():
            for m in var.parents:
                if not m.is_constant():
                    dfs(m)
        visited.add(var.unique_id)
        order.insert(0, var)

    dfs(variable)
    return order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
        None: No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # topological sort
    top_sort = topological_sort(variable)
    # variable, derivative dictionary
    var_derivs = {}
    var_derivs[variable.unique_id] = deriv
    # for each node
    for var in top_sort:
        curr_deriv = var_derivs[var.unique_id]
        # check if leaf and accumulate
        if var.is_leaf():
            var.accumulate_derivative(curr_deriv)
        else:
            # call backward with d
            for prev, lderiv in var.chain_rule(curr_deriv):
                if prev.is_constant():
                    continue
                var_derivs.setdefault(prev.unique_id, 0.0)
                var_derivs[prev.unique_id] = var_derivs[prev.unique_id] + lderiv


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Get the associated saved values."""
        return self.saved_values
