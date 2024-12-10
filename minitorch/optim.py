from typing import Sequence

from .module import Parameter
from .scalar import Scalar


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Set gradients and derivatives to None."""
        for param in self.parameters:
            if param.value is None:
                continue
            if hasattr(param.value, "derivative"):
                if param.value.derivative is not None:
                    param.value.derivative = None
            if hasattr(param.value, "grad"):
                if param.value.grad is not None:
                    param.value.grad = None

    def step(self) -> None:
        """Take step in gradient/derivative direction."""
        for param in self.parameters:
            if param.value is None:
                continue
            if hasattr(param.value, "derivative"):
                if param.value.derivative is not None:
                    param.update(
                        Scalar(param.value.data - self.lr * param.value.derivative)
                    )
            elif hasattr(param.value, "grad"):
                if param.value.grad is not None:
                    param.update(param.value - self.lr * param.value.grad)
