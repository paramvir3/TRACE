"""Special functions evaluated exactly in closed form, for speed.

``half_integer_regularized_gamma``: P(n + 1/2, x) = gamma(n + 1/2, x) / Gamma(n + 1/2), n >= 1, the
damping factors of every Gaussian multipole kernel.  ``torch.special.gammainc`` evaluates a
general-order series or continued fraction per element.  For half-integer order the function has
a closed form, which needs one ``exp`` (and one ``erf`` for large x):

    P(1/2, x) = erf(sqrt x)
    P(a + 1, x) = P(a, x) - x^a e^{-x} / Gamma(a + 1)                 (exact recursion)
    P(a, x)     = x^a e^{-x} sum_{k >= 0} x^k / Gamma(a + k + 1)      (exact series, all x)

Numerically:
* x < 3: the series for the highest requested order (32 terms; the remainder is below 1e-24
  relative there), then the recursion run DOWNWARD, P(a - 1) = P(a) + x^(a-1) e^{-x} / Gamma(a),
  which only adds positive terms.  Upward recursion from erf would cancel: at x = 0.5,
  P(9/2) ~ 2e-4 against erf ~ 0.68.
* x >= 3: the recursion run UPWARD from erf(sqrt x).  There P(9/2) >= 0.2, so the amplification
  of erf's rounding is at most ~5.

Each branch sees clamped inputs, so the branch not taken never produces an infinite derivative
(sqrt at 0).  Accuracy against 50-digit mpmath is within 32 eps over x in [0, 1000]
(tests/electrostatics/test_half_integer_gamma.py).  ``torch.special.gammainc`` itself is off by
up to 1.8e-14 relative on the same grid.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import Tensor

_SWITCH = 3.0
_TERMS = 32


def _values(x: Tensor, orders: Tuple[int, ...]) -> Tuple[Tensor, ...]:
    """The closed forms, no graph.  The series runs only on the entries with x < 3."""
    top = max(orders)
    out = {n: torch.empty_like(x) for n in range(1, top + 1)}
    small = x < _SWITCH
    if bool(small.any()):
        xs = x[small]
        a = top + 0.5
        term = torch.full_like(xs, 1.0 / math.gamma(a + 1.0))
        total = term.clone()
        for k in range(1, _TERMS):
            term = term * xs / (a + k)
            total = total + term
        decay = torch.exp(-xs)
        p = torch.pow(xs, a) * decay * total
        out[top][small] = p
        for n in range(top - 1, 0, -1):
            p = p + torch.pow(xs, n + 0.5) * decay / math.gamma(n + 1.5)
            out[n][small] = p
    large = ~small
    if bool(large.any()):
        xl = x[large]
        decay = torch.exp(-xl)
        p = torch.erf(torch.sqrt(xl))
        for n in range(1, top + 1):
            p = p - torch.pow(xl, n - 0.5) * decay / math.gamma(n + 0.5)
            out[n][large] = p
    return tuple(out[n] for n in orders)


def _derivative(x: Tensor, a: float) -> Tensor:
    """dP(a, x)/dx = x^(a - 1) e^(-x) / Gamma(a), written in differentiable ops (double backward)."""
    positive = x > 0
    safe = torch.where(positive, x, torch.ones_like(x))
    # pow and exp of the EXACT input: a few ulp.  exp((a-1) log x - x - lgamma a) would round its
    # O(100) exponent first and lose ~|exponent| eps (measured 108 eps at x = 244).
    value = torch.pow(safe, a - 1.0) * torch.exp(-safe) / math.gamma(a)
    return torch.where(positive, value, torch.zeros_like(x))


class _HalfIntegerGamma(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, *orders: int):
        ctx.orders = orders
        ctx.save_for_backward(x)
        with torch.no_grad():
            return _values(x, orders)

    @staticmethod
    def backward(ctx, *grads):
        (x,) = ctx.saved_tensors
        total = torch.zeros_like(x)
        for n, g in zip(ctx.orders, grads):
            if g is not None:
                total = total + g * _derivative(x, n + 0.5)
        return (total,) + (None,) * len(ctx.orders)


def half_integer_regularized_gamma(x: Tensor, orders: Sequence[int] = (1, 2, 3, 4)) -> Tuple[Tensor, ...]:
    """``(P(n + 1/2, x) for n in orders)``, n >= 1, x >= 0, elementwise.

    Differentiable to any order: the first derivative is the analytic
    x^(a-1) e^(-x) / Gamma(a) (no graph through the series), itself written in
    differentiable operations.
    """
    orders = tuple(int(n) for n in orders)
    if not orders or min(orders) < 1:
        raise ValueError(f"orders must be integers >= 1, got {orders}")
    return _HalfIntegerGamma.apply(x, *orders)
