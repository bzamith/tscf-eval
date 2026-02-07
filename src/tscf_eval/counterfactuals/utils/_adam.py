"""Adam optimizer for NumPy-based gradient optimization.

This module provides a lightweight, stateful Adam optimizer that can be
shared across gradient-based counterfactual explainers (Glacier, LatentCF,
CELS) to avoid code duplication.

Classes
-------
AdamState
    Stateful Adam optimizer that tracks first and second moment estimates
    and produces parameter updates.

References
----------
.. [adam] Kingma, D. P. & Ba, J. (2015). Adam: A Method for Stochastic
   Optimization. ICLR 2015.
"""

from __future__ import annotations

import numpy as np


class AdamState:
    """Stateful Adam optimizer for NumPy arrays.

    Maintains first-moment (mean) and second-moment (uncentered variance)
    estimates of the gradient and produces bias-corrected parameter updates.

    Parameters
    ----------
    shape : tuple of int
        Shape of the parameter array to optimize.
    beta1 : float, optional
        Exponential decay rate for the first moment estimate. Default is 0.9.
    beta2 : float, optional
        Exponential decay rate for the second moment estimate. Default is 0.999.
    epsilon : float, optional
        Small constant for numerical stability in the denominator.
        Default is 1e-8.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.zeros(10)
    >>> adam = AdamState.zeros_like(x)
    >>> gradient = np.random.randn(10)
    >>> update = adam.step(gradient, lr=0.01)
    >>> x = x - update
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """Initialise zero-valued moment estimates and optimizer constants.

        Parameters
        ----------
        shape : tuple of int
            Shape of the parameter array to optimize.
        beta1 : float, default 0.9
            Exponential decay rate for the first moment estimate.
        beta2 : float, default 0.999
            Exponential decay rate for the second moment estimate.
        epsilon : float, default 1e-8
            Small constant for numerical stability in the denominator.
        """
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    @classmethod
    def zeros_like(
        cls,
        x: np.ndarray,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> AdamState:
        """Create an AdamState with moment arrays shaped like ``x``.

        Parameters
        ----------
        x : np.ndarray
            Reference array whose shape is used for moment initialization.
        beta1 : float, optional
            First moment decay rate. Default is 0.9.
        beta2 : float, optional
            Second moment decay rate. Default is 0.999.
        epsilon : float, optional
            Numerical stability constant. Default is 1e-8.

        Returns
        -------
        AdamState
            Initialized optimizer state.
        """
        return cls(shape=x.shape, beta1=beta1, beta2=beta2, epsilon=epsilon)

    def step(self, gradient: np.ndarray, lr: float) -> np.ndarray:
        """Compute the Adam update vector for one optimization step.

        Updates the internal moment estimates and returns the parameter
        update to *subtract* from the current parameters.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient of the loss with respect to the parameters, same
            shape as the parameter array.
        lr : float
            Learning rate (step size).

        Returns
        -------
        np.ndarray
            Update vector. Apply as ``params = params - update``.
        """
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        update: np.ndarray = lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update
