# -*- coding: utf-8 -*-
import sys

import numpy as np
from allennlp.common import Registrable


class LossWeight(Registrable):
    """Abstract class for loss weights.

    Whenever the loss function is composed of more then a single term weighting is probable.
    Use children of this class for different constant/annealed weights
    """

    def __init__(self, initial_weight: float) -> None:
        self._weight = initial_weight

    def get(self) -> float:
        """Get the current weight."""
        return self._weight

    def step(self) -> None:
        """Step and update wieght."""
        raise NotImplementedError

    def __lt__(self, other):
        """Less than operator."""
        if isinstance(other, LossWeight):
            return self._weight < other._weight  # noqa: WPS437
        return self._weight < other


@LossWeight.register('constant_weight')
class ConstantWeight(LossWeight):
    """Constant weight scalar."""

    def step(self) -> None:
        """Nothing to do."""


class AnnealedWeight(LossWeight):
    """LossWeight that can be annealed."""

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        warmup: int = 0,
        early_stop_iter: int = sys.maxsize,
    ) -> None:
        super().__init__(min_weight)
        self.iteration = 0
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.warmup = warmup
        self.early_stop_iter = early_stop_iter

    def step(self):
        """Step and update wieght."""
        weight = self._get_weight() if self.iteration > self.warmup else 0.0
        if self.iteration < self.early_stop_iter:
            self._weight = weight
        self._weight = min(self._weight, self.max_weight)
        self._weight = max(self._weight, self.min_weight)
        self.iteration += 1

    def _get_weight(self) -> float:
        raise NotImplementedError


@LossWeight.register('linear_annealed')
class LinearAnnealedWeight(AnnealedWeight):
    """Linearly nneals weights."""

    def __init__(
        self,
        slope: float,
        intercept: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        warmup: int = 0,
        early_stop_iter: int = sys.maxsize,
    ) -> None:
        super().__init__(min_weight, max_weight, warmup, early_stop_iter)
        self.slope = slope
        self.intercept = intercept

    def _get_weight(self):
        return self.slope * self.iteration + self.intercept


@LossWeight.register('tanh_annealed')
class TanhAnnealedWeight(AnnealedWeight):
    """Tanh annealing schedule."""

    def __init__(
        self,
        slope: float,
        margin: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        warmup: int = 0,
        early_stop_iter: int = sys.maxsize,
    ) -> None:
        super().__init__(min_weight, max_weight, warmup, early_stop_iter)
        self.slope = slope
        self.margin = margin

    def _get_weight(self):
        return 0.5 * (np.tanh(self.slope * (self.iteration - self.margin)) + 1)


@LossWeight.register('sigmoid_annealed')
class SigmoidAnnealedWeight(AnnealedWeight):
    """Sigmoid annealing schedule."""

    def __init__(
        self,
        slope: float,
        margin: float,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        warmup: int = 0,
        early_stop_iter: int = sys.maxsize,
    ) -> None:
        super().__init__(min_weight, max_weight, warmup, early_stop_iter)
        self.slope = slope
        self.margin = margin

    def _get_weight(self):
        return 1 / (1 + np.exp(-self.slope * (self.iteration - self.margin)))
