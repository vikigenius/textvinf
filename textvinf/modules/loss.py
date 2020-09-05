# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional

import torch
from allennlp.common import FromParams
from allennlp.training.metrics import Average

from textvinf.modules.annealer import LossWeight


def mmd(z_tilde: torch.Tensor, z: torch.Tensor, z_var: float = 1.0):
    r"""Calculate maximum mean discrepancy described in the WAE paper.

    Parameters
    ----------
    z_tilde : Tensor, required
        2D Tensor(batch_size x dimension).
        samples from deterministic non-random encoder Q(Z|X).
    z : Tensor, required
        samples from prior distributions. same shape with z_tilde.
    z_var : float, required
        scalar variance of isotropic gaussian prior P(Z).
    """
    n = z.size(0)
    return im_kernel_sum(z, z, z_var, exclude_diag=True).div(n * (n - 1)) + \
        im_kernel_sum(z_tilde, z_tilde, z_var, exclude_diag=True).div(n * (n - 1)) - \
        im_kernel_sum(z, z_tilde, z_var, exclude_diag=False).div(n * n).mul(2)


def im_kernel_sum(z1: torch.Tensor, z2: torch.Tensor, z_var: float, exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel.

    Parameters
    ----------
    z1 : Tensor, required
        batch of samples from a multivariate gaussian distribution
    z2 : Tensor, required
        batch of samples from another multivariate gaussian distribution
    z_var : flaot, required
        scalar variance of tensor
    exclude_diag : bool, required
        whether to exclude diagonal kernel measures before sum it all.
    """
    z_dim = z1.size(1)
    C = 2 * z_dim * z_var  # noqa: N806

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C / (1e-9 + C + (z11 - z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


class VariationalLoss(FromParams):
    """Class that handles the weighted loss of the encoded distributions.

    This class takes in various annealed weights as parameters and computes
    losses based on that
    """

    def __init__(
        self,
        kl_weight: Optional[LossWeight] = None,
        mmd_weight: Optional[LossWeight] = None,
    ):
        self.kl_weight = kl_weight
        self.mmd_weight = mmd_weight
        self._mmd_metric = Average()

    def __call__(self, encoder_outs: Dict[str, Any]):
        """Calculates loss given the encoder outputs."""
        loss = 0.0
        kl_weight = 0.0
        if self.kl_weight is not None:
            kl_weight = self.kl_weight.get()
            kl_loss = kl_weight * encoder_outs['kl']
            logj_loss = (kl_weight - 1) * encoder_outs.get('logj', 0.0)
            loss += kl_loss + logj_loss
        if self.mmd_weight is not None:
            # Compute mmd
            mmd_weight = self.mmd_weight.get() - kl_weight
            mmd_val = mmd(encoder_outs['z'], encoder_outs['prior'].rsample())
            self._mmd_metric(mmd_val)
            loss += mmd_weight * mmd_val
        return loss

    def step(self) -> None:
        """Increase the step of the annealed weights."""
        if self.kl_weight is not None:
            self.kl_weight.step()
        if self.mmd_weight is not None:
            self.mmd_weight.step()

    def get_metrics(self, reset: bool = True):
        """Collect all metrics and return."""
        metrics = {}
        if self.kl_weight is not None:
            metrics['_klw'] = float(self.kl_weight.get())

        if self.mmd_weight is not None:
            metrics['_mmdw'] = float(self.mmd_weight.get())
            metrics['_mmd'] = float(self._mmd_metric.get_metric(reset=reset))

        return metrics
