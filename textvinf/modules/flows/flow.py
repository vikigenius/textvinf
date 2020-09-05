# -*- coding: utf-8 -*-

from typing import Dict, Union

import torch
from allennlp.common.registrable import Registrable
from allennlp.training.metrics import Average
from pyro.distributions import Distribution
from pyro.distributions.torch import TransformedDistribution
from pyro.distributions.transforms import Planar
from torch import nn
from torch.distributions.transforms import ComposeTransform

from textvinf.utils.math_utils import flow_transform_kl


class NormalizingFlow(nn.Module, Registrable):
    """NormalizingFlow implementation."""

    default_implementation = 'identity'

    def __init__(self, num_flows: int = 1):
        """Initialize the flow."""
        super().__init__()  # type: ignore
        self.num_flows = num_flows
        self._kl_metric = Average()
        self._logj_metric = Average()

    def transform_sample(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Given a sample from the original posterior it applies the transform."""
        raise NotImplementedError

    def forward(self, encoder_outs: Dict[str, Union[torch.Tensor, Distribution]]):
        """Forward pass implementation."""
        raise NotImplementedError

    def get_metrics(self, reset: bool = False):
        """Get normalizing flow metrics."""
        all_metrics: Dict[str, float] = {}
        all_metrics.update({'_logJ': float(self._logj_metric.get_metric(reset=reset))})
        all_metrics.update({'_flow_kl': float(self._kl_metric.get_metric(reset=reset))})
        return all_metrics


@NormalizingFlow.register('identity')
class IdentityNormalizingFlow(NormalizingFlow):
    """Identity Normalizing flow."""

    def forward(self, encoder_outs: Dict[str, Union[torch.Tensor, Distribution]]):
        """Returns the encoder_outs as is."""
        return encoder_outs


@NormalizingFlow.register('planar')
class PlanarNormalizingFlow(NormalizingFlow):
    """Planar Normalizing Flows."""

    def __init__(self, input_dim: int, num_flows: int = 1):
        """Initialize the flow."""
        super().__init__(num_flows)
        self.input_dim = input_dim
        transforms = [Planar(self.input_dim) for _ in range(self.num_flows)]
        self.transforms = nn.ModuleList(transforms)
        self.transform = ComposeTransform(self.transforms)  # type: ignore

    def forward(self, encoder_outs: Dict[str, Union[torch.Tensor, Distribution]]):
        """Forward pass implementation."""
        # TODO Move the old posterior, z and kl to z0 and kl0 before overwriting them
        z0 = encoder_outs['z']
        encoder_outs['qz0'] = encoder_outs['posterior']
        encoder_outs['z'] = self.transform(z0)
        logj = self.transform.log_abs_det_jacobian(z0, encoder_outs['z'])
        encoder_outs['kl'] = flow_transform_kl(
            encoder_outs['posterior'],
            encoder_outs['prior'],
            encoder_outs['z'],
            logj,
        )
        flow_posterior = TransformedDistribution(encoder_outs['posterior'], self.transform)
        encoder_outs['flow_posterior'] = flow_posterior
        encoder_outs['logj'] = logj.mean()
        self._kl_metric(encoder_outs['kl'])
        self._logj_metric(encoder_outs['logj'])
        return encoder_outs
