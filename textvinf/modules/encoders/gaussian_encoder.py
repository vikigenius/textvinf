# -*- coding: utf-8 -*-

from typing import Any, Dict

import torch
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.text_field_embedders.text_field_embedder import (
    TextFieldEmbedder,
)
from overrides import overrides
from pyro.distributions.torch import Independent, Normal
from torch.distributions.kl import kl_divergence
from torch.nn import BatchNorm1d, Linear, Sequential

from textvinf.modules.encoders.variational_encoder import VariationalEncoder


@VariationalEncoder.register('gaussian')
class GaussianEncoder(VariationalEncoder):
    """Gaussian Encoder."""

    def __init__(
        self,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2VecEncoder,
        latent_dim: int,
    ) -> None:
        super().__init__(text_field_embedder, encoder, latent_dim)
        self._latent_to_mean = Sequential(
            Linear(self._encoder.get_output_dim(), self.latent_dim),
            BatchNorm1d(self.latent_dim),
        )
        self._latent_to_logvar = Sequential(
            Linear(self._encoder.get_output_dim(), self.latent_dim),
            BatchNorm1d(self.latent_dim),
        )

    def forward(self, source_tokens: Dict[str, torch.LongTensor]) -> Dict[str, Any]:
        # pylint: disable=arguments-differ
        """Make a forward pass of the encoder."""
        encoder_outs = self.encode(source_tokens)
        encoder_outputs = encoder_outs['encoder_outputs']
        mask = encoder_outs['mask']
        mean = self._latent_to_mean(encoder_outputs)
        logvar = self._latent_to_logvar(encoder_outputs)
        prior = Independent(
            Normal(
                torch.zeros((mean.size(0), self.latent_dim), device=mean.device),
                torch.ones((mean.size(0), self.latent_dim), device=mean.device),
            ),
            1,
        )
        posterior = Independent(Normal(mean, (0.5 * logvar).exp()), 1)
        batch_size = mean.size(0)
        z = posterior.rsample() if self.training else posterior.mean
        kld = kl_divergence(posterior, prior).sum() / batch_size
        self._kl_metric(kld)
        return {
            'prior': prior,
            'posterior': posterior,
            'mask': mask,
            'kl': kld,
            'z': z,
        }

    @overrides
    def get_metrics(self, reset=False):
        """Get encoder metrics."""
        return super().get_metrics(reset=reset)
