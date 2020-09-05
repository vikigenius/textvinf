# -*- coding: utf-8 -*-

from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp.nn import util as nn_util
from overrides import overrides
from pyro.distributions import Normal

from textvinf.modules.decoders import Decoder
from textvinf.modules.encoders import VariationalEncoder
from textvinf.modules.flows import IdentityNormalizingFlow, NormalizingFlow
from textvinf.modules.loss import VariationalLoss


@Model.register('lvm')
class LVM(Model):
    """A latent variable model.

    This ``LVM`` class is a :class:`Model` which implements a simple LVM as first described
    in https://arxiv.org/pdf/1711.01558 (Tolstikhin et al., 2018), but modifies the posterior using
    normalizing flows.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    variational_encoder : ``VariationalEncoder``, required
        The encoder model of which to pass the source tokens
    variational_loss : ``VariationalLoss``, required
        The variational loss that contains information about weights and loss components
    decoder : ``Model``, required
        The variational decoder model of which to pass the the latent variable
    flow : ``NormalizingFlow``, optional (default=``IdentityNormalizingFlow``)
        The normalizing flow to be applied
    latent_dim : ``int``, required
        The dimention of the latent, z vector. This is not necessarily the same size as the encoder
        output dim
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        variational_encoder: VariationalEncoder,
        variational_loss: VariationalLoss,
        decoder: Decoder,
        flow: NormalizingFlow = IdentityNormalizingFlow(),  # noqa: WPS404
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:
        super().__init__(vocab)

        self._encoder = variational_encoder
        self._decoder = decoder

        self._latent_dim = variational_encoder.latent_dim

        self._vloss = variational_loss

        self._flow = flow

        initializer(self)

    def encode(self, source_tokens: Dict[str, torch.LongTensor]):
        """Performs encoding tasks and adds the encoding loss to dict."""
        encoder_outs = self._encoder(source_tokens)
        encoder_outs = self._flow(encoder_outs)
        eloss = self._vloss(encoder_outs)
        if self.training:
            self._vloss.step()
        encoder_outs['loss'] = eloss
        return encoder_outs

    @overrides
    def forward(
        self,
        source_tokens: Dict[str, torch.LongTensor],
        target_tokens: Optional[Dict[str, torch.LongTensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Make forward pass for both training/validation/test time."""
        encoder_outs = self.encode(source_tokens)
        decoder_outs = self._decoder(encoder_outs, target_tokens)

        return {
            'latent': encoder_outs['z'],
            'predictions': decoder_outs['predictions'],
            'loss': encoder_outs['loss'] + decoder_outs['loss'],
        }

    @overrides
    def make_output_human_readable(self, output_dict: Dict[str, torch.Tensor]):
        """Return the decoded values."""
        return self._decoder.post_process(output_dict)

    def generate(self, num_to_sample: int = 1):
        """Generate samples from prior."""
        cuda_device = self._get_prediction_device()
        prior_mean = nn_util.move_to_device(
            torch.zeros((num_to_sample, self._latent_dim)),
            cuda_device,
        )
        prior_stddev = torch.ones_like(prior_mean)
        prior = Normal(prior_mean, prior_stddev)
        latent = prior.sample()
        generated = self._decoder.generate(latent)

        return self.make_output_human_readable(generated)

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """Retreive all metrics."""
        all_metrics: Dict[str, float] = {}
        all_metrics.update(self._decoder.get_metrics(reset=reset))
        all_metrics.update(self._encoder.get_metrics(reset=reset))
        all_metrics.update(self._flow.get_metrics(reset=reset))
        all_metrics.update(self._vloss.get_metrics(reset=reset))
        return all_metrics
