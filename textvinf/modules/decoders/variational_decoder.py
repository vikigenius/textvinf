# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional, Tuple

import numpy
import torch
from allennlp.data import TextFieldTensors
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn.util import (
    get_lengths_from_binary_sequence_mask,
    get_text_field_mask,
    sequence_cross_entropy_with_logits,
)
from overrides import overrides
from torch.nn import functional as torch_f
from torch.nn.modules import Dropout, Linear

from textvinf.modules.decoders.decoder import Decoder

DecoderState = Optional[Tuple[torch.Tensor, torch.Tensor]]

TOL = 1e-13


# TODO: Refactor this to match allennlp SeqDecoder api
# TODO: cleanup target embedder


@Decoder.register('variational_decoder')
class VariationalDecoder(Decoder):
    """A Variational Decoder is decoder that uses variational inferenc.

    This ``VariationalDecoder`` Trains a Variational Decoder given the latent variable for
    the Language Modeling task

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        Vocabulary containing source and target vocabularies. They may be under the same namespace
        (`tokens`) or the target tokens can have a different namespace, in which case it needs to
        be specified as `target_namespace`.
    target_embedder : ``TextFieldEmbedder``, required
        Embedder for target side sequences
    rnn : ``Seq2SeqEncoder``, required
        The decoder of the "encoder/decoder" model
    latent_dim : ``int``, required
        The dimention of the latent, z vector. This is not necessarily the same size as the encoder
        output dim
    dropout_p : ``float``, optional (default = 0.5)
        This scalar is used to twice. once as a scalar for the dropout to the input embeddings
        for the decoder. Imitating word-dropout, and once as a dropout layer after the decoder RNN.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        rnn: Seq2SeqEncoder,
        latent_dim: int,
        target_embedder: TextFieldEmbedder,
        target_namespace: str = 'tokens',
        dropout_p: float = 0.5,
    ) -> None:
        super().__init__(vocab)
        self._target_embedder = target_embedder
        self._target_namespace = target_namespace
        self.rnn = rnn
        self.dec_num_layers = self.rnn._module.num_layers  # noqa: WPS437
        self.dec_hidden = self.rnn._module.hidden_size  # noqa: WPS437

        self._latent_to_dec_hidden = Linear(latent_dim, self.dec_hidden)
        self._dec_linear = Linear(self.rnn.get_output_dim(), self.vocab.get_vocab_size())

        self._emb_dropout = Dropout(dropout_p)
        self._dec_dropout = Dropout(dropout_p)

    def forward(
        self,
        encoder_outs: Dict[str, Any],
        target_tokens: Optional[TextFieldTensors] = None,
    ) -> Dict[str, torch.Tensor]:
        """Implementation of the forward pass.

        Make a forward pass of the decoder, given the latent vector and the sent as references.
        Notice explanation in simple_seq2seq.py for creating relavant targets and mask

        Notice the indexing here, contrast with indexing in _get_reconstruction_loss
        """
        # TODO: Right now we are always using teacher forcing if target tokens are available.
        # Change this to use scheduled sampling
        z = encoder_outs['z']
        batch_size = z.size(0)
        if target_tokens is not None:
            target_mask = get_text_field_mask(target_tokens)
            relevant_targets = {'tokens': {'tokens': target_tokens['tokens']['tokens'][:, :-1]}}
            relevant_mask = target_mask[:, :-1]

            # num_tokens should technically be target_mask[:, 1:] but the lengths turn out the same
            # So we make use of this precomputed slice
            num_tokens = (get_lengths_from_binary_sequence_mask(relevant_mask) - 1).sum()
            embeddings, state = self._prepare_decoder(z, relevant_targets)
            logits = self._run_decoder(embeddings, relevant_mask, state, z)

            class_probabilities = torch_f.softmax(logits, 2)
            _, best_predictions = torch.max(class_probabilities, 2)

            # Notice that we are not using relevant targets here
            loss = self._get_reconstruction_loss(logits, target_tokens['tokens']['tokens'], target_mask)
            output_dict = {'logits': logits, 'predictions': best_predictions, 'loss': loss}
            if not self.training:
                self._bleu(output_dict['predictions'], target_tokens['tokens']['tokens'])
        else:
            output_dict = self.generate(z, target_tokens)
            # This computation of num_tokens is technically wrong, but we don't really care
            # about computing this kind of PPL for this case so we leave it here
            # TODO verify -2 because both start and end ?
            num_tokens = (output_dict['predictions'].size(1) - 2) * batch_size

        nll = output_dict['loss'] + encoder_outs['kl']
        self._nll(nll)
        self._ppl(nll * batch_size, num_tokens)
        return output_dict

    def _prepare_decoder(
        self, latent: torch.Tensor,
        relevant_targets: TextFieldTensors,
    ) -> Tuple[torch.Tensor, DecoderState]:
        embeddings = self._target_embedder(relevant_targets)
        embeddings = self._emb_dropout(embeddings)
        h0 = embeddings.new_zeros(
            self.dec_num_layers, embeddings.size(0), self.dec_hidden,
        )
        c0 = embeddings.new_zeros(
            self.dec_num_layers, embeddings.size(0), self.dec_hidden,
        )
        h0[-1] = self._latent_to_dec_hidden(latent)
        state = (h0, c0)
        return embeddings, state

    def _run_decoder(
        self,
        embeddings: torch.Tensor,
        relevant_mask: torch.LongTensor,
        state: DecoderState,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        expanded_latent = latent.unsqueeze(1).expand(
            embeddings.size(0), embeddings.size(1), latent.size(1),
        )
        dec_input = torch.cat([embeddings, expanded_latent], 2)
        decoder_out = self.rnn(dec_input, relevant_mask, state)
        decoder_out = self._dec_dropout(
            decoder_out.contiguous().view(embeddings.size(0) * embeddings.size(1), -1),
        )
        logits = self._dec_linear(decoder_out)
        return logits.view(embeddings.size(0), embeddings.size(1), -1)

    def _get_reconstruction_loss(
        self, logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        relevant_targets = targets[:, 1:].contiguous()
        relevant_mask = target_mask[:, 1:].contiguous()
        reconstruction_loss_per_token = sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, average=None,
        )

        # In VAE, we want both loss terms to be comparable, for this reason, we compare kld to each
        # predicted token NLL. For this reason we return to the following mean.
        return (
            reconstruction_loss_per_token * (relevant_mask.sum(1).float() + TOL)
        ).mean()

    def get_reconstruction_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ):
        relevant_targets = targets[:, 1:].contiguous()
        # relevant_mask = target_mask[:, 1:].contiguous()
        batch_sum_loss = torch_f.cross_entropy(
            logits.view(-1, logits.size(2)),
            relevant_targets.view(-1),
            ignore_index=0,
            reduction='sum',
        )
        return batch_sum_loss / logits.size(0)

    def generate(
        self,
        latent: torch.Tensor,
        target_tokens: Optional[TextFieldTensors] = None,
        reset_states=True,
        max_len: int = 30,
    ) -> Dict[str, Any]:
        batch_size, _ = latent.size()
        if reset_states:
            self.rnn.reset_states()
        last_genereation = latent.new_full(
            torch.Size((batch_size, 1)),
            fill_value=self._start_index,
            dtype=torch.long,
        )
        h0 = latent.new_zeros(self.dec_num_layers, batch_size, self.dec_hidden)
        c0 = latent.new_zeros(self.dec_num_layers, batch_size, self.dec_hidden)
        h0[-1] = self._latent_to_dec_hidden(latent)
        # We are decoding step by step. So we are using a stateful decoder
        self.rnn.stateful = True
        restoration_indices = torch.arange(batch_size)
        restoration_indices = restoration_indices.to(h0.device)
        self.rnn._update_states((h0, c0), restoration_indices)  # noqa: WPS437
        generations = [last_genereation]
        all_logits = []
        for _ in range(max_len):
            embeddings = self._target_embedder({'tokens': {'tokens': last_genereation}})
            mask = get_text_field_mask({'tokens': {'tokens': last_genereation}})
            logits = self._run_decoder(embeddings, mask, None, latent)
            class_probabilities = torch_f.softmax(logits, 2)
            _, last_genereation = torch.max(class_probabilities, 2)
            generations.append(last_genereation)
            all_logits.append(logits)

        self.rnn.stateful = False

        output_dict = {'logits': all_logits, 'predictions': torch.cat(generations, 1)}
        if target_tokens is not None:
            target_mask = get_text_field_mask(target_tokens)
            output_dict['loss'] = self._get_reconstruction_loss(
                torch.cat(all_logits[:-1], 1),
                target_tokens['tokens']['tokens'],
                target_mask,
            )
        return output_dict

    @overrides
    def post_process(self, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Executed as post processing for decoder.

        This method trims the output predictions to the first end symbol, replaces indices with
        corresponding tokens, and adds a field called ``predicted_tokens`` to the ``output_dict``.
        """
        predicted_indices = output_dict['predictions']
        if not isinstance(predicted_indices, numpy.ndarray):
            predicted_indices = predicted_indices.detach().cpu().numpy()
        all_predicted_sents = []
        for indices in predicted_indices:
            # Beam search gives us the top k results for each source sentence in the batch
            # but we just want the single best.
            if len(indices.shape) > 1:
                indices = indices[0]
            indices = list(indices)
            # Collect indices till the first end_symbol
            if self._end_index in indices:
                indices = indices[: indices.index(self._end_index)]
            predicted_tokens = [
                self.vocab.get_token_from_index(x, namespace=self._target_namespace)
                for x in indices
                if x not in {self._start_index, self._end_index, self._pad_index}
            ]
            all_predicted_sents.append(' '.join(predicted_tokens))
        output_dict['predicted_sentences'] = all_predicted_sents
        return output_dict
