# -*- coding: utf-8 -*-

from typing import Any, Dict

import torch
from allennlp.common.registrable import Registrable
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics import BLEU, Average

from textvinf.modules.metrics import WordPPL


class Decoder(torch.nn.Module, Registrable):
    """``Decoder`` class is a wrapper for different decoders."""

    def __init__(self, vocab: Vocabulary):
        super().__init__()  # type: ignore
        self.vocab = vocab
        self._nll = Average()
        self._ppl = WordPPL()
        self._start_index = self.vocab.get_token_index(START_SYMBOL)
        self._end_index = self.vocab.get_token_index(END_SYMBOL)
        self._pad_index = self.vocab.get_token_index(self.vocab._padding_token)  # noqa: WPS437
        self._bleu = BLEU(exclude_indices={self._pad_index, self._end_index, self._start_index})

    def forward(
        self,
        encoder_outs: Dict[str, Any],
        target_tokens: Dict[str, torch.LongTensor],
    ) -> Dict[str, torch.Tensor]:
        """Run the module's forward function."""
        raise NotImplementedError

    def post_process(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Post process after decoding."""
        raise NotImplementedError

    def get_metrics(self, reset: bool = False):
        """Collect all available metrics."""
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self._bleu.get_metric(reset=reset))
        all_metrics.update({'nll': float(self._nll.get_metric(reset=reset))})
        all_metrics.update({'_ppl': float(self._ppl.get_metric(reset=reset))})
        return all_metrics
