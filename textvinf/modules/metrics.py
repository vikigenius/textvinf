# -*- coding: utf-8 -*-
import math
from pathlib import Path
from typing import List

import kenlm
import torch
from allennlp.training.metrics import Average, Metric
from overrides import overrides


@Metric.register('kenlm_ppl')
class KenLMPPL(Average):
    def __init__(self, lm_path: Path):
        super().__init__()
        self.lm = kenlm.LanguageModel(lm_path)

    @overrides
    def __call__(self, predicted_tokens: List[List[str]]):
        for sent_tokens in predicted_tokens:
            sentence = ' '.join(sent_tokens[1:])
            ppl = self.lm.perplexity(sentence)
            super().__call__(ppl)


@Metric.register('word_ppl')
class WordPPL(Metric):
    def __init__(self) -> None:
        self.num_words = 0
        self.nll = 0

    def __call__(self, nll: torch.Tensor, num_words: int):
        """
        Args:
            nll: sum of nll for all batch_elements
            num_words: total number of words in this batch
        """
        self.num_words += num_words
        self.nll += list(self.detach_tensors(nll))[0]

    @overrides
    def reset(self):
        self.num_words = 0
        self.nll = 0

    @overrides
    def get_metric(self, reset: bool = False):
        try:
            ppl = math.exp(self.nll / self.num_words)
        except ZeroDivisionError:
            ppl = 1
        if reset:
            self.reset()
        return ppl
