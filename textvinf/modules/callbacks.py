# -*- coding: utf-8 -*-
import logging
import random
from typing import Any, Dict

from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.training.trainer import EpochCallback, GradientDescentTrainer

logger = logging.getLogger(__name__)



@EpochCallback.register('print_reconstruction_example')
class PrintReconstructionExample(EpochCallback):
    """Callback that prints an example reconstruction."""

    def __call__(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ):
        """Callback call implementation."""
        batch = next(iter(trainer._validation_data_loader))
        outputs = trainer.model.make_output_human_readable(
                trainer.batch_outputs(batch, for_training=False),
        )['predicted_sentences']
        idx = random.randrange(0, len(outputs))

        vocab = trainer.model.vocab
        removal_tokens = {START_SYMBOL, END_SYMBOL, vocab._padding_token}

        pred_sentence = outputs[idx]
        source_sentence = ' '.join(
            [
                vocab.get_token_from_index(tidx.item())
                for tidx in batch['source_tokens']['tokens']['tokens'][idx]
                if vocab.get_token_from_index(tidx.item()) not in removal_tokens
            ],
        )

        logger.info('{0} -> {1}'.format(source_sentence, pred_sentence))


@EpochCallback.register('print_generation_example')
class PrintGenerationExample(EpochCallback):
    """Callback that prints an example reconstruction."""

    def __call__(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ):
        """Callback call implementation."""
        pred_sentence = random.choice(trainer.model.generate()['predicted_sentence'])
        logger.info(pred_sentence)
