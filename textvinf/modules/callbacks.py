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
        pred_tokens = trainer.model.make_output_human_readable(
            trainer.batch_outputs(batch, for_training=False),
        )['predicted_tokens']
        outputs = trainer.model.make_output_human_readable(
                trainer.batch_outputs(batch, for_training=False),
        )['predicted_tokens']
        idx = random.randrange(0, len(outputs))

        vocab = trainer.model.vocab
        removal_tokens = {START_SYMBOL, END_SYMBOL, vocab._padding_token}

        pred_tokens = ' '.join(
            token for token in outputs[idx] if token not in removal_tokens
        )
        text_tokens = ' '.join(
            [
                vocab.get_token_from_index(tidx.item())
                for tidx in batch['source_tokens']['tokens']['tokens'][idx]
                if vocab.get_token_from_index(tidx.item()) not in removal_tokens
            ],
        )

        logger.info('{0} -> {1}'.format(text_tokens, pred_tokens))


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
        pred_tokens = trainer.model.generate()
        vocab = trainer.model.vocab
        removal_tokens = {START_SYMBOL, END_SYMBOL, vocab._padding_token}
        pred_tokens = ' '.join(
            token
            for token in trainer.model.generate()['predicted_tokens'][0]
            if token not in removal_tokens
        )
        logger.info(pred_tokens)
