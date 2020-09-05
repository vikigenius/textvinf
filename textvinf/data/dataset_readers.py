# -*- coding: utf-8 -*-
import logging
from typing import Dict, Optional

from allennlp.common.file_utils import cached_path
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import SpacyTokenizer, Token, Tokenizer
from overrides import overrides

logger = logging.getLogger(__name__)


@DatasetReader.register('autoencoder')
class AutoencoderDatasetReader(DatasetReader):
    """DatasetReader for an AutoEncoder that reads a list of sentences."""

    def __init__(
        self,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        tokenizer: Optional[Tokenizer] = None,
        max_sequence_length: Optional[int] = None,
    ) -> None:
        """Initialize the reader."""
        super().__init__()
        self._tokenizer = tokenizer or SpacyTokenizer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_sequence_length = max_sequence_length

    @overrides
    def text_to_instance(self, input_string: str) -> Instance:
        """Convert text to instance."""
        tokenized_string = self._tokenizer.tokenize(input_string)[:self._max_sequence_length]
        tokenized_source = tokenized_string.copy()
        tokenized_target = tokenized_string.copy()
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._token_indexers)

        tokenized_target.insert(0, Token(START_SYMBOL))
        tokenized_target.append(Token(END_SYMBOL))
        target_field = TextField(tokenized_target, self._token_indexers)
        return Instance({'source_tokens': source_field, 'target_tokens': target_field})

    @overrides
    def _read(self, file_path):
        """Read lines from file and return instance."""
        with open(cached_path(file_path), 'r') as data_file:
            logger.info('Reading instances from lines in file at: {0}'.format(file_path))
            for _, line in enumerate(data_file):
                line = line.strip('\n')

                if not line:
                    continue

                yield self.text_to_instance(line)
