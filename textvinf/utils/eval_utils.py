#!/usr/bin/env python3
import logging
import torch
import allennlp.nn.util as nn_util
import numpy as np
from typing import List, Iterable
from allennlp.common.checks import check_for_gpu
from allennlp.common import Registrable
from allennlp.common.tqdm import Tqdm
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators import DataIterator
from src.utils.model_utils import load_model


logger = logging.getLogger(__name__)


class Evaluator(Registrable):
    def __init__(self, model_dir: str,
                 dataset_reader: DatasetReader,
                 iterator: DataIterator,
                 evaluation_data_path: str,
                 cuda_device: int = 0,
                 epoch=-1):
        super().__init__()
        self.cuda_device = cuda_device
        self.model = load_model(model_dir, epoch, cuda_device)
        logger.info("Reading evaluation data from %s", evaluation_data_path)
        self.instances = dataset_reader.read(evaluation_data_path)
        self.iterator = iterator
        self.iterator.index_with(self.model.vocab)

    def get_encodings(self) -> Iterable[torch.Tensor]:
        check_for_gpu(self.cuda_device)
        with torch.no_grad():
            self.model.eval()
            iterator = self.iterator(self.instances, num_epochs=1, shuffle=False)
            logger.info("Iterating over dataset")
            generator_tqdm = Tqdm.tqdm(iterator, total=self.iterator.get_num_batches(self.instances))

            # Number of batches in instances.
            batch_count = 0

            for batch in generator_tqdm:
                batch_count += 1
                batch = nn_util.move_to_device(batch, self.cuda_device)
                z = self.model.encode(batch['source_tokens'])['latent']
                yield z

    def get_model_output(self, key: str) -> List[torch.Tensor]:
        check_for_gpu(self.cuda_device)
        outs = []
        with torch.no_grad():
            self.model.eval()
            iterator = self.iterator(self.instances, num_epochs=1, shuffle=False)
            logger.info("Iterating over dataset")
            generator_tqdm = Tqdm.tqdm(iterator, total=self.iterator.get_num_batches(self.instances))

            # Number of batches in instances.
            batch_count = 0

            for batch in generator_tqdm:
                batch_count += 1
                batch = nn_util.move_to_device(batch, self.cuda_device)
                output_dict = self.model(**batch)
                outs.append(output_dict[key].cpu())
        return outs


@Evaluator.register('decomposition')
class DecompositionEvaluator(Evaluator):
    def __init__(self, model_dir: str,
                 dataset_reader: DatasetReader,
                 iterator: DataIterator,
                 evaluation_data_path: str,
                 cuda_device: int = 0,
                 epoch=-1):
        super().__init__(model_dir, dataset_reader, iterator, evaluation_data_path, cuda_device, epoch)

    def get_subspaces(self) -> List[np.ndarray]:
        spzs = self.get_model_output('spz')
        num_subspaces = spzs[0].size(1)
        subspaces = [[] for _ in range(num_subspaces)]
        for spze in spzs:
            subspaces_ele = np.split(spze.numpy(), num_subspaces)
            for sidx, subspace in enumerate(subspaces_ele):
                subspaces[sidx].append(subspace)

        subspaces = [np.concatenate(subspace_eles) for subspace_eles in subspaces]
        return subspaces

    def get_decomposed_subspaces(self):
        for encoding in self.get_encodings():
            yield self.model._decomposer.get_decomposed_subspaces(encoding)

    def get_subspace_mean(self, subspace_idx) -> torch.Tensor:
        """
        return a z where each subspace is reduced to the mean
        """
        return torch.stack(
            [subspaces[subspace_idx].mean(dim=0) for subspaces in self.get_decomposed_subspaces()],
            dim=0
        ).mean(dim=0).squeeze()

    def get_transferred_sentences(self, embedding_list, subspace_idx, subspace_target: str):
        tsents = []
        rsents = []
        isents = []
        for encoding in self.get_encodings():
            subspaces = self.model._decomposer.get_decomposed_subspaces(encoding)
            zi = self.model._decomposer.invert(subspaces)
            subspaces[subspace_idx][:] = torch.tensor(
                embedding_list[subspace_idx][subspace_target],
                device=subspaces[subspace_idx].device
            )
            zm = self.model._decomposer.invert(subspaces)
            tsents.extend([' '.join(tokens[1:]) for tokens in self.model.generate_conditional(zm)['predicted_tokens']])
            rsents.extend(
                [' '.join(tokens[1:]) for tokens in self.model.generate_conditional(encoding)['predicted_tokens']])
            isents.extend([' '.join(tokens[1:]) for tokens in self.model.generate_conditional(zi)['predicted_tokens']])
        return tsents, rsents, isents
