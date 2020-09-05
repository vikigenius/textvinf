#!/usr/bin/env python3
import os
from allennlp.models.archival import load_archive


def load_model(model_dir, epoch, cuda_device):
    weights_file = None
    if epoch > 0:
        weights_file = os.path.join(model_dir, f'model_state_epoch_{epoch}.th')
    archive_file = os.path.join(model_dir, 'model.tar.gz')
    model = load_archive(archive_file, cuda_device, weights_file=weights_file).model
    return model
