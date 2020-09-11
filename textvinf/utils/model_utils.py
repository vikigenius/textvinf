# -*- coding: utf-8 -*-
import os

from allennlp.models.archival import load_archive


def load_model(model_dir, epoch, cuda_device):
    """Load the model."""
    weights_file = None
    if epoch > 0:
        weights_file = os.path.join(model_dir, 'model_state_epoch_{0}.th'.format(epoch))
    archive_file = os.path.join(model_dir, 'model.tar.gz')
    return load_archive(archive_file, cuda_device, weights_file=weights_file).model
