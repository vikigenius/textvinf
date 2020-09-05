#!/usr/bin/env python3
import torch
import numpy as np
from typing import Iterable


def get_mask(mask_dims: Iterable[int]) -> torch.Tensor:
    "Given a list of mask dimensions, return a mask tensor"
    num_masks = len(mask_dims)
    mask_len = sum(mask_dims)
    mask = np.zeros(num_masks, mask_len)
    dimsum = 0
    for i, dim in enumerate(mask_dims):
        mask[i, dimsum:dim+dimsum] = 1
    return torch.tensor(mask)
