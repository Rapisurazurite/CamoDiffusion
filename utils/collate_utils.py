# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Callable, Type, Union, List, Set
import numpy as np
import torch
from collections.abc import Mapping, Sequence
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data.dataloader import default_collate, DataLoader
from torch.utils.data import Dataset


def collate(batch: Sequence, collect_types=set()):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.
    """
    collect_types.update({str, Image.Image})
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if any(isinstance(batch[0], collect_type) for collect_type in collect_types):
        return batch
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, collect_types) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: collate([d[key] for d in batch], collect_types)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


class SampleDataset(Dataset):
    def __init__(self, full_dataset: Dataset, indices: Sequence[int] = None, interval: int = None):
        """
        Args:
            full_dataset (Dataset): The full dataset.
            indices (Sequence[int]): The indices of the samples in the full dataset to be used.
            interval (int): The interval between two samples in the full dataset to be used.
        """
        super().__init__()
        assert (indices is None) ^ (interval is None), "Either indices or interval should be specified."
        self.full_dataset = full_dataset
        self.indices = indices if indices is not None else range(0, len(full_dataset), interval)

    def __getitem__(self, index):
        return self.full_dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return f"{self.__class__.__name__}(full_dataset={self.full_dataset}, indices={self.indices})"