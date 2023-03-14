"""
adapted dataset class for mutli-d-sprites dataset from:
https://github.com/monniert/dti-sprites/blob/main/src/dataset/multi_object.py
"""

from functools import lru_cache
from PIL import Image

import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor, Compose

from CONFIG import CONFIG
PATH = CONFIG["paths"]["data_path"]


class MultiDSprites(Dataset):

    def __init__(self, path="./datasets", mode="train", **kwargs):
        assert mode in ["train", "val", "test", "valid", "eval"]
        mode = "test" if mode in ["test", "eval"] else mode
        mode = "val" if mode in ["val", "valid"] else mode
        path = path if path is not None else PATH

        self.name = 'dsprites_gray'
        self.img_size = (64, 64)
        self.N = 60000
        self.n_classes = 4
        self.path = path
        self.mode = mode
        self.eval_mode = kwargs.get('eval_mode', False) or mode == 'test'
        self.eval_semantic = kwargs.get('eval_semantic', False)

        if self.eval_mode:
            self.size = 320
        elif mode == 'val':
            np.random.seed(42)
            self.val_indices = np.random.choice(range(self.N), 100, replace=False)
            self.size = 100
        else:
            self.size = self.N

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        path = self.path
        if self.mode == 'val':
            idx = self.val_indices[idx]
        inp = self.transform(
                Image.open(f"{path}/multi-dsprites/dsprites_gray/images/{idx}.png").convert('RGB')
            )
        if self.eval_semantic:
            label = (self.transform_gt(
                    Image.open(f"{path}/multi-dsprites/dsprites_gray/sem_masks/{idx}.png").convert('L')
                ) * 255).long()
        else:
            label = (self.transform_gt(
                    Image.open(f"{path}/multi-dsprites/dsprites_gray/masks/{idx}.png").convert('L')
                ) * 255).long()

        return inp, inp, label

    @property
    @lru_cache()
    def transform(self):
        return Compose([ToTensor()])

    @property
    @lru_cache()
    def transform_gt(self):
        return Compose([ToTensor()])
