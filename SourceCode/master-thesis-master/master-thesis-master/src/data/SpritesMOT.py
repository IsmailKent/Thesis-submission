"""
Loading for the synthetic datasets from  Object-Centric-Representation-Benchmark
https://github.com/ecker-lab/object-centric-representation-benchmark
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from CONFIG import CONFIG
PATH = CONFIG["paths"]["data_path"]


class SpritesDataset(Dataset):
    """
    Dataset with colored sprites (hearts, ellipsis, squares, ...) moving around in a canvas

    Args:
    -----
    path: string
        Path from which the data is loaded. If none, path is taken from CONFIG file
    mode: string
        current dataset split
    n_imgs: integer
        number of frames to load
    dataset: string
        Sprites dataset to load. It can be ['vmds', 'vor', 'spmot']
    """

    DATASETS = ['vmds', 'vor', 'spmot']

    def __init__(self, path=None, mode='train', n_imgs=10, dataset='spmot', transform=None, T=0, rgb=True):
        """ """
        assert dataset in SpritesDataset.DATASETS, f"Unknown dataset = {dataset}. Not in {SpritesDataset.DATASETS}"
        assert mode in ["train", "val", "test", "valid", "eval"]
        mode = "test" if mode in ["test", "eval"] else mode
        mode = "val" if mode in ["val", "valid"] else mode
        path = path if path is not None else PATH
        self.mode = mode
        self.n_imgs = n_imgs
        self.path = path
        self.dataset = dataset
        self.rgb = rgb
        self.transform = transform
        data_path = os.path.join(path, dataset)
        data_file = os.path.join(data_path, f'{dataset}_{mode}.npy')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Path {data_path} does not exist")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} does not exist")

        imgs = np.load(data_file)
        imgs = imgs[:, :n_imgs]
        if T and T < n_imgs:
            imgs = np.concatenate(np.split(imgs, imgs.shape[1] // T, axis=1))
        self.num_samples = len(imgs)
        print(f"self.num_samples {self.num_samples}")
        print(f"imgs.shape {imgs.shape}")

        self.imgs = imgs
        if mode == "test":
            self.get_gt_test_masks()
        return

    def __getitem__(self, index):
        """ """
        x = self.imgs[index]
        x = torch.from_numpy(x) / 255
        if self.transform is not None:
            x = self.transform(x)
        if(not self.rgb):
            x = self.rgb2gray(x)
        masks = self.masks[index] if self.mode == "test" else torch.zeros(x.shape)

        targets = x
        all_reps = {
                "imgs": x,
                "masks": masks
            }
        return x, targets, all_reps

    def __len__(self):
        """ """
        return self.num_samples

    def rgb2gray(self, rgb):
        """ """
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray[:, np.newaxis]

    def get_gt_dict(self):
        """ Loading the dictionary with ground truths for evaluation """
        file = os.path.join(self.path, self.dataset, "gt_masks_test.json")
        with open(file) as f:
            gt_dict = json.load(f)
        return gt_dict

    def get_gt_test_masks(self):
        """ """
        file = os.path.join(CONFIG["paths"]["data_path"], "vmds", "vmds_test.json")
        with open(file) as f:
            gt_dict = json.load(f)
        self.masks = []
        for V in range(self.num_samples):
            video_masks = torch.zeros((self.n_imgs, self.imgs.shape[-2], self.imgs.shape[-1]), dtype=torch.uint8)
            for F in range(self.n_imgs):
                gt_dict_frame = gt_dict[V][F]
                mask = self.decode_gt_frame(gt_dict_frame)
                video_masks[F] = mask
            self.masks.append(video_masks)

    def decode_gt_frame(self, gt_frame):
        """ """
        # Compute pairwise distances between gt objects and predictions per frame.
        H, W = 64, 64
        n_gt = len(gt_frame["masks"])
        # accumulate gt masks for frame
        gts = []
        gt_ids = []
        combined_mask = torch.zeros((H, W), dtype=torch.uint8)
        for h in range(n_gt):
            # excluding gt-background mask
            cur_id = gt_frame["ids"][h]
            if(not isinstance(cur_id, int) and "bg" in cur_id):
                continue
            mask = self.decode_rle(gt_frame['masks'][h], (H, W))
            mask_number = h+1
            combined_mask += (mask * mask_number)
        return combined_mask

    def decode_rle(self, mask_rle, shape):
        """
        from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = torch.zeros(shape[0]*shape[1], dtype=torch.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
