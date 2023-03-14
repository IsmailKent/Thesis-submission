"""
DataClass for the PhysicalConcepts dataset.
Multiple geometric objects in a plance, where movements, interactions, and occlusions occur

https://github.com/deepmind/physical_concepts
"""

import os
import json
from itertools import compress
from tqdm import tqdm
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from webcolors import name_to_rgb

from CONFIG import COLORS, CONFIG
from data.data_utils import masks_to_boxes
PATH = CONFIG["paths"]["data_path"]


class PhysicalConcepts(Dataset):
    """
    DataClass for the PhysicalConcepts dataset.
    Multiple geometric objects in a plance, where movements, interactions, and occlusions occur

    Args:
    -----
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load
    img_size: tuple
        Images are resized to this resolution
    get_masks: bool
        If True, segmentation masks are also loaded
    """

    MAX_OBJS = 7
    MAX_LEN = {
            "train": 15,
            "val": 15,
            "test": 15,
        }
    TRAIN_VAL_SPLIT = 0.1
    DATA_PATH = "/home/nfs/inf6/data/datasets/PhysicalConcepts"

    def __init__(self, split, num_frames, img_size=(80, 120), get_masks=False,
                 slot_initializer="LearnedRandom"):
        """ Dataset initializer"""
        assert split in ["train", "val", "valid", "test"]
        split = "val" if split == "valid" else split
        split_dir = "test" if split == "test" else "train"
        data_dir = os.path.join(PhysicalConcepts.DATA_PATH, split_dir, "data")
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"PhysicalConcepts dataset does not exist in {data_dir}...")

        # dataset parameters
        self.data_dir = data_dir
        self.split = split
        self.num_frames = self._check_num_frames_param(num_frames=num_frames, split=split)
        self.img_size = img_size
        self.get_masks = get_masks
        self.slot_initializer = slot_initializer

        # resizer modules for the images and masks respectively
        self.resizer = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        self.resizer_mask = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST
            )

        # data
        self.sequences = self._find_valid_sequences()
        self.keys = list(self.sequences.keys())
        return

    def __len__(self):
        """ Number of sequences in dataset """
        return len(self.sequences)

    def __getitem__(self, i, get_masks=None):
        """
        Sampling a sequence from the dataset
        """
        get_masks = get_masks if get_masks is not None else self.get_masks
        cur_key = self.keys[i]
        cur_sequence = self.sequences[cur_key]

        img_paths = cur_sequence["imgs"][:self.num_frames]
        imgs = [imageio.imread(frame) / 255. for frame in img_paths]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        imgs = self.resizer(imgs).float()

        masks = []
        get_masks = True if self.get_masks or self.slot_initializer == "Masks" else False
        if get_masks or self.slot_initializer in ["CoM", "BBox"]:
            mask_paths = cur_sequence["masks"][:self.num_frames]
            masks = [imageio.imread(frame) for frame in mask_paths]
            masks = np.stack(masks, axis=0)
            masks = torch.from_numpy(masks).unsqueeze(1)
            masks = self.resizer_mask(masks)
            mask_vals = masks.unique()  # TODO: This can be optimized
            for i, val in enumerate(mask_vals):
                masks[masks == val] = i

        # preparing extra representations given slot intialization
        com = self._instances_to_com(masks) if self.slot_initializer == "CoM" else []
        bbox = self._instances_to_bbox(masks) if self.slot_initializer == "BBox" else []

        targets = imgs
        data = {
                "frames": imgs,
                "masks": masks,
                "com_coords": com,
                "bbox_coords": bbox,
                "seq_name": self.sequences[cur_key]["seq_name"]
            }
        return imgs, targets, data

    def _find_valid_sequences(self):
        """
        Finding valid sequences in the corresponding dataset split.
        """
        print(f"Preparing PhysicalConcepts {self.split} set...")
        sequences = {}

        # loading file with indices of sequences to skip
        skip_file = os.path.join(PhysicalConcepts.DATA_PATH, f"{self.split}_skip.json")
        if os.path.exists(skip_file):
            with open(skip_file) as f:
                skip_seqs = json.load(f)
        else:
            print(f"File for skipping undesired sequences {skip_file} does not exist...")
            skip_seqs = []

        # feetching valid sequences
        sequence_dirs = sorted(os.listdir(self.data_dir), key=lambda x: int(x.split("_")[-1]))
        sequence_dirs = self._train_val_split(sequence_dirs)
        sequence_dirs = [os.path.join(self.data_dir, seq) for seq in sequence_dirs]
        for seq_dir in tqdm(sequence_dirs):
            if seq_dir in skip_seqs:
                continue
            idx = int(seq_dir.split("_")[-1])
            sequences[idx] = {}
            imgs = [f"img_{i:02d}.png" for i in range(15)]
            masks = [f"mask_{i:02d}.png" for i in range(15)]
            sequences[idx]["imgs"] = [os.path.join(seq_dir, img) for img in imgs]
            sequences[idx]["masks"] = [os.path.join(seq_dir, mask) for mask in masks]
            sequences[idx]["seq_name"] = seq_dir

        if len(sequences) <= 0:
            raise ValueError("No valid sequences were found...")
        else:
            print(f"--> There have been found a total of {len(sequences)} sequences...")
        return sequences

    def _check_num_frames_param(self, num_frames, split):
        """
        Making sure the given 'num_frames' is valid for the corresponding split
        """
        if num_frames > self.MAX_LEN[split]:
            print(f"PhysicalConcept sequences have 15 frames. Your {num_frames = } will be overridden")
            num_frames = 15
        return num_frames

    def _train_val_split(self, sequence_dirs):
        """
        Splitting the training sequences into training and validation
        """
        num_seqs = len(sequence_dirs)
        val_seqs_freq = int(round(1 / PhysicalConcepts.TRAIN_VAL_SPLIT))

        all_idx = np.arange(num_seqs)
        train_idx = all_idx % val_seqs_freq != 0
        val_idx = (all_idx % val_seqs_freq) == 0

        if self.split == "train":
            sequence_dirs = list(compress(sequence_dirs, train_idx))
        elif self.split == "val":
            sequence_dirs = list(compress(sequence_dirs, val_idx))
        else:
            sequence_dirs = sequence_dirs[:]
        return sequence_dirs

    def _instances_to_one_hot(self, x):
        """
        Converting from instance indices into instance-wise one-hot encodings
        """
        num_classes = self.MAX_OBJS
        shape = x.shape
        x = x.flatten().to(torch.int64).view(-1,)
        y = torch.nn.functional.one_hot(x, num_classes=num_classes)
        y = y.view(*shape, num_classes)  # (..., Height, Width, Classes)
        y = y.transpose(-3, -1).transpose(-2, -1).unsqueeze(-3)  # (..., Classes, Height, Width)
        return y

    def _one_hot_to_instances(self, x):
        """
        Converting from one-hot multi-channel instance representation to single-channel instance mask
        """
        masks_merged = torch.argmax(x, dim=2)
        return masks_merged

    def _instances_to_rgb(self, x):
        """
        Converting from instance IDs to RGB images
        """
        img = torch.zeros(*x.shape, 3)
        for cls in range(self.NUM_CLASSES):
            color = COLORS[cls]
            color_rgb = torch.tensor(name_to_rgb(color)).float()
            img[x == cls, :] = color_rgb / 255
        img = img.transpose(-3, -1).transpose(-2, -1)
        return img

    def _instances_to_com(self, x):
        """
        Converting from instance IDs to Center of Mass coordinates, parameterized as (y, x)
        """
        masks = self._instances_to_one_hot(x)[:, 0]

        # computing center of mass by multiplying normalized masks with coordinate grid
        H, W = masks.shape[-2:]
        norm_masks = masks / (masks.sum(dim=(-1, -2), keepdim=True) + 1e-12)
        x, y = torch.arange(0, H, device=x.device), torch.arange(0, W, device=x.device)
        xx, yy = torch.meshgrid(x, y)
        norm_masks_xx, norm_masks_yy = (norm_masks * xx), (norm_masks * yy)
        center_of_mass = torch.cat([
                norm_masks_xx.sum(dim=(-2, -1)), norm_masks_yy.sum(dim=(-2, -1))
            ], dim=-1).flip(dims=(-1,)).round()
        return center_of_mass

    def _instances_to_bbox(self, x):
        """
        Converting from instance IDs to BBox coordinates, parameterized as (x_min, y_min, x_max, y_max)
        """
        masks = self._instances_to_one_hot(x)[:, 0]
        B, N, _, H, W = masks.shape
        masks = masks.reshape(B * N, H, W)
        bbox = masks_to_boxes(masks)
        bbox = bbox.view(B, N, 4)
        if N < self.MAX_OBJS:
            rest = self.MAX_OBJS - N
            rest_bbox = torch.ones((B, rest, 4), device=x.device) * -1
            bbox = torch.cat([bbox, rest_bbox], dim=1)
        return bbox

#
