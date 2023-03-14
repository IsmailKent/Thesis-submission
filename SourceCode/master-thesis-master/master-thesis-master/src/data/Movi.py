"""
Dataclass and loading of the MOVI dataset

This datasets use the preprocess MOVi files, which have been eytracted either as
images or .pt files into the following directory:
  -->

https://github.com/google-research/kubric/tree/main/challenges/movi
"""

import os
import imageio
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from CONFIG import CONFIG
PATH = CONFIG["paths"]["data_path"]


class MOVI(Dataset):
    """
    DataClass for the MOVI dataset.

    Args:
    -----
    datapath: string
        Path to the directory where the MOVi data is stored
    target: string
        Target data to load.
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load
    img_size: tuple
        Images are resized to this resolution
    slot_initializer: string
        Initialization mode used to initialize the slots
    """

    TARGETS = ["rgb", "flow"]
    MAX_OBJS = 11

    def __init__(self, datapath, target="rgb", split="validation", num_frames=24, img_size=(64, 64),
                 slot_initializer="LearnedInit"):
        """ Dataset initializer """
        assert target in MOVI.TARGETS
        assert split in ["train", "val", "valid", "validation", "test"]
        split = "validation" if split in ["val", "valid", "validation", "test"] else split

        # dataset parameters
        self.cur_data_path = os.path.join(datapath, split)
        if not os.path.exists(self.cur_data_path):
            raise FileNotFoundError(f"Data path {self.cur_data_path} does not exist")
        self.target = target
        self.split = split
        self.num_frames = num_frames
        self.img_size = img_size
        self.slot_initializer = slot_initializer
        self.get_rgb = True if target == "rgb" else False
        self.get_flow = True if target == "flow" else False
        self.get_masks = False
        self.get_bbox = False

        # resizer modules for the images and masks respectively
        self.resizer = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            )
        self.resizer_mask = transforms.Resize(
                self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST
            )

        # loading data
        self.db = self._load_data()
        return

    def __len__(self):
        """ Number of sequences in dataset """
        return len(self.db)

    def __getitem__(self, i):
        """
        Sampling a sequence from the dataset
        """
        all_data = self.db[i]
        start_frame = 0

        # loading images. Always the input, and sometimes the target as well
        imgs = []
        img_paths = all_data["imgs"][start_frame:start_frame+self.num_frames]
        imgs = [imageio.imread(frame) / 255. for frame in img_paths]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        imgs = self.resizer(imgs).float()

        # loading optical flow, which can be used as target
        flow = []
        if self.get_flow:
            flow_paths = all_data["flow"][start_frame:start_frame+self.num_frames]
            flow = [imageio.imread(flow) for flow in flow_paths]
            flow = np.stack(flow, axis=0)
            flow = torch.from_numpy(flow).permute(0, 3, 1, 2) / 255.
            flow = self.resizer_mask(flow).float()

        # default representation is either RGB-frames or optical flow  # TODO: depth
        targets = imgs if self.target == "rgb" else flow

        # loading instance segmentation masks if necessary. Can be used as conditioning or for eval
        segmentation = []
        if self.get_masks or self.slot_initializer == "Masks":
            semgentation = torch.load(all_data["masks"])  # TODO: Somehow, the masks were not saved
            semgentation = semgentation["masks"][start_frame:start_frame+self.num_frames, ..., 0]
            semgentation = np.stack(semgentation, axis=0)
            segmentation = torch.from_numpy(segmentation).permute(0, 3, 1, 2)
            segmentation = self.resizer_mask(segmentation)

        # loading center of mass and bounding box, if necessary. Can be used as conditioning
        com, bbox = [], []
        if self.get_bbox or self.slot_initializer in ["CoM", "BBox"]:
            coords = torch.load(all_data["coords"])
            bbox, com = coords["bbox"], coords["com"]
            bbox = bbox * imgs.shape[-1] / 128
            com = com * imgs.shape[-1] / 128
            bbox = bbox[start_frame:start_frame+self.num_frames]
            com = com[start_frame:start_frame+self.num_frames]

        data = {
                "frames": imgs,
                "flow": flow,
                "masks": segmentation,
                "com_coords": com,
                "bbox_coords": bbox,
            }
        return imgs, targets, data

    def _get_bbox_com(self, all_data, imgs):
        """
        Obtaining BBox information
        """
        bboxes = all_data["instances"]["bboxes"].numpy()
        bbox_frames = all_data["instances"]["bbox_frames"].numpy()
        num_frames, _, H, W = imgs.shape
        num_objects = bboxes.shape[0]
        com = torch.zeros(num_frames, num_objects, 2)
        bbox = torch.zeros(num_frames, num_objects, 4)
        for t in range(num_frames):
            for k in range(num_objects):
                if t in bbox_frames[k]:
                    idx = np.nonzero(bbox_frames[k] == t)[0][0]
                    min_y, min_x, max_y, max_x = bboxes[k][idx]
                    min_y, min_x = max(1, min_y * H), max(1, min_x * W)
                    max_y, max_x = min(H - 1, max_y * H), min(W - 1, max_x * W)
                    bbox[t, k] = torch.tensor([min_x, min_y, max_x, max_y])
                    com[t, k] = torch.tensor([(max_x + min_x) / 2, (max_y + min_y) / 2]).round()
                else:
                    bbox[t, k] = torch.ones(4) * -1
                    com[t, k] = torch.ones(2) * -1

        # padding so as to batch BBoxes or CoMs
        if num_objects < self.MAX_OBJS:
            rest = self.MAX_OBJS - num_objects
            rest_bbox = torch.ones((bbox.shape[0], rest, 4), device=imgs.device) * -1
            rest_com = torch.ones((bbox.shape[0], rest, 2), device=imgs.device) * -1
            bbox = torch.cat([bbox, rest_bbox], dim=1)
            com = torch.cat([com, rest_com], dim=1)

        return bbox, com

    def _load_data(self):
        """ Loading the data into a nice structure """
        db = {}
        all_files = os.listdir(self.cur_data_path)
        seqs = sorted(list(set([int(f.split("_")[1]) for f in all_files if "rgb" in f])))
        for seq in tqdm(seqs):
            db[seq] = {}
            db[seq]["imgs"] = [
                    os.path.join(self.cur_data_path, f"rgb_{seq:05d}_{i:02d}.png") for i in range(24)
                ]
            db[seq]["flow"] = [
                    os.path.join(self.cur_data_path, f"flow_{seq:05d}_{i:02d}.png") for i in range(24)
                ]
            db[seq]["masks"] = os.path.join(self.cur_data_path, f"mask_{seq:05d}.pt")
            db[seq]["coords"] = os.path.join(self.cur_data_path, f"coords_{seq:05d}.pt")
        return db


#
