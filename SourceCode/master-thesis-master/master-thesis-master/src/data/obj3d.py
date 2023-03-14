"""
Dataset class to load OBJ3D dataset
  - Source: https://github.com/zhixuan-lin/G-SWM/blob/master/src/dataset/obj3d.py
"""

from torch.utils.data import Dataset
from torchvision import transforms
import glob
import os
import os.path as osp
import torch
from PIL import Image, ImageFile

from CONFIG import CONFIG
PATH = CONFIG["paths"]["data_path"]

ImageFile.LOAD_TRUNCATED_IMAGES = True


class OBJ3D(Dataset):
    """
    DataClass for the Obj3D Dataset
    """

    def __init__(self, mode, ep_len=30, sample_length=20):
        """ Dataset Initializer """
        assert mode in ["train", "val", "valid", "eval", "test"], f"Unknown dataset split {mode}..."
        mode = "val" if mode in ["val", "valid"] else mode
        mode = "test" if mode in ["test", "eval"] else mode

        assert mode in ['train', 'val', 'test']
        self.root = os.path.join(PATH, "OBJ3D", mode)
        self.mode = mode
        self.sample_length = sample_length

        # Get all numbers
        self.folders = []
        for file in os.listdir(self.root):
            try:
                self.folders.append(int(file))
            except ValueError:
                continue
        self.folders.sort()

        self.epsisodes = []
        self.EP_LEN = ep_len
        self.seq_per_episode = self.EP_LEN - self.sample_length + 1 if mode != "test" else 1

        for f in self.folders:
            dir_name = os.path.join(self.root, str(f))
            paths = list(glob.glob(osp.join(dir_name, 'test_*.png')))
            # if len(paths) != self.EP_LEN:
            #     continue
            # assert len(paths) == self.EP_LEN, 'len(paths): {}'.format(len(paths))
            get_num = lambda x: int(osp.splitext(osp.basename(x))[0].partition('_')[-1])
            paths.sort(key=get_num)
            self.epsisodes.append(paths)
        return

    def __getitem__(self, index):
        """ """
        imgs = []
        # Implement continuous indexing
        ep = index // self.seq_per_episode
        offset = index % self.seq_per_episode
        end = offset + self.sample_length

        e = self.epsisodes[ep]
        for image_index in range(offset, end):
            img = Image.open(osp.join(e[image_index]))
            img = img.resize((64, 64))
            img = transforms.ToTensor()(img)[:3]
            imgs.append(img)
        img = torch.stack(imgs, dim=0).float()

        targets = img
        all_reps = {"videos": img}
        return img, targets, all_reps

    def __len__(self):
        """ """
        length = len(self.epsisodes)
        return length


#
