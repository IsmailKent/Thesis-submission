"""
DataClass for the Sketchy dataset.
In this dataset, a robotic gripper grabs and moves around some colorful cubes

https://sites.google.com/view/data-driven-robotics/
"""

import os
import numpy as np
from tqdm import tqdm
import imageio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from CONFIG import CONFIG
PATH = CONFIG["paths"]["data_path"]


class Sketchy(Dataset):
    """
    DataClass for the Sketchy dataset.
    In this dataset, a robotic gripper grabs and moves around some colorful cubes

    Args:
    -----
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load
    seq_step: int
        Temporal resolution at which we use frames. seq_step=2 means we use one frame out of each two
    max_overlap: float
        Determines amount of overlap between consecuitve sequences, given as percentage of num_frames.
        For instance, 0.25 means that consecutive sequence will overlap for 75% of the frames
    img_size: tuple
        Images are resized to this resolution
    """

    CAMERAS = ["back_left", "front_right", "front_left"]
    CAMERA = "front_left"
    SPLIT_IDX = [10000, 11300]  # approx 80% training, 10% validation and 10% test
    DATA_PATH = "/home/nfs/inf6/data/datasets/SketchyRobot/sketchy_data"

    MAX_OBJS = 11
    NICE_SIZES = [(80, 120), (120, 192), (600, 960)]  # frames are originally (600 x 960)
    NUM_FRAMES_LIMITS = {
            "train": [50, 50],
            "val": [50, 50],
            "test": [50, 50],  # please keep this fixed
        }

    def __init__(self, split, num_frames, seq_step=2, img_size=(80, 120), max_overlap=0., mode=None):
        """ Dataset initializer"""
        assert mode is None or mode in self.CAMERAS
        assert split in ["train", "val", "valid", "test"]
        split = "val" if split == "valid" else split
        assert max_overlap <= 0.95 and max_overlap >= 0
        self.data_dir = os.path.join(Sketchy.DATA_PATH)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Sketchy dataset does not exist in {self.data_dir}...")

        # dataset parameters
        self.split = split
        self.given_num_frames = num_frames
        self.num_frames = self._check_num_frames_param(split=split, num_frames=num_frames)
        self.seq_step = seq_step
        self.img_size = img_size
        self.max_overlap = max_overlap
        self.mode = mode if mode is not None else self.CAMERA

        # aux modules
        self.resizer = transforms.Resize(self.img_size)

        # generating sequences
        self.valid_sequences = []
        self.valid_idx = []
        self.episodes, self.accumulated_sum = self._get_episode_data()
        self.num_imgs = self.accumulated_sum[-1]
        self.allow_seq_overlap = (split == "train")
        self._find_valid_sequences()
        return

    def __len__(self):
        """ Number of sequences in dataset """
        return len(self.valid_sequences)

    def __getitem__(self, i):
        """ Sampling a sequence from the dataset """
        cur_frames = self.valid_sequences[i]
        cur_frames = cur_frames[:self.given_num_frames]
        imgs = [imageio.imread(frame) / 255. for frame in cur_frames]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)
        imgs = self.resizer(imgs)

        targets = imgs
        data = {
                "frames": imgs,
            }
        return imgs, targets, data

    def _find_valid_sequences(self):
        """
        Finding valid sequences in the corresponding dataset split.
        """
        print(f"Preparing Sketchy {self.split} set...")
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        last_valid_idx = -1 * seq_len
        last_episode = ""

        for idx in tqdm(range(self.num_imgs - seq_len + 1)):
            # handle overlapping sequences
            episode, start_frame_idx = self._get_episode_from_idx(idx)

            if (not self.allow_seq_overlap and (idx < last_valid_idx + seq_len) and (last_episode == episode)):
                continue
            if self.allow_seq_overlap and (idx < last_valid_idx + seq_len * (1-self.max_overlap)):
                continue

            # obtaining frames for the sequence, if current idx is valid
            cur_sequence = self._frames_from_id(idx, episode, start_frame_idx)
            if len(cur_sequence) == 0:
                continue

            self.valid_sequences.append(cur_sequence)
            self.valid_idx.append(idx)
            last_valid_idx = idx
            last_episode = episode

        if len(self.valid_idx) <= 0:
            raise ValueError("No valid sequences were found...")
        if len(self.valid_sequences) <= 0:
            raise ValueError("No valid sequences were found...")
        return

    def _get_episode_from_idx(self, frame_idx):
        """ Getting episode name from idx """
        start_indices = [0] + self.accumulated_sum.tolist()
        idx = np.argwhere(self.accumulated_sum > frame_idx)[0][0]
        episode_dir = self.episodes[idx]
        start_frame_idx = start_indices[idx]
        return episode_dir, start_frame_idx

    def _frames_from_id(self, frame_idx, episode_dir, start_frame_idx):
        """ Fetching frame paths given frame_idx """
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        start_offset = frame_idx - start_frame_idx
        episode_path = os.path.join(self.data_dir, episode_dir)
        num_episodes = len([f for f in os.listdir(episode_path) if self.mode in f])
        if num_episodes - start_offset >= seq_len:
            # episode_frames = sorted(os.listdir(episode_path))
            frame_offsets = range(
                    start_offset,
                    start_offset + self.num_frames * self.seq_step,
                    self.seq_step
                )
            # frame_paths = [episode_frames[i] for i in frame_offsets]
            paths = [os.path.join(episode_path, f"{self.mode}_{i:04d}.png") for i in frame_offsets]
        else:
            paths = []
        return paths

    def _ep_num_from_id(self, frame_idx):
        """ Getting episode number given frame_idx """
        episode_idx = np.argwhere(self.accumulated_sum > frame_idx)[0][0]
        episode_dir = f"episode_{episode_idx:04d}"
        return episode_dir

    def _check_num_frames_param(self, num_frames, split):
        """
        Making sure the given 'num_frames' is valid for the corresponding split
        """
        if num_frames != self.NUM_FRAMES_LIMITS[split][0]:
            print(f"Sketchy sequences have 50 frames. Your {num_frames = } will be overridden")
            num_frames = 50
        return num_frames

    def _get_episode_data(self):
        """ Obtaining paths for the frames and episodes for the current dataset split """
        all_episodes = sorted(os.listdir(self.data_dir))
        if self.split == "train":
            episodes = all_episodes[:self.SPLIT_IDX[0]]
        elif self.split == "val":
            episodes = all_episodes[self.SPLIT_IDX[0]:self.SPLIT_IDX[1]]
        else:
            episodes = all_episodes[self.SPLIT_IDX[1]:]

        len_episodes = [0] * len(episodes)
        for i, ep in enumerate(episodes):
            for f in os.listdir(os.path.join(self.data_dir, ep)):
                if f.startswith(self.mode):
                    len_episodes[i] += 1
        accumulated_sum = np.cumsum(len_episodes)
        return episodes, accumulated_sum
