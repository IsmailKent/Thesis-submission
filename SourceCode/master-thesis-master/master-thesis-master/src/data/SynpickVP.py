"""
SynpickVP dataset: Obtaining video sequences from the SynpickVP dataset.
However, we modify the original DataClass, since we are only interested in the instance
segmentation masks.

Adapted from: https://github.com/angelvillar96/SynpickVP
"""

import json
import math
import os
from tqdm import tqdm
import imageio
import torch
from torchvision import transforms
import numpy as np
from collections import defaultdict
from webcolors import name_to_rgb

from CONFIG import COLORS
from data.data_utils import masks_to_boxes


class SynpickVP:
    """
    Each sequence depicts a robotic suction cap gripper that moves around in a red bin filled with objects.
    Over the course of the sequence, the robot approaches 4 waypoints that are randomly chosen from the 4 corners.
    On its way, the robot is pushing around the objects in the bin.

    Args:
    -----
    split: string
        Dataset split to load
    num_frames: int
        Desired length of the sequences to load
    seq_step: int
        Temporal resolution at which we use frames. seq_step=2 means we use one frame out of each two
    max_overlap: float
        Determins amount of overlap between consecuitve sequences, given as percentage of num_frames.
        For instance, 0.25 means that consecutive sequence will overlap for (1 - 0.25)=75% of the frames
    img_size: tuple
        Images are resized to this resolution
    use_segmentation: bool
        If True, instance segmentations are the primary representation loaded. Otherwise,
        RGB frames are loaded instead.
    slot_initializer: string
        Type of slot intialization employed. If it is one of ["Mask", "CoM", "BBox"], the necessary
        data is computed and returned
    """

    DATA_PATH = "/home/nfs/inf6/data/datasets/SynpickRaw/processed"

    CATEGORIES = ["master_chef_can", "cracker_box", "sugar_box", "tomato_soup_can", "mustard_bottle",
                  "tuna_fish_can", "pudding_box", "gelatin_box", "potted_meat_can", "banana", "pitcher_base",
                  "bleach_cleanser", "bowl", "mug", "power_drill", "wood_block", "scissors", "large_marker",
                  "large_clamp", "extra_large_clamp", "foam_brick", "gripper"]
    NUM_CLASSES = len(CATEGORIES)  # 22
    MAX_OBJS = 14

    BIN_SIZE = (373, 615)
    SKIP_FIRST_N = 72            # To skip the first few frames in which gripper is not visible.
    GRIPPER_VALID_OFFSET = 0.01  # To skip sequences where the gripper_pos is near the edges of the bin.

    NICE_SIZES = [(64, 112), (136, 240)]
    NUM_FRAMES_LIMITS = {
            "train": [4, 50],
            "val": [50, 50],
            "test": [50, 50],  # please keep this fixed
        }

    def __init__(self, split, num_frames, seq_step=2, max_overlap=0.25, img_size=(136, 240),
                 use_segmentation=True, slot_initializer="LearnedRandom"):
        """
        Dataset initializer
        """
        assert split in ["train", "val", "valid", "test"]
        split = "val" if split == "valid" else split
        assert max_overlap <= 0.95 and max_overlap >= 0
        self.data_dir = os.path.join(SynpickVP.DATA_PATH, split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Synpick dataset does not exist in {self.data_dir}...")

        self.split = split
        self.img_size = img_size
        self.given_num_frames = num_frames
        self.num_frames = self._check_num_frames_param(split=split, num_frames=num_frames)
        self.seq_step = seq_step
        self.max_overlap = max_overlap
        self.use_segmentation = use_segmentation
        self.slot_initializer = slot_initializer
        self.get_semantic = False
        self.get_masks = False

        # obtaining paths to data
        images_dir = os.path.join(self.data_dir, "rgb")
        scene_dir = os.path.join(self.data_dir, "scene_gt")
        masks_dir = os.path.join(self.data_dir, "masks")
        instances_dir = os.path.join(self.data_dir, "instance_masks")
        self.image_ids = sorted(os.listdir(images_dir))
        self.image_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.scene_gt_fps = [os.path.join(scene_dir, scene) for scene in sorted(os.listdir(scene_dir))]
        self.mask_fps = [os.path.join(masks_dir, mask_fp) for mask_fp in sorted(os.listdir(masks_dir))]
        self.instance_fps = [os.path.join(instances_dir, inst_fp) for inst_fp in sorted(os.listdir(instances_dir))]

        # obtaining pose information from the different objectd
        self.object_poses = {}
        self._create_frame_to_obj_kpoints_dict()

        # generating sequences
        self.valid_idx = []
        self.allow_seq_overlap = (split == "train")
        self._find_valid_sequences()
        return

    def __len__(self):
        """ Number of valid sequences in the dataset """
        return len(self.valid_idx)

    def __getitem__(self, i):
        """
        Sampling sequence from the dataset, along with all the necessary representations
        """
        i = self.valid_idx[i]  # only consider valid indices
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        idx = range(i, i + seq_len, self.seq_step)  # create range of indices for frame sequence
        idx = idx[:self.given_num_frames]

        # loading either RGB imgs or instance segmentations
        imgs = self._get_rgb_images(idx=idx) if not self.use_segmentation else []
        semantic = self._get_segmentation_maps(path=self.mask_fps, idx=idx) if self.get_semantic else []
        if self.use_segmentation or self.slot_initializer in ["Masks", "CoM", "BBox"]:
            instances = self._get_segmentation_maps(path=self.instance_fps, idx=idx)
        else:
            instances = []
        self.last_instances = instances

        # preparing extra representations given slot intialization
        get_masks = True if self.get_masks or self.slot_initializer == "Masks" else False
        masks = self._instances_to_one_hot(instances) if get_masks else []
        com = self._instances_to_com(instances) if self.slot_initializer == "CoM" else []
        bbox = self._instances_to_bbox(instances) if self.slot_initializer == "BBox" else []

        video = self._instances_to_rgb(instances) if self.use_segmentation else imgs
        targets = video
        data = {
                "frames": imgs,
                "segmentation": semantic,
                "instances": instances,
                "masks": masks,
                "com_coords": com,
                "bbox_coords": bbox,
            }
        return video, targets, data

    def get_last_masks(self):
        """ """
        masks = self._instances_to_one_hot(self.instances)
        return masks

    def _get_rgb_images(self, idx):
        """
        Loading RGB images corresponding to sequence idx
        """
        imgs = [imageio.imread(self.image_fps[id_]) / 255. for id_ in idx]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)
        imgs = transforms.Resize(self.img_size)(imgs)
        return imgs

    def _get_segmentation_maps(self, path, idx):
        """
        Loading segmentation maps for sequence idx, the segmentations can be
        either semantic or instance segmentation maps
        """
        seg_maps = np.array([imageio.imread(path[id_]) for id_ in idx])[..., 0]
        seg_maps = torch.from_numpy(seg_maps)
        seg_maps = transforms.Resize(
                size=self.img_size,
                interpolation=transforms.InterpolationMode.NEAREST
            )(seg_maps)
        return seg_maps

    def _create_frame_to_obj_kpoints_dict(self):
        """
        Loading object poses from scene-ground-truth file into a data structure.
        """
        for scene_gt_fp in self.scene_gt_fps:
            # loading ground truth (meta-)data from an episode
            ep_num = self._ep_num_from_fname(scene_gt_fp)
            with open(scene_gt_fp, 'r') as scene_json_file:
                ep_dict = json.load(scene_json_file)
            # obtaining objects and adding poses
            object_poses = []
            for frame_elem in ep_dict.values():
                frame_obj_dict = defaultdict(list)
                for obj_elem in frame_elem:
                    obj_id = obj_elem["obj_id"]
                    pos_3d = obj_elem["cam_t_m2c"]
                    frame_obj_dict[obj_id].append(pos_3d)
                object_poses.append(frame_obj_dict)
            self.object_poses[ep_num] = object_poses
        return

    def _find_valid_sequences(self):
        """
        Finding valid sequences in the corresponding dataset split.
        The following conditions are enforced:
            - Sequence has the sufficient number of frames
            - No overlap between sequences in validation and test sets
            - Selecting only sequences in which the robot gripper is present
        """
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        last_valid_idx = -1 * seq_len
        frame_offsets = range(0, self.num_frames * self.seq_step, self.seq_step)
        for idx in tqdm(range(len(self.image_ids) - seq_len + 1)):
            # handle overlapping sequences
            if (not self.allow_seq_overlap and (idx < last_valid_idx + seq_len)) \
                    or (self.allow_seq_overlap and (idx < last_valid_idx + seq_len * self.max_overlap)):
                continue

            ep_nums = [self._ep_num_from_id(self.image_ids[idx + offset]) for offset in frame_offsets]
            frame_nums = [self._frame_num_from_id(self.image_ids[idx + offset]) for offset in frame_offsets]
            # first few frames are discarded
            if frame_nums[0] < self.SKIP_FIRST_N:
                continue
            # last T frames of an episode should not be chosen as the start of a sequence
            if ep_nums[0] != ep_nums[-1]:
                continue

            # obtaining the pose of the gripper in the selected frames
            gripper_obj_id = self.NUM_CLASSES
            gripper_pos = [self.object_poses[ep_nums[0]][frame_num][gripper_obj_id][0] for frame_num in frame_nums]

            # discard sequences where the gripper_pos is not low enough (e.g. Gripper still descending)
            offset = self.GRIPPER_VALID_OFFSET
            gripper_xy = self._to_img_plane(gripper_pos[0])
            if not ((offset <= gripper_xy[0] <= 1.0 - offset) and (offset <= gripper_xy[1] <= 1.0 - offset)):
                continue
            gripper_xy = self._to_img_plane(gripper_pos[-1])
            if not ((offset <= gripper_xy[0] <= 1.0 - offset) and (offset <= gripper_xy[1] <= 1.0 - offset)):
                continue

            # discard sequences without considerable gripper movement
            gripper_pos_deltas = self._get_gripper_pos_xydist(gripper_pos)
            gripper_pos_deltas_above_min = [(delta > 1.0) for delta in gripper_pos_deltas]
            gripper_pos_deltas_below_max = [(delta < 30.0) for delta in gripper_pos_deltas]
            most = lambda lst, factor=0.67: sum(lst) >= factor * len(lst)
            gripper_movement_ok = most(gripper_pos_deltas_above_min) and all(gripper_pos_deltas_below_max)
            if not gripper_movement_ok:
                continue
            self.valid_idx.append(idx)
            last_valid_idx = idx

        if len(self.valid_idx) <= 0:
            raise ValueError("No valid sequences were found...")
        return

    def _to_img_plane(self, pos_3d):
        """ Pose3d to pose2d """
        Bin_H, Bin_W = self.BIN_SIZE
        px, py, _ = pos_3d
        x, y = px / Bin_W + 0.5, py / Bin_H + 0.5
        # discard objects dropped outside of the bin
        return (x, y) if ((0 <= x < 1.0) and (0 <= y < 1.0)) else (0., 0.)

    def _comp_gripper_pos(self, old, new):
        x_diff, y_diff = new[0] - old[0], new[1] - old[1]
        return math.sqrt(x_diff * x_diff + y_diff * y_diff)

    def _get_gripper_pos_xydist(self, gripper_pos):
        return [self._comp_gripper_pos(old, new) for old, new in zip(gripper_pos, gripper_pos[1:])]

    def _get_gripper_pos_diff(self, gripper_pos):
        gripper_pos_numpy = np.array(gripper_pos)
        return np.stack([new-old for old, new in zip(gripper_pos_numpy, gripper_pos_numpy[1:])], axis=0)

    def _ep_num_from_id(self, file_id):
        """ Obtaining episode number from RBG image filename """
        return int(file_id[-17:-11])

    def _frame_num_from_id(self, file_id):
        """ Obtaining frame number in episode from RBG image filename """
        return int(file_id[-10:-4])

    def _ep_num_from_fname(self, file_name):
        """ Fetching video/episode name from the scene_gt file name """
        if "scene_gt" not in file_name:
            raise ValueError(f"Given {file_name = } is not a *scene_gt.json file...")
        return int(file_name[-20:-14])

    def _check_num_frames_param(self, num_frames, split):
        """
        Making sure the given 'num_frames' is valid for the corresponding split
        """
        if split == "test" or split == "val":
            if num_frames != self.NUM_FRAMES_LIMITS[split][0]:
                print(f"SynpickVP test-sequences have 50 frames. Your {num_frames = } will be overridden")
                num_frames = 50
        else:
            if num_frames < self.NUM_FRAMES_LIMITS[split][0]:
                raise ValueError(f"{num_frames = } must be >= {self.NUM_FRAMES_LIMITS[split][0]}")
            if num_frames > self.NUM_FRAMES_LIMITS[split][1]:
                raise ValueError(f"{num_frames = } must be <= {self.NUM_FRAMES_LIMITS[split][1]}")
        return num_frames

    def _instances_to_one_hot(self, x, instance=True):
        """
        Converting from instance indices into instance-wise one-hot encodings
        """
        num_classes = self.MAX_OBJS if instance else self.NUM_CLASSES
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
        masks = self._instances_to_one_hot(x)

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
        masks = self._instances_to_one_hot(x)
        B, N, _, H, W = masks.shape
        masks = masks.reshape(B * N, H, W)
        bbox = masks_to_boxes(masks)
        bbox = bbox.view(B, N, 4)
        if N < self.MAX_OBJS:
            rest = self.MAX_OBJS - N
            rest_bbox = torch.ones((B, rest, 4), device=x.deivce) * -1
            bbox = torch.cat([bbox, rest_bbox], dim=1)
        return bbox


class SynpickInstances(SynpickVP):
    """
    Version of the SynpickVP dataset that simply returns the instance segmentation of the scene.
    Instance segmentation is represented by a multi-channel representation, where each channel
    correspond to the binary segmentation mask of one instance.
    """

    def __getitem__(self, i):
        """
        Sampling sequence from the dataset, along with all the necessary representations
        """
        i = self.valid_idx[i]  # only consider valid indices
        seq_len = (self.num_frames - 1) * self.seq_step + 1
        idx = range(i, i + seq_len, self.seq_step)  # create range of indices for frame sequence
        idx = idx[:self.given_num_frames]

        instances = self._get_segmentation_maps(path=self.instance_fps, idx=idx)
        instance_masks = self._instances_to_one_hot(instances)
        data = {
                "frames": [],
                "segmentation": [],
                "instances": instances,
                "masks": instance_masks,
                "com_coords": [],
                "bbox_coords": [],
                "video": [],
            }
        return instance_masks, data


#
