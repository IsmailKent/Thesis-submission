"""
Computation of different metrics
"""

import os
import json
import piqa
import numpy as np
import torch
import lib.utils as utils
from lib.logger import print_
from lib.utils import create_directory, timestamp
from lib.visualizations import visualize_ari, visualize_metric
from CONFIG import METRICS

# from sklearn.metrics import adjusted_rand_score
from scipy.special import comb


class MetricTracker:
    """
    Class for computing several evaluation metrics
    """

    def __init__(self, exp_path, metrics=["accuracy"], num_slots=6, use_tboard=True, tboard_name=None):
        """ Module initializer """
        assert isinstance(metrics, list), f"Metrics argument must be a list, and not {type(metrics)}"
        for metric in metrics:
            if metric not in METRICS:
                raise NotImplementedError(f"Metric {metric} not implemented. Use one of {METRICS}")
        self.n_instances = num_slots
        self.exp_path = exp_path

        # starting new tensorboard logs in exp_directory to store evalution results
        if use_tboard:
            tboard_name = tboard_name if tboard_name is not None else f"tboard_evaluation_{timestamp()}"
            tboard_logs = os.path.join(self.exp_path, "tboard_evaluation_logs", tboard_name)
            utils.create_directory(tboard_logs)
            self.tb_writer = utils.TensorboardWriter(logdir=tboard_logs)
            print_(f"  --> Storing evaluation results in tensorboard: {tboard_logs}")
        else:
            self.tb_writer = None

        self.metric_computers = {}
        print("Using evaluation metrics:")
        for metric in metrics:
            print(f"  --> {metric}")
            self.metric_computers[metric] = self._get_metric(metric, self.tb_writer)
        self.reset_results()
        self.acc_step = 0
        return

    def reset_results(self):
        """ Reseting results and metric computers """
        self.results = {m: None for m in self.metric_computers.keys()}
        for m in self.metric_computers.values():
            m.reset()
        self.acc_step = 0
        return

    def get_best_trial(self, num_trials=3):
        """ Selecting the best result among the last 'n_trials' """
        for _, metric_computer in self.metric_computers.items():
            metric_computer.get_best_trial(num_trials=num_trials)
        return

    def accumulate(self, preds, targets):
        """ Computing the different metrics and adding them to the results list """
        for metric_name, metric_computer in self.metric_computers.items():
            score = metric_computer.accumulate(preds=preds, targets=targets)
            if self.tb_writer:
                self.tb_writer.add_scalar(metric_name, score, self.acc_step)
            self.acc_step += 1
        return

    def aggregate(self):
        """ Aggregating the results for each metric """
        for metric, metric_computer in self.metric_computers.items():
            metric_data = metric_computer.aggregate()
            if isinstance(metric_data, (list, tuple)) and len(metric_data) == 2:
                mean_results, framewise_results = metric_data
                self.results[metric] = {}
                self.results[metric]["mean"] = mean_results
                self.results[metric]["framewise"] = framewise_results
            else:
                self.results[metric] = metric_data
        return

    def get_results(self):
        """ Retrieving results """
        return self.results

    def summary(self, get_results=True):
        """ Printing and fetching the results """
        print("RESULTS:")
        print("--------")
        for metric in self.metric_computers.keys():
            if isinstance(self.results[metric], dict):
                result = round(self.results[metric]["mean"], 3)
            else:
                result = round(self.results[metric].item(), 3)
            print(f"  {metric}:  {result}")
        return self.results

    def save_results(self, exp_path, fname):
        """ Storing results into JSON file """
        results_dir = os.path.join(exp_path, "results", fname)
        create_directory(dir_path=results_dir)
        results_file = os.path.join(results_dir, "results.json")

        # converting to list/float and rounding numerical values
        cur_results = {}
        for metric in self.results:
            if self.results[metric] is None:
                continue
            elif isinstance(self.results[metric], dict):
                cur_results[metric] = {}
                cur_results[metric]["mean"] = round(self.results[metric]["mean"], 5)
                cur_results[metric]["framewise"] = []
                for r in self.results[metric]["framewise"].cpu().detach().tolist():
                    cur_results[metric]["framewise"].append(round(r, 5))
            else:
                cur_results[metric] = round(self.results[metric].item(), 5)

        with open(results_file, "w") as file:
            json.dump(cur_results, file)
        return

    def make_plots(self, start_idx=5, savepath=None, prefix="_", **kwargs):
        """ Making and saving plots for each of the framewise results"""
        for metric in self.results:
            if not isinstance(self.results[metric], dict):
                continue
            cur_vals = [round(r, 5) for r in self.results[metric]["framewise"].cpu().detach().tolist()]
            cur_savepath = os.path.join(savepath, f"results_{metric}.png")
            visualize_metric(
                    vals=cur_vals,
                    start_x=start_idx,
                    title=metric,
                    savepath=cur_savepath,
                    xlabel="Frame"
                )
        return

    def _get_metric(self, metric, tb_writer):
        """ """
        if metric == "accuracy":
            metric_computer = Accuracy()
        elif metric == "segmentation_ari":
            metric_computer = SegmentationARI(self.n_instances, tb_writer)
        elif metric == "IoU":
            metric_computer = IoU(self.n_instances)
        elif metric == "mse":
            metric_computer = MSE()
        elif metric == "psnr":
            metric_computer = PSNR()
        elif metric == "ssim":
            metric_computer = SSIM()
        elif metric == "lpips":
            metric_computer = LPIPS()
        else:
            raise NotImplementedError(f"Unknown metric {metric}. Use one of {METRICS} ...")
        return metric_computer


class Metric:
    """
    Base class for metrics
    """

    def __init__(self):
        """ Metric initializer """
        self.results = None
        self.reset()

    def reset(self):
        """ Reseting precomputed metric """
        raise NotImplementedError("Base class does not implement 'reset' functionality")

    def accumulate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'accumulate' functionality")

    def aggregate(self):
        """ """
        raise NotImplementedError("Base class does not implement 'aggregate' functionality")

    def _shape_check(self, tensor, name="Preds"):
        """ """
        if len(tensor.shape) not in [3, 4, 5]:
            raise ValueError(f"{name} has shape {tensor.shape}, but it must have one of the folling shapes\n"
                             " - (B, F, C, H, W) for frame or heatmap prediction.\n"
                             " - (B, F, D) or (B, F, N_joints, N_coords) for pose skeleton prediction")


class Accuracy(Metric):
    """ Accuracy computer """

    def __init__(self):
        """ """
        self.correct = 0
        self.total = 0
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.correct = 0
        self.total = 0

    def accumulate(self, preds, targets):
        """ Computing metric """
        cur_correct = len(torch.where(preds == targets)[0])
        cur_total = len(preds)
        self.correct += cur_correct
        self.total += cur_total

    def aggregate(self):
        """ Computing average accuracy """
        accuracy = self.correct / self.total
        return accuracy


class IoU(Metric):
    """
    Compter for the Intersection over Union
    """

    LOWER_BETTER = False

    def __init__(self, n_instances, tb_writer=None):
        """ """
        self.iou_scores = []
        self.n_instances = n_instances
        self.tb_writer = tb_writer
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.iou_scores = []

    def accumulate(self, preds, targets):
        """
        Computing the IoU metric for quantifying the quality of the reconstructed intance
        segmentation masks.

        Args:
        -----
        preds: torch Tensor
            Reconstructed instance segmentation maps. Shape is (B, n_imgs, 1, H, W)
        targets: torch Tensor / numpy array
            Target instance segmentation maps. Shape is (B, n_imgs, 1, H, W)
        """
        if len(preds.shape) == 4 or len(targets.shape) == 4:
            preds = preds.unsqueeze(1)
            targets = targets.unsqueeze(1)

        self.n_frames = targets.shape[1]
        self.n_instances = len(torch.unique(targets))
        batch_size = targets.shape[0]
        cur_ious = torch.zeros(batch_size, self.n_frames, self.n_instances)
        for i in range(batch_size):
            for f in range(self.n_frames):
                cur_pred, cur_target = targets[i, f], preds[i, f]
                for j in range(self.n_instances):
                    mask_pred = cur_pred == j
                    mask_target = cur_target == j
                    cur_ious[i, f, j] = self._iou(mask_target, mask_pred)
            self.iou_scores.append(cur_ious.mean(dim=-1))
        return

    def _iou(self, mask1, mask2):
        """ Actually computing the IoU between two binary masks """
        union = mask1.sum() + mask2.sum() - (mask1 * mask2).sum()
        iou = (mask1 * mask2).sum() / (union + 1e-12)
        iou = 1. if union == 0. else iou
        return iou

    def aggregate(self):
        """ Computing mean Intersection over Union """
        all_iou_data = torch.cat(self.iou_scores, dim=0)
        mIoU = all_iou_data.mean(dim=-1).mean(dim=-1)
        return mIoU


class SegmentationARI(Metric):
    """
    Segmentation ARI computer
        Adapted from: https://github.com/monniert/dti-sprites/
    """

    LOWER_BETTER = False

    def __init__(self, n_instances, tb_writer=None):
        """ """
        self.ari_scores = []
        self.n_instances = n_instances
        self.tb_writer = tb_writer
        super().__init__()

    def reset(self):
        """ Reseting counters """
        self.ari_scores = []

    def get_best_trial(self, num_trials=3):
        """ Selecting the best result among the last 'n_trials' """
        competing_scores = self.ari_scores[-num_trials:]
        best_score = np.max(competing_scores)
        self.ari_scores = self.ari_scores[:-num_trials] + [best_score]
        return

    def accumulate(self, preds, targets, use_bkg=False):
        """
        Computing the ARI metric for quantifying instance segmentation. We consider
        foreground pixel assignments as cluster assignemnts, and measure a clustering
        evaluation

        Args:
        -----
        pred, gt: torch Tensor / numpy array
            ground truth and predicted instance semgmentations of an image
        use_bkg: boolean
            If True, background is also considered as cluster for evaluation. Otherwise,
            we only use the instance segmentation masks
        """
        self.n_instances = len(torch.unique(targets))  # + 1
        preds_images, targets_images = preds.clone(), targets.clone()
        preds, targets = preds.cpu().flatten(1), targets.cpu().flatten(1)
        batch_size = targets.shape[0]
        for i in range(batch_size):
            t, p = targets[i], preds[i]
            ari_value = self.cpu_ari(t, p, use_bkg)
            if self.tb_writer is not None:
                visualize_ari(
                        tb_writer=self.tb_writer,
                        pred=preds_images[i],
                        target=targets_images[i],
                        score=ari_value,
                        step=len(self.ari_scores)
                    )
            self.ari_scores.append(ari_value)
        return

    def cpu_ari(self, label_true, label_pred, use_bkg=False):
        label_true, label_pred = label_true.flatten(), label_pred.flatten()
        if not use_bkg:
            # we remove background gt pixels from the computation
            good_idx = label_true != 0
            label_true, label_pred = label_true[good_idx], label_pred[good_idx]
        confusion_matrix = self._fast_hist(label_true, label_pred)
        sum_comb_c = sum(_comb2(n_c) for n_c in np.ravel(confusion_matrix.sum(axis=1)))
        sum_comb_k = sum(_comb2(n_k) for n_k in np.ravel(confusion_matrix.sum(axis=0)))
        sum_comb_table = sum(_comb2(n_ij) for n_ij in confusion_matrix.flatten())
        sum_comb_n = _comb2(confusion_matrix.sum())
        if (sum_comb_c == sum_comb_k == sum_comb_n == sum_comb_table):
            return 1.0
        else:
            prod_comb = (sum_comb_c * sum_comb_k) / sum_comb_n
            mean_comb = (sum_comb_k + sum_comb_c) / 2.
            return (sum_comb_table - prod_comb) / (mean_comb - prod_comb)

    def _fast_hist(self, label_true, label_pred):
        try:
            hist = np.bincount(
                    self.n_instances * label_true + label_pred,
                    minlength=self.n_instances ** 2
                ).reshape(self.n_instances, self.n_instances)
        except ValueError:
            val = np.unique(label_true)[1:]
            new_label = np.zeros(label_true.shape, dtype=np.uint8)
            for k, v in enumerate(val):
                new_label[label_true == v] = k+1
            hist = np.bincount(self.n_instances * new_label + label_pred.numpy(),
                               minlength=self.n_instances ** 2).reshape(self.n_instances, self.n_instances)
        return hist

    def aggregate(self):
        """ Computing average ARI """
        return torch.tensor(sum(self.ari_scores) / len(self.ari_scores))


class MSE(Metric):
    """ Mean Squared Error computer """

    LOWER_BETTER = True

    def __init__(self):
        """ """
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_mse = (preds.float() - targets.float()).pow(2).mean(dim=(-1, -2, -3))
        cur_mse = cur_mse.view(B, F)
        self.values.append(cur_mse)
        return cur_mse.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values


class PSNR(Metric):
    """ Peak Signal-to-Noise ratio computer """

    LOWER_BETTER = False

    def __init__(self):
        """ """
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_psnr = piqa.psnr.psnr(preds, targets)
        cur_psnr = cur_psnr.view(B, F)
        self.values.append(cur_psnr)
        return cur_psnr.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values


class SSIM(Metric):
    """ Structural Similarity computer """

    LOWER_BETTER = False

    def __init__(self, window_size=11, sigma=1.5, n_channels=3):
        """ """
        self.ssim = piqa.ssim.SSIM(
                window_size=window_size,
                sigma=sigma,
                n_channels=n_channels,
                reduction=None
            )
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if self.ssim.kernel.device != preds.device:
            self.ssim = self.ssim.to(preds.device)

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_ssim = self.ssim(preds, targets)
        cur_ssim = cur_ssim.view(B, F)
        self.values.append(cur_ssim)
        return cur_ssim.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values


class LPIPS(Metric):
    """ Learned Perceptual Image Patch Similarity computers """

    LOWER_BETTER = True

    def __init__(self, network="alex", pretrained=True, reduction=None):
        """ """
        self.lpips = piqa.lpips.LPIPS(
                network=network,
                pretrained=pretrained,
                reduction=reduction
            )
        super().__init__()
        self.values = []

    def reset(self):
        """ Reseting counters """
        self.values = []

    def accumulate(self, preds, targets):
        """ Computing metric """
        self._shape_check(tensor=preds, name="Preds")
        self._shape_check(tensor=targets, name="Targets")
        if not hasattr(self.lpips, "device"):
            self.lpips = self.lpips.to(preds.device)
            self.lpips.device = preds.device

        B, F, C, H, W = preds.shape
        preds, targets = preds.view(B * F, C, H, W), targets.view(B * F, C, H, W)
        cur_lpips = self.lpips(preds, targets)
        cur_lpips = cur_lpips.view(B, F)
        self.values.append(cur_lpips)
        return cur_lpips.mean()

    def aggregate(self):
        """ Computing average metric, both global and framewise"""
        all_values = torch.cat(self.values, dim=0)
        mean_values = all_values.mean()
        frame_values = all_values.mean(dim=0)
        return float(mean_values), frame_values


#############
#   UTILS   #
#############


def calculate_iou(mask1, mask2):
    """
    Calculate IoU of two segmentation masks.
    https://github.com/ecker-lab/object-centric-representation-benchmark
    Args:
    -----
        mask1: HxW
        mask2: HxW
    """
    eps = np.finfo(float).eps
    mask1 = np.float32(mask1)
    mask2 = np.float32(mask2)
    union = ((np.sum(mask1) + np.sum(mask2) - np.sum(mask1*mask2)))
    iou = np.sum(mask1*mask2) / (union + eps)
    iou = 1. if union == 0. else iou
    return iou


def compute_mot_metrics(acc, summary):
    """
    Computing MOT metrics
    https://github.com/ecker-lab/object-centric-representation-benchmark

    Args:
    -----
    acc: motmetric accumulator
    summary: pandas dataframe with mometrics summary
    """

    df = acc.mot_events
    df = df[(df.Type != 'RAW')
            & (df.Type != 'MIGRATE')
            & (df.Type != 'TRANSFER')
            & (df.Type != 'ASCEND')]
    obj_freq = df.OId.value_counts()
    n_objs = len(obj_freq)
    tracked = df[df.Type == 'MATCH']['OId'].value_counts()
    detected = df[df.Type != 'MISS']['OId'].value_counts()

    track_ratios = tracked.div(obj_freq).fillna(0.)
    detect_ratios = detected.div(obj_freq).fillna(0.)

    summary['mostly_tracked'] = track_ratios[track_ratios >= 0.8].count() / n_objs * 100
    summary['mostly_detected'] = detect_ratios[detect_ratios >= 0.8].count() / n_objs * 100

    n = summary['num_objects'][0]
    summary['num_matches'] = (summary['num_matches'][0] / n * 100)
    summary['num_false_positives'] = (summary['num_false_positives'][0] / n * 100)
    summary['num_switches'] = (summary['num_switches'][0] / n * 100)
    summary['num_misses'] = (summary['num_misses'][0] / n * 100)

    summary['mota'] = (summary['mota'][0] * 100)
    summary['motp'] = ((1. - summary['motp'][0]) * 100)

    return summary


def compute_dists_per_frame(gt_frame, pred_frame):
    """ Compute pairwise distances between gt objects and predictions per frame """
    IOU_THR = 0.5

    n_pred = pred_frame.shape[0]
    n_gt = gt_frame.shape[0]

    dists = np.ones((n_gt, n_pred))
    for h in range(n_gt):
        for j in range(n_pred):
            mask_gt = gt_frame == h
            mask_pred = pred_frame == j
            dists[h, j] = calculate_iou(mask_gt, mask_pred)

    dists = 1. - dists
    dists[dists > IOU_THR] = np.nan

    return dists, torch.unique(gt_frame), torch.unique(pred_frame)


def _comb2(n):
    # the exact version is faster for k == 2: use it by default globally in
    # this module instead of the float approximate variant
    return comb(n, 2, exact=1)

#
