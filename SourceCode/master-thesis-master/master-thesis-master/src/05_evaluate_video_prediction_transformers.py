"""
Evaluating a transformer predictor model checkpoint
"""

import torch

from lib.arguments import get_predictor_evaluation_arguments
from lib.logger import Logger, print_
from lib.metrics import MetricTracker
import lib.utils as utils

from base.basePredictorEvaluator import BasePredictorEvaluator


class Evaluator(BasePredictorEvaluator):
    """
    Class for evaluating a transformer predictor model checkpoint
    """

    def set_metric_tracker(self):
        """ Initializing the metric tracker """
        model_name = self.exp_params["model"]["model_name"]
        num_slots = self.exp_params["model"][model_name]["num_slots"]
        self.metric_tracker = MetricTracker(
                exp_path,
                metrics=["psnr", "ssim", "lpips"],
                num_slots=num_slots
            )
        return

    def unwrap_batch_data(self, batch_data):
        """
        Unwrapping the batch data depending on the dataset that we are training on
        """
        initializer_kwargs = {}
        if self.exp_params["dataset"]["dataset_name"] in ["VMDS", "Sketchy"]:
            videos, _ = batch_data
        elif self.exp_params["dataset"]["dataset_name"] in ["SynpickVP", "MoviA", "MoviB", "MoviC"]:
            videos, all_reps = batch_data
            initializer_kwargs["instance_masks"] = all_reps["masks"]
            initializer_kwargs["com_coords"] = all_reps["com_coords"]
            initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
        else:
            videos = batch_data
        return videos, initializer_kwargs

    @torch.no_grad()
    def forward_eval(self, batch_data, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        """
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        skip_first_slot = self.exp_params["training_prediction"]["skip_first_slot"]
        video_length = self.exp_params["training_prediction"]["sample_length"]
        num_slots = self.model.num_slots
        slot_dim = self.model.slot_dim

        videos, initializer_data = self.unwrap_batch_data(batch_data)
        videos = videos.to(self.device)
        B, L, C, H, W = videos.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq. length {L} smaller that #seed {num_context} + #preds {num_preds}")

        # encoding images into object-centric slots, and temporally aligning slots
        with torch.no_grad():
            out_model = self.model(videos, num_imgs=video_length, **initializer_data)
            slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        # Autoregressively predicting the future slots
        pred_slots = []
        first_slot_idx = 1 if skip_first_slot else 0
        predictor_input = slot_history[:, first_slot_idx:num_context].clone()
        for t in range(num_preds):
            cur_preds = self.predictor(predictor_input)[:, -1]  # get predicted slots at last time step
            next_input = cur_preds
            predictor_input = torch.cat([predictor_input[:, 1:], next_input.unsqueeze(1)], dim=1)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)

        # decoding predicted slots into predicted frames
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.model.decode(pred_slots_decode)
        pred_imgs = img_recons.view(B, num_preds, C, H, W)
        target_imgs = videos[:, num_context:num_context+num_preds, :, :]

        # computing evaluation metrics
        self.metric_tracker.accumulate(
                preds=pred_imgs.clamp(0, 1),
                targets=target_imgs.clamp(0, 1),
            )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    all_args = get_predictor_evaluation_arguments()
    exp_path, sa_model_directory, checkpoint, name_predictor_experiment, args = all_args

    logger = Logger(exp_path=f"{exp_path}/{name_predictor_experiment}/evaluation")
    logger.log_info("Starting video prediction evaluation procedure", message_type="new_exp")
    print_("Initializing Evaluator...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")

    evaluator = Evaluator(
            name_predictor_experiment=name_predictor_experiment,
            exp_path=exp_path,
            sa_model_directory=sa_model_directory,
            checkpoint=args.checkpoint,
            num_preds=args.num_preds
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and predictor and loading pretrained parameters")
    evaluator.load_model()
    evaluator.setup_predictor()
    print_("Starting evaluation")
    evaluator.evaluate()


#