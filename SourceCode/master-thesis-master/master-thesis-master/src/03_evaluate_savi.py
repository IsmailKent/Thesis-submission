"""
Evaluating a SAVI model checkpoint using object-centric metrics
"""

import torch

from lib.arguments import get_sa_eval_arguments
from lib.logger import Logger, print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.utils as utils

from base.baseEvaluator import BaseEvaluator


@for_all_methods(log_function)
class Evaluator(BaseEvaluator):
    """
    Class for evaluating a SAVI model using object-centric metrics
    """

    def set_metric_tracker(self):
        """ Initializing the metric tracker """
        self.metric_tracker = MetricTracker(
                exp_path,
                metrics=["segmentation_ari", "IoU"],
                num_slots=self.exp_params["model"]["SAVi"]["num_slots"],
                use_tboard=False
            )
        return

    def load_data(self):
        """ Loading data """
        super().load_data()
        self.test_set.get_masks = True
        return

    def unwrap_batch_data(self, batch_data):
        """
        Unwrapping the batch data depending on the dataset that we are training on
        """
        dbs = ["VMDS", "SynpickVP", "MoviA", "MoviB", "MoviC"]
        dataset_name = self.exp_params["dataset"]["dataset_name"]
        initializer_kwargs = {}
        if dataset_name == "VMDS":
            videos, masks = batch_data
        elif dataset_name in ["SynpickVP", "MoviA", "MoviB", "MoviC"]:
            videos, all_reps = batch_data
            masks = all_reps["masks"]
            initializer_kwargs["instance_masks"] = all_reps["masks"]
            initializer_kwargs["com_coords"] = all_reps["com_coords"]
            initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
        else:
            raise ValueError(f"Only {dbs} support object evaluation. Given {dataset_name = }")
        return videos, masks, initializer_kwargs

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
        videos, masks, initializer_kwargs = self.unwrap_batch_data(batch_data)
        videos, masks = videos.to(self.device), masks.to(self.device)
        num_slots = len(torch.unique(masks)) if self.exact_slots else self.model.num_slots

        # forward pass
        out_model = self.model(
                videos,
                num_imgs=videos.shape[1],
                num_slots=num_slots,
                **initializer_kwargs
            )
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        # computing evaluation metrics
        combined_masks = torch.argmax(masks_history, dim=2).squeeze(2)
        self.metric_tracker.accumulate(
                preds=combined_masks,
                targets=masks
            )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_sa_eval_arguments()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting object-cetric evaluation procedure", message_type="new_exp")

    print_("Initializing Evaluator...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    evaluator = Evaluator(
            exp_path=exp_path,
            checkpoint=args.checkpoint,
            exact_slots=args.exact_slots
        )
    print_("Loading dataset...")
    evaluator.load_data()
    print_("Setting up model and loading pretrained parameters")
    evaluator.setup_model()
    print_("Starting evaluation")
    evaluator.evaluate()


#
