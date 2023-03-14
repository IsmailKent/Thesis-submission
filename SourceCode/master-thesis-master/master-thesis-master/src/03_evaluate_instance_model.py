"""
Evaluating a SAVI model checkpoint using object-centric metrics
"""

from data.load_data import unwrap_batch_data
from lib.arguments import get_sa_eval_arguments
from lib.logger import Logger, print_, log_function, for_all_methods
from lib.metrics import MetricTracker
import lib.utils as utils

from base.baseEvaluator import BaseEvaluator


@for_all_methods(log_function)
class Evaluator(BaseEvaluator):
    """
    Class for evaluating a SAVI model using instace-centric metrics
    """

    def set_metric_tracker(self):
        """ Initializing the metric tracker """
        self.metric_tracker = MetricTracker(
                exp_path=exp_path,
                metrics=["segmentation_ari", "IoU"],
                use_tboard=False
            )
        return

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
        instances, targets, initializer_kwargs = unwrap_batch_data(batch_data)
        instances = instances.to(self.device).float()

        # forward pass
        out_model = self.model(instances, num_imgs=instances.shape[1], **initializer_kwargs)
        slot_history, recons_history = out_model

        # computing evaluation metrics
        recons_masks = self.test_loader.dataset._one_hot_to_instances(recons_history)
        target_masks = self.test_loader.dataset._one_hot_to_instances(targets)
        self.metric_tracker.accumulate(preds=recons_masks, targets=target_masks)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_sa_eval_arguments()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting instance-centric evaluation procedure", message_type="new_exp")

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
