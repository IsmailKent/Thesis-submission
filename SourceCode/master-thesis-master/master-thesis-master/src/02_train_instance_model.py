"""
Training and Validation a RecursiveInstanceEncoder model
"""

import os
import torch

from data.load_data import unwrap_batch_data
from lib.arguments import get_directory_argument
from lib.logger import Logger, print_
import lib.utils as utils
from lib.visualizations import visualize_decomp, visualize_recons, one_hot_to_instances, instances_to_rgb

from base.baseTrainer import BaseTrainer


class Trainer(BaseTrainer):
    """ Class for training a SAVi model """

    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):
        """
        Computing a forwad pass through the model, and (if necessary) the loss values and metrics

        Args:
        -----
        batch_data: dict
            Dictionary containing the information for the current batch, including images, poses,
            actions, or metadata, among others.
        training: bool
            If True, model is in training mode
        inference_only: bool
            If True, only forward pass through the model is performed

        Returns:
        --------
        pred_data: dict
            Predictions from the model for the current batch of data
        loss: torch.Tensor
            Total loss for the current batch
        """
        instances, targets, initializer_kwargs = unwrap_batch_data(batch_data)
        instances = instances.to(self.device).float()

        # forward pass
        out_model = self.model(instances, num_imgs=instances.shape[1], **initializer_kwargs)
        slot_history, recons_history = out_model
        # recons_history = recons_history.view(B, num_objs, num_frames, C, H, W).transpose(1, 2)

        if inference_only:
            return out_model, None

        # if necessary, doing loss computation, backward pass, optimization, and computing metrics
        self.loss_tracker(
                pred_imgs=recons_history.clamp(0, 1),
                target_imgs=targets.clamp(0, 1)
            )

        loss = self.loss_tracker.get_last_losses(total_only=True)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training_slots"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.exp_params["training_slots"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss

    def visualizations(self, batch_data, epoch, iter_):
        """
        Making a visualization of some ground-truth, targets and predictions from the current model.
        """
        if(iter_ % self.exp_params["training_slots"]["image_log_frequency"] != 0):
            return

        instances, initializer_kwargs = self.unwrap_batch_data(batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        slot_history, recons_history = out_model

        # converting from masks into colored
        num_objects = instances.shape[2]
        instances_merged = one_hot_to_instances(instances).squeeze(2)
        instances_rgb = instances_to_rgb(instances_merged, num_channels=num_objects).clamp(0, 1)
        recons_instances_merged = one_hot_to_instances(recons_history).squeeze(2)
        recons_rgb = instances_to_rgb(recons_instances_merged, num_channels=num_objects).clamp(0, 1)

        N = min(10, instances.shape[1])
        savepath = os.path.join(self.plots_path, f"recons_epoch_{epoch}_iter_{iter_}.png")
        savepath = None
        visualize_recons(
                imgs=instances_rgb[0][:N],
                recons=recons_rgb[0][:N].clamp(0, 1),
                savepath=savepath,
                tb_writer=self.writer,
                iter=iter_
            )
        savepath = os.path.join(self.plots_path, f"masks_epoch_{epoch}_iter_{iter_}.png")
        savepath = None
        _ = visualize_decomp(
                recons_history[0][:N].clamp(0, 1),
                savepath=savepath,
                tag="masks",
                cmap="gray",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_
            )
        savepath = os.path.join(self.plots_path, f"gt_masks_epoch_{epoch}_iter_{iter_}.png")
        savepath = None
        _ = visualize_decomp(
                instances[0][:N].clamp(0, 1),
                savepath=savepath,
                tag="gt masks",
                cmap="gray",
                vmin=0,
                vmax=1,
                tb_writer=self.writer,
                iter=iter_
            )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting training procedure", message_type="new_exp")

    print_("Initializing Trainer...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    trainer = Trainer(
            exp_path=exp_path,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training
        )
    print_("Loading dataset...")
    trainer.load_data()
    print_("Setting up model and optimizer")
    trainer.setup_model()
    print_("Starting to train")
    trainer.training_loop()


#
