"""
Training and Validation of a transformer-based predictor module using a frozen and pretrained
SAVI model.
This script is used for the instance-based models
"""

import os
import torch

from data.load_data import unwrap_batch_data
from lib.arguments import get_predictor_training_arguments
from lib.logger import Logger, print_
import lib.utils as utils
import lib.visualizations as visualizations

from base.basePredictorTrainer import BasePredictorTrainer


class Trainer(BasePredictorTrainer):
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
        predictor_name = self.exp_params["model"]["predictor"]["predictor_name"]
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        teacher_force = self.exp_params["training_prediction"]["teacher_force"]
        skip_first_slot = self.exp_params["training_prediction"]["skip_first_slot"]
        video_length = self.exp_params["training_prediction"]["sample_length"]
        num_slots = self.model.num_slots
        slot_dim = self.model.slot_dim
        if self.predictor.train is False:
            teacher_force = False

        # fetching data
        instances, initializer_kwargs = unwrap_batch_data(batch_data)
        instances = instances.to(self.device).float()
        B, L, N_Objs, C, H, W = instances.shape
        if L < num_context + num_preds:
            raise ValueError(f"Seq. length {L} smaller that #seed {num_context} + #preds {num_preds}")

        # encoding images into object-centric slots
        with torch.no_grad():
            out_model = self.model(instances, num_imgs=video_length, **initializer_kwargs)
            slot_history, recons_history = out_model

        # Autoregressively predicting the future slots
        pred_slots = []
        first_slot_idx = 1 if skip_first_slot else 0
        predictor_input = slot_history[:, first_slot_idx:num_context].clone()

        # Using LSTM predictor
        if predictor_name == "LSTM":
            # reshaping slots: (B, L, num_slots, slot_dim) --> (B * num_slots, L, slot_dim)
            slot_history_lstm_input = predictor_input.permute(0, 2, 1, 3)
            slot_history_lstm_input = slot_history_lstm_input.reshape(B * num_slots, L, slot_dim)

            # using seed images to initialize the RNN predictor
            self.predictor.init_hidden(b_size=B * num_slots, device=self.device)
            for t in range(num_context-1):
                _ = self.predictor(slot_history_lstm_input[:, t])

            # Autoregressively predicting the future slots
            next_input = slot_history_lstm_input[:, num_context-1]
            for t in range(num_preds):
                cur_preds = self.predictor(next_input)
                next_input = slot_history_lstm_input[:, num_context+t] if teacher_force else cur_preds
                pred_slots.append(cur_preds)
            pred_slots = torch.stack(pred_slots, dim=1)  # (B * num_slots, num_preds, slot_dim)
            pred_slots = pred_slots.reshape(B, num_slots, num_preds, slot_dim).permute(0, 2, 1, 3)
        # Using any of the transformer-based predictors
        elif "Transformer" in predictor_name:
            for t in range(num_preds):
                cur_preds = self.predictor(predictor_input)[:, -1]  # get predicted slots at last time step
                next_input = slot_history[:, num_context+t] if teacher_force else cur_preds
                predictor_input = torch.cat([predictor_input[:, 1:], next_input.unsqueeze(1)], dim=1)
                pred_slots.append(cur_preds)
            pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        else:
            raise ValueError(f"Unknown {predictor_name = }...")

        # decoding predicted slots into predicted frames
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        recons_history = self.model.decode(pred_slots_decode)
        pred_imgs = recons_history.view(B, num_preds, N_Objs, C, H, W)
        out_model = (pred_imgs)

        # for generating only model outputs
        if inference_only:
            return out_model, None

        # if necessary, doing loss computation, backward pass, optimization, and computing metrics
        target_slots = slot_history[:, num_context:num_context+num_preds, :, :]
        target_instances = instances[:, num_context:num_context+num_preds, :, :]
        self.loss_tracker(
                preds=pred_slots,
                targets=target_slots,
                pred_imgs=pred_imgs,
                target_imgs=target_instances
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

    def visualizations(self, batch_data, epoch):
        """
        Making a visualization of some ground-truth, targets and predictions from the current model.
        """
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]

        # forward pass
        instances, targets, initializer_kwargs = self.unwrap_batch_data(batch_data)
        out_model, _ = self.forward_loss_metric(
                batch_data=batch_data,
                training=False,
                inference_only=True
            )
        pred_instances = out_model

        # some postprocessing for visualization
        N = min(10, instances.shape[1])
        num_objects = instances.shape[2]
        instances_merged = visualizations.one_hot_to_instances(instances).squeeze(2)
        pred_instances_merged = visualizations.one_hot_to_instances(pred_instances).squeeze(2)
        instances_rgb = visualizations.instances_to_rgb(
                instances_merged,
                num_channels=num_objects
            ).clamp(0, 1)
        pred_rgb = visualizations.instances_to_rgb(
                pred_instances_merged,
                num_channels=num_objects
            ).clamp(0, 1)
        seed_instances_rgb = instances_rgb[:, :num_context, :, :]
        target_instances_rgb = instances_rgb[:, num_context:num_context+num_preds, :, :]

        # visualizations
        ids = torch.linspace(0, instances.shape[0]-1, 3).round().int()  # equispaced videos in batch
        for idx in range(3):
            k = ids[idx]

            # qualititative evalution: seed/tagets/preds
            savepath = os.path.join(self.plots_path, f"PredInstances_epoch_{epoch}_{k}.png")
            fig, ax = visualizations.visualize_qualitative_eval(
                context=seed_instances_rgb[k],
                targets=target_instances_rgb[k],
                preds=pred_rgb[k],
                savepath=savepath
            )
            self.writer.add_figure(tag=f"Qualitative Eval {k+1}", figure=fig, step=epoch + 1)

            # predicted instance masks
            savepath = os.path.join(self.plots_path, f"Recons_Masks_{k+1}.png")
            fig, ax = visualizations.visualize_decomp(
                    pred_instances[k][:N],
                    savepath=savepath,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )
            self.writer.add_figure(tag=f"Pred Instance Masks {k+1}", figure=fig, step=epoch + 1)

            # ground-truth instance masks
            savepath = os.path.join(self.plots_path, f"GT_Masks_{k+1}.png")
            fig, ax = visualizations.visualize_decomp(
                    instances[0, :N, num_context:num_context+num_preds],
                    savepath=savepath,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )
            self.writer.add_figure(tag=f"GT Instance Masks {k+1}", figure=fig, step=epoch + 1)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, sa_model_directory, checkpoint, name_predictor_experiment, args = get_predictor_training_arguments()
    logger = Logger(exp_path=f"{exp_path}/{name_predictor_experiment}")
    logger.log_info("Starting transformer predictor training procedure", message_type="new_exp")

    print_("Initializing Trainer...")
    print_("Args:")
    print_("-----")
    for k, v in vars(args).items():
        print_(f"  --> {k} = {v}")
    trainer = Trainer(
            name_predictor_experiment=name_predictor_experiment,
            exp_path=exp_path,
            sa_model_directory=sa_model_directory,
            checkpoint=args.checkpoint,
            resume_training=args.resume_training
        )
    print_("Loading dataset...")
    trainer.load_data()
    print_("Setting up model, predictor and optimizer")
    trainer.load_model()
    trainer.setup_predictor()
    print_("Starting to train")
    trainer.training_loop()


#
