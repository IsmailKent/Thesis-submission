"""
Generating some figures using a pretrained SAVi model and the corresponding predictor
"""

import os
from tqdm import tqdm
import numpy as np
import torch

from base.baseFigGenerator import BaseFigGenerator

from lib.arguments import get_predictor_training_arguments
from lib.logger import print_
from lib.metrics import MetricTracker
import lib.utils as utils
from lib.visualizations import add_border, make_gif, visualize_qualitative_eval, visualize_aligned_slots, \
    visualize_tight_row, masks_to_rgb, idx_to_one_hot, overlay_segmentations, COLORS


# GLOBALS    TODO: make these command line arguments
MODE = "RANDOM_"
NUM_CONTEXT = 5
NUM_PREDS = 25
NUM_SEQS = 50


class FigGenerator(BaseFigGenerator):
    """
    Class for generating some figures using a pretrained object-centric video prediction model
    """

    def __init__(self, exp_path, sa_model_directory, checkpoint, name_predictor_experiment, num_seqs=NUM_SEQS):
        """
        Initializing the trainer object
        """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, name_predictor_experiment)
        super().__init__(
                exp_path=self.exp_path,
                sa_model_directory=sa_model_directory,
                num_seqs=num_seqs
            )
        self.checkpoint = checkpoint
        self.name_predictor_experiment = name_predictor_experiment

        self.pred_name = self.exp_params["model"]["predictor"]["predictor_name"]
        self.plots_path = os.path.join(
                self.exp_path,
                "plots",
                f"figGeneration_pred_{self.pred_name}_{name_predictor_experiment}",
                f"{checkpoint[:-4]}_NumPreds={NUM_PREDS}"
            )
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.plots_path)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        if NUM_CONTEXT is not None and NUM_PREDS is not None:
            self.exp_params["training_prediction"]["num_context"] = NUM_CONTEXT
            self.exp_params["training_prediction"]["num_preds"] = NUM_PREDS
            self.exp_params["training_prediction"]["sample_length"] = NUM_CONTEXT + NUM_PREDS
        super().load_data()
        return

    def unwrap_batch_data(self, batch_data):
        """
        Unwrapping the batch data depending on the dataset that we are training on
        """
        initializer_kwargs = {}
        if self.exp_params["dataset"]["dataset_name"] in ["VMDS", "Sketchy"]:
            videos, _ = batch_data
        elif self.exp_params["dataset"]["dataset_name"] == "SynpickVP":
            videos, all_reps = batch_data
            initializer_kwargs["instance_masks"] = all_reps["masks"]
            initializer_kwargs["com_coords"] = all_reps["com_coords"]
            initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
        else:
            videos = batch_data
        return videos, initializer_kwargs

    @torch.no_grad()
    def generate_figs(self):
        """
        Evaluating model epoch loop
        """
        utils.set_random_seed()
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        video_length = self.exp_params["training_prediction"]["sample_length"]

        metric_tracker = MetricTracker(exp_path=None, metrics=["psnr", "lpips"], use_tboard=False)

        for i in tqdm(range(self.num_seqs)):
            idx = np.random.randint(0, len(self.test_set)) if MODE == "RANDOM" else i
            batch_data = self.test_set[idx]
            videos, initializer_data = self.unwrap_batch_data(batch_data)
            videos = videos.unsqueeze(0).to(self.device)
            initializer_data = {k: v.unsqueeze(0) for k, v in initializer_data.items() if torch.is_tensor(v)}
            B, L, C, H, W = videos.shape
            if L < num_context + num_preds:
                raise ValueError(f"Seq. length {L} smaller that #seed {num_context} + #preds {num_preds}")

            if "Transformer" in self.pred_name:
                forward_func = self.transformer_forward_pass
            elif self.pred_name == "LSTM":
                forward_func = self.lstm_forward_pass
            else:
                raise ValueError(f"Unknown {self.pred_name = }")
            pred_results = forward_func(
                    videos=videos,
                    initializer_data=initializer_data,
                    num_context=num_context,
                    num_preds=num_preds,
                    video_length=video_length
                )
            pred_imgs, pred_recons, pred_masks, individual_recons_history, masks_history = pred_results

            # computing metrics for sequence to visualize
            metric_tracker.reset_results()
            metric_tracker.accumulate(
                    preds=pred_imgs.clamp(0, 1),
                    targets=videos[:1, num_context:num_context+num_preds].clamp(0, 1)
                )
            metric_tracker.aggregate()
            results = metric_tracker.get_results()
            psnr, lpips = results["psnr"]["mean"], results["lpips"]["mean"]
            cur_dir = f"img_{i+1}_psnr={round(psnr,2)}_lpips={round(lpips, 3)}"

            # generating and saving visualizations
            self.compute_visualization(
                    videos=videos,
                    pred_imgs=pred_imgs,
                    individual_recons_history=individual_recons_history,
                    masks_history=masks_history,
                    pred_objs=pred_recons,
                    pred_masks=pred_masks,
                    seed_objs=individual_recons_history,
                    seed_masks=masks_history,
                    img_idx=i,
                    cur_dir=cur_dir
                )
        return

    def transformer_forward_pass(self, videos, initializer_data, num_context, num_preds, video_length):
        """
        Computing predictions using one of the transformer models
        """
        B, L, C, H, W = videos.shape
        num_slots = self.model.num_slots
        slot_dim = self.model.slot_dim

        # computing seed slots using pretrained SAVi encoder
        with torch.no_grad():
            out_savi = self.model(videos, num_imgs=video_length, **initializer_data)
            slot_history, recons_history, individual_recons_history, masks_history = out_savi

        # predicting future slots using the corresponding predictor module
        pred_slots = []
        predictor_input = slot_history[:, :num_context].clone()
        for t in range(num_preds):
            cur_preds = self.predictor(predictor_input)[:, -1]
            next_input = cur_preds  # no teacher forcing at validation/test
            predictor_input = torch.cat([predictor_input[:, 1:], next_input.unsqueeze(1)], dim=1)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B,num_slots, num_preds, slot_dim)

        # decoding slots into objects and images
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.model.decode(pred_slots_decode)
        pred_imgs = img_recons.view(B, num_preds, C, H, W)

        return pred_imgs, pred_recons, pred_masks, individual_recons_history, masks_history

    def lstm_forward_pass(self, videos, initializer_data, num_context, num_preds, video_length):
        """
        Computing predictions using the LSTM model
        """
        B, L, C, H, W = videos.shape
        num_slots = self.model.num_slots
        slot_dim = self.model.slot_dim

        # computing seed slots using pretrained SAVi encoder
        with torch.no_grad():
            out_savi = self.model(videos, num_imgs=video_length, **initializer_data)
            slot_history, recons_history, individual_recons_history, masks_history = out_savi

        # predicting future slots using the corresponding predictor module
        pred_slots = []

        # reshaping slots: (B, L, num_slots, slot_dim) --> (B * num_slots, L, slot_dim)
        slot_history_lstm_input = slot_history.permute(0, 2, 1, 3)
        slot_history_lstm_input = slot_history_lstm_input.reshape(B * num_slots, L, slot_dim)

        # using seed images to initialize the RNN predictor
        self.predictor.init_hidden(b_size=B * num_slots, device=self.device)
        for t in range(num_context-1):
            _ = self.predictor(slot_history_lstm_input[:, t])

        # Autoregressively predicting the future slots
        next_input = slot_history_lstm_input[:, num_context-1]
        for t in range(num_preds):
            cur_preds = self.predictor(next_input)
            next_input = cur_preds  # no teacher forcing at validation/test
            pred_slots.append(cur_preds)

        # reconvering original slot shape
        pred_slots = torch.stack(pred_slots, dim=1)  # (B * num_slots, num_preds, slot_dim)
        pred_slots = pred_slots.reshape(B, num_slots, num_preds, slot_dim).permute(0, 2, 1, 3)

        # decoding slots into objects and images
        pred_slots_decode = pred_slots.clone().reshape(B * num_preds, num_slots, slot_dim)
        img_recons, (pred_recons, pred_masks) = self.model.decode(pred_slots_decode)
        pred_imgs = img_recons.view(B, num_preds, C, H, W)
        pred_masks = pred_masks.view(B, num_preds, num_slots, 1, H, W)

        return pred_imgs, pred_recons, pred_masks, individual_recons_history, masks_history

    def compute_visualization(self, videos, pred_imgs, pred_objs, pred_masks,
                              seed_objs, seed_masks, img_idx, **kwargs):
        """
        Making a forwad pass through the model and computing the evaluation metrics

        Args:
        -----
        videos: torch Tensor
            Videos sequence from the dataset, containing the seed and target frames.
            Shape is (B, num_frames, C, H, W)
        pred_imgs: torch Tensor
            Predicted video frames. Shape is (B, num_preds, C, H, W)
        pred_objs: torch Tensor
            Predicted objects corresponding to the predicted video frames.
            Shape is (B, num_preds, num_objs, C, H, W)
        pred_masks: torch Tensor
            Predicted object masks corresponding to the objects in the predicted video frames.
            Shape is (B, num_preds, num_objs, 1, H, W)
        seed_objs: torch Tensor
            Objects extracted by the SAVi model from the seed video frames.
            Shape is (B, num_seed, num_objs, C, H, W)
        seed_masks: torch Tensor
            Masks extracted by the SAVi model from the seed video frames.
            Shape is (B, num_seed, num_objs, 1, H, W)
        img_idx: int
            Index of the visualization to compute and save
        """
        cur_dir = kwargs.get("cur_dir", f"img_{img_idx+1}")
        utils.create_directory(self.plots_path, cur_dir)

        # some hpyer-parameters of the video model
        B = videos.shape[0]
        num_slots = self.model.num_slots
        num_context = self.exp_params["training_prediction"]["num_context"]
        num_preds = self.exp_params["training_prediction"]["num_preds"]
        seed_imgs = videos[:, :num_context, :, :]
        target_imgs = videos[:, num_context:num_context+num_preds, :, :]

        # aligned objects (seed and pred)
        seed_objs = add_border(seed_objs * seed_masks, color_name="green", pad=2)[:, :num_context]
        pred_objs = add_border(pred_objs * pred_masks, color_name="red", pad=2)
        pred_objs = pred_objs.reshape(B, num_preds, num_slots, *pred_objs.shape[-3:])
        all_objs = torch.cat([seed_objs, pred_objs], dim=1)
        _ = visualize_aligned_slots(
                all_objs[0],
                savepath=os.path.join(self.plots_path, cur_dir, "aligned_slots.png")
            )

        # Video predictions
        fig, ax = visualize_qualitative_eval(
                context=seed_imgs[0],
                targets=target_imgs[0],
                preds=pred_imgs[0],
                savepath=os.path.join(self.plots_path, cur_dir, "qual_eval.png")
            )
        fig, ax = visualize_tight_row(
                frames=videos[0],
                num_context=num_context,
                is_gt=True,
                savepath=os.path.join(self.plots_path, cur_dir, "row_rgb_gt.png")
            )
        fig, ax = visualize_tight_row(
                frames=pred_imgs[0],
                num_context=num_context,
                is_gt=False,
                savepath=os.path.join(self.plots_path, cur_dir, "row_rgb_pred.png")
            )

        # masks
        seed_masks_categorical = seed_masks[0, :num_context].argmax(dim=1)
        if len(pred_masks.shape) > 4:
            pred_masks = pred_masks[0]
        pred_masks_categorical = pred_masks.argmax(dim=1)
        all_masks_categorical = torch.cat([seed_masks_categorical, pred_masks_categorical], dim=0)
        masks_vis = masks_to_rgb(x=all_masks_categorical)[:, 0]
        fig, ax = visualize_tight_row(
                frames=masks_vis,
                num_context=num_context,
                is_gt=True,
                savepath=os.path.join(self.plots_path, cur_dir, "row_masks.png")
            )

        # overlay masks
        masks_categorical_channels = idx_to_one_hot(x=all_masks_categorical[:, 0])
        disp_overlay = overlay_segmentations(
            videos[0].cpu().detach(),
            masks_categorical_channels.cpu().detach(),
            colors=COLORS,
            alpha=0.6
        )
        fig, ax = visualize_tight_row(
                frames=disp_overlay,
                num_context=num_context,
                is_gt=True,
                savepath=os.path.join(self.plots_path, cur_dir, "row_overlay.png")
            )

        # Sequence GIFs
        gt_frames = torch.cat([seed_imgs, target_imgs], dim=1)
        pred_frames = torch.cat([seed_imgs, pred_imgs], dim=1)
        make_gif(
                gt_frames[0],
                savepath=os.path.join(self.plots_path, cur_dir, "gt_GIF_frames.gif"),
                n_seed=1000,
                use_border=True
            )
        make_gif(
                pred_frames[0],
                savepath=os.path.join(self.plots_path, cur_dir, "pred_GIF_frames.gif"),
                n_seed=num_context,
                use_border=True
            )
        make_gif(
                masks_vis,
                savepath=os.path.join(self.plots_path, cur_dir, "masks_GIF_masks.gif"),
                n_seed=num_context,
                use_border=True
            )
        make_gif(
                disp_overlay,
                savepath=os.path.join(self.plots_path, cur_dir, "overlay_GIF.gif"),
                n_seed=num_context,
                use_border=True
            )

        # Object GIFs
        for obj_id in range(all_objs.shape[2]):
            make_gif(
                    all_objs[0, :, obj_id],
                    savepath=os.path.join(self.plots_path, cur_dir, f"gt_obj_{obj_id+1}.gif"),
                    n_seed=num_context,
                    use_border=True
                )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, sa_model_directory, checkpoint, name_predictor_experiment, args = get_predictor_training_arguments()
    print_("Generating figures for predictor model...")
    figGenerator = FigGenerator(
            exp_path=exp_path,
            sa_model_directory=sa_model_directory,
            checkpoint=args.checkpoint,
            name_predictor_experiment=name_predictor_experiment,
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up model and loading pretrained parameters")
    figGenerator.load_model(exp_path=figGenerator.parent_exp_path)
    print_("Setting up predictor and loading pretrained parameters")
    figGenerator.load_predictor()
    print_("Generating and saving figures")
    figGenerator.generate_figs()


#
