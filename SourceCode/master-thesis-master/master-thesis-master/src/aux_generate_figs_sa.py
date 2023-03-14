"""
Generating some figures using a pretrained slot attention model
"""


import os
import torch

from base.baseFigGenerator import BaseFigGenerator

from lib.arguments import get_test_align_arguments
from lib.logger import print_
import lib.utils as utils
from lib.visualizations import visualize_evaluation_slots, visualize_recons


class FigGenerator(BaseFigGenerator):
    """
    Class for generating figures using a pretrained SAVI model
    """

    def __init__(self, exp_path, sa_model_directory, num_seqs=10):
        """ Initializing the figure generation module """
        super().__init__(
                exp_path=exp_path,
                sa_model_directory=sa_model_directory,
                num_seqs=num_seqs
            )

        model_name = sa_model_directory.split('.')[0]
        self.plots_path = os.path.join(
                self.exp_path,
                "plots",
                f"figGeneration_SaModel_{model_name}"
            )
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.plots_path)
        return

    @torch.no_grad()
    def compute_visualization(self, batch_data, img_idx):
        """ Computing visualizations """
        imgs, labels = batch_data
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        dataset_name = self.exp_params["dataset"]["dataset_name"]
        if (dataset_name in ["SpritesMOT", "VMDS"]):
            B, L, C, H, W = imgs.shape
            B_m, L_m, H_m, W_m = labels.shape
            imgs, labels = imgs[:, 0], labels[:, 0]
        num_slots = len(torch.unique(labels))
        reconstructions, (recons, masks, slot_embs) = self.model(imgs, num_slots)

        visualize_evaluation_slots(
                self.metric_tracker.tb_writer,
                masks.clamp(0, 1),
                img_idx,
                tag="generated_masks"
            )
        visualize_evaluation_slots(
                self.metric_tracker.tb_writer,
                (recons * masks).clamp(0, 1),
                img_idx,
                tag="decomposed_objects"
            )
        visualize_recons(
                imgs=imgs.clamp(0, 1),
                recons=reconstructions.clamp(0, 1),
                tb_writer=self.metric_tracker.tb_writer,
                iter=img_idx,
                n_cols=1,
                tag="Target_Recons_Error"
            )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, sa_model_directory, num_seqs, exact_slots = get_test_align_arguments()
    print_("Figure generation for a Slot Attention model...")
    figGenerator = FigGenerator(
            exp_path=exp_path,
            sa_model_directory=sa_model_directory,
            num_seqs=num_seqs
        )
    print_("Loading dataset...")
    figGenerator.load_data()
    print_("Setting up model and loading pretrained parameters")
    figGenerator.load_model()
    print_("Generating and saving figures")
    figGenerator.test_align()


#
