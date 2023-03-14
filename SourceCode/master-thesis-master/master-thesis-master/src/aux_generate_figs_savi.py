"""
Generating figures using a pretrained SAVI model
"""

import os
import torch

from base.baseFigGenerator import BaseFigGenerator

from lib.arguments import get_test_align_arguments
from lib.logger import print_
import lib.utils as utils
from lib.visualizations import visualize_recons, visualize_decomp


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
                f"figGeneration_SaVIModel_{model_name}"
            )
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.plots_path)
        return

    def unwrap_batch_data(self, batch_data):
        """
        Unwrapping the batch data depending on the dataset that we are training on
        """
        initializer_kwargs = {}
        if self.exp_params["dataset"]["dataset_name"] in ["VMDS", "Sketchy"]:
            videos, _ = batch_data
        elif self.exp_params["dataset"]["dataset_name"] in ["SynpickVP", "SynpickInstances"]:
            videos, all_reps = batch_data
            initializer_kwargs["instance_masks"] = all_reps["masks"]
            initializer_kwargs["com_coords"] = all_reps["com_coords"]
            initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
        else:
            videos = batch_data
        return videos, initializer_kwargs

    @torch.no_grad()
    def compute_visualization(self, batch_data, img_idx):
        """ Computing visualization """
        videos, initializer_kwargs = self.unwrap_batch_data(batch_data)
        videos = videos.to(self.device)
        out_model = self.model(videos, num_imgs=videos.shape[1], **initializer_kwargs)
        slot_history, reconstruction_history, individual_recons_history, masks_history = out_model

        N = min(10, videos.shape[1])
        savepath = os.path.join(self.plots_path, f"Recons_{img_idx+1}.png")
        visualize_recons(
                imgs=videos[0, :N].clamp(0, 1),
                recons=reconstruction_history[0, :N].clamp(0, 1),
                n_cols=10,
                savepath=savepath
            )

        savepath = os.path.join(self.plots_path, f"Objects_{img_idx+1}.png")
        _ = visualize_decomp(
                individual_recons_history[0, :N],
                savepath=savepath,
                vmin=0,
                vmax=1,
            )

        savepath = os.path.join(self.plots_path, f"masks_{img_idx+1}.png")
        _ = visualize_decomp(
                masks_history[0][:N],
                savepath=savepath,
                cmap="gray_r",
                vmin=0,
                vmax=1,
            )
        savepath = os.path.join(self.plots_path, f"maskedObj_{img_idx+1}.png")
        recon_combined = masks_history[0][:N] * individual_recons_history[0][:N]
        recon_combined = torch.clamp(recon_combined, min=0, max=1)
        _ = visualize_decomp(
                recon_combined,
                savepath=savepath,
                vmin=0,
                vmax=1,
            )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, sa_model_directory, num_seqs, exact_slots = get_test_align_arguments()
    print_("Generating figures for SAVI...")
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
    figGenerator.generate_figs()


#
