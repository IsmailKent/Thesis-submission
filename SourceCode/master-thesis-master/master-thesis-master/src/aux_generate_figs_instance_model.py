"""
Generating figures using a pretrained Recursive Instance model
"""

import os
import torch

from base.baseFigGenerator import BaseFigGenerator

from lib.arguments import get_test_align_arguments
from lib.logger import print_
import lib.utils as utils
from lib.visualizations import visualize_decomp, visualize_recons, one_hot_to_instances, instances_to_rgb


class FigGenerator(BaseFigGenerator):
    """
    Class for generating figures using a pretrained Recursive Instance model
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
        if self.exp_params["dataset"]["dataset_name"] == "SynpickInstances":
            videos, all_reps = batch_data
            initializer_kwargs["instance_masks"] = all_reps["masks"]
            initializer_kwargs["com_coords"] = all_reps["com_coords"]
            initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
        else:
            raise ValueError("Only dataset supported in this script is 'SynpickInstances'")
        return videos, initializer_kwargs

    @torch.no_grad()
    def compute_visualization(self, batch_data, img_idx):
        """ Computing visualization"""
        instances, initializer_kwargs = self.unwrap_batch_data(batch_data)
        instances = instances.to(self.device).float()
        out_model = self.model(instances, num_imgs=instances.shape[1], **initializer_kwargs)
        slot_history, recons_instances = out_model

        # some postprocessing for visualization
        N = min(10, instances.shape[1])
        num_objects = instances.shape[2]
        instances_merged = one_hot_to_instances(instances).squeeze(2)
        instances_rgb = instances_to_rgb(instances_merged, num_channels=num_objects).clamp(0, 1)
        recons_instances_merged = one_hot_to_instances(recons_instances).squeeze(2)
        recons_rgb = instances_to_rgb(recons_instances_merged, num_channels=num_objects).clamp(0, 1)

        savepath = os.path.join(self.plots_path, f"Recons_{img_idx+1}.png")
        visualize_recons(
                imgs=instances_rgb[0, :N],
                recons=recons_rgb[0, :N],
                n_cols=10,
                savepath=savepath
            )

        savepath = os.path.join(self.plots_path, f"Recons_Masks_{img_idx+1}.png")
        _ = visualize_decomp(recons_instances[0][:N], savepath=savepath, cmap="gray", vmin=0, vmax=1)
        savepath = os.path.join(self.plots_path, f"GT_Masks_{img_idx+1}.png")
        _ = visualize_decomp(instances[0][:N], savepath=savepath, cmap="gray", vmin=0, vmax=1)
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, sa_model_directory, num_seqs, exact_slots = get_test_align_arguments()
    print_("Generating figures with a pretrained Recursive Instance model...")
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
