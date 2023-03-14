"""
Testing and visualizing the alignment results given a bunch of models
"""


import os
from tqdm import tqdm
import torch

from lib.arguments import get_test_align_arguments
from lib.config import Config
from lib.hungarian import align_sequence
from lib.logger import print_
import lib.setup_model as setup_model
import lib.utils as utils
from lib.visualizations import visualize_recons, visualize_aligned_slots, display_alignment_scores
from models.model_utils import freeze_params
import data
import data.data_utils as data_utils


class Aligner:
    """
    Class for evaluating a model
    """

    def __init__(self, exp_path, sa_model_directory, num_seqs=10, exact_slots=False):
        """
        Initializing the trainer object
        """
        self.parent_exp_path = exp_path
        self.exp_path = os.path.join(exp_path, "predictor")
        self.cfg = Config(self.parent_exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.sa_model_directory = sa_model_directory
        self.num_seqs = num_seqs
        self.exact_slots = exact_slots

        self.plots_path = os.path.join(
                self.parent_exp_path,
                "plots",
                f"test_align_ExactSlots_{exact_slots}_SaModel_{sa_model_directory}")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        return

    def load_data(self):
        """
        Loading dataset and fitting data-loader for iterating in a batch-like fashion
        """
        batch_size = 1
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]
        test_set = data.load_data(
                dataset_name=self.exp_params["dataset"]["dataset_name"],
                split="test"
            )
        self.test_loader = data.build_data_loader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return

    def load_model(self):
        """
        load slot attention model from checkpoint
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        utils.log_architecture(self.model, exp_path=self.parent_exp_path)
        self.model = self.model.eval().to(self.device)

        checkpoint_path = os.path.join(self.parent_exp_path, "models", self.sa_model_directory)
        self.model = setup_model.load_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                only_model=True
            )
        freeze_params(self.model)
        return

    @torch.no_grad()
    def test_align(self):
        """
        Evaluating model epoch loop
        """
        progress_bar = tqdm(enumerate(self.test_loader), total=self.num_seqs)
        slot_dim = self.model.slot_dim
        num_context = self.exp_params["training_prediction"]["num_context"]

        # iterating test set and accumulating the results
        for iter_, (videos, masks) in progress_bar:
            if iter_ >= self.num_seqs:
                break
            videos = videos.to(self.device)
            B, L, C, H, W = videos.shape
            imgs = videos.view(B * L, C, H, W)

            # setting exact number of slots, or num_slots from experiment parameters
            stats = data_utils.get_slots_stats(seq=videos[0], masks=masks[0])
            if self.exact_slots:
                num_slots = stats["max_num_slots"]
            else:
                num_slots = self.model.num_slots

            # encoding images into object-centric slots, and temporally aligning sequece
            _, (orig_recons, orig_masks, slot_embs) = self.model(imgs[:num_context], num_slots)
            slots = slot_embs.view(B, num_context, num_slots, slot_dim)
            orig_recons = orig_recons.view(B, num_context, num_slots, C, H, W)
            aligned_slots, sim_scores = align_sequence(slots)  # (B, num_context, num_slots, slot_dim)

            # decoding aligned slots
            aligned_slots = aligned_slots.reshape(B * num_context, num_slots, slot_dim)
            reconstructions, (recons, masks) = self.model.decode(aligned_slots)
            recons = recons.view(B, num_context, num_slots, C, H, W)

            # some visualizations
            savepath = os.path.join(self.plots_path, f"Reconstructions_{iter_ + 1}.png")
            visualize_recons(
                    imgs=imgs,
                    recons=reconstructions,
                    savepath=savepath,
                    n_cols=num_context
                )
            savepath = os.path.join(self.plots_path, f"Aligned_Slots_{iter_ + 1}.png")
            _ = visualize_aligned_slots(
                    recons_objs=(masks * recons)[0].clamp(0, 1),
                    savepath=savepath,
                )
            savepath = os.path.join(self.plots_path, f"Unaligned_Slots_{iter_ + 1}.png")
            _ = visualize_aligned_slots(
                    recons_objs=(orig_recons * orig_masks)[0].clamp(0, 1),
                    savepath=savepath,
                )
            savepath = os.path.join(self.plots_path, f"Alignment_Scores_{iter_ + 1}.png")
            _ = display_alignment_scores(
                    scores=sim_scores[0],
                    savepath=savepath
                )
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, sa_model_directory, num_seqs, exact_slots = get_test_align_arguments()
    print_("Testing alignment procedure...")
    aligner = Aligner(
            exp_path=exp_path,
            sa_model_directory=sa_model_directory,
            num_seqs=num_seqs,
            exact_slots=exact_slots
        )
    print_("Loading dataset...")
    aligner.load_data()
    print_("Setting up model and loading pretrained parameters")
    aligner.load_model()
    print_("Testing Alignments")
    aligner.test_align()


#
