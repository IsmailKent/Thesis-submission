"""
Converting checkpoints without initializer, so that they can
be loaded with the new version of the code
"""

import os
import torch

from lib.arguments import get_sa_eval_arguments
from lib.config import Config
import lib.setup_model as setup_model
import lib.utils as utils


class ModelConverter:
    """
    Module that converts checkpoints without initializer, so that they can
    be loaded with the new version of the code
    """

    def __init__(self, exp_path, checkpoint):
        """
        Initializing the trainer object
        """
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint

        self.models_path = os.path.join(self.exp_path, "models")
        self.out_path = os.path.join(self.models_path, f"{checkpoint.split('.')[0]}_converted.pth")
        utils.create_directory(self.models_path)

    def setup_model(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """
        torch.backends.cudnn.fastest = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # loading model
        self.model = setup_model.setup_model(model_params=self.exp_params["model"])
        self.model = self.model.eval().to(self.device)

        checkpoint_path = os.path.join(self.models_path, self.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
        checkpoint = torch.load(checkpoint_path)

        # creating new state dictionary
        new_checkpoint = {key: val for key, val in checkpoint.items()}
        old_keys = ["slot_attention.slots_mu", "slot_attention.slots_sigma"]
        new_keys = ["initializer.slots_mu", "initializer.slots_sigma"]
        for old_key, new_key in zip(old_keys, new_keys):
            new_checkpoint["model_state_dict"][new_key] = new_checkpoint["model_state_dict"].pop(old_key)

        self.model.load_state_dict(new_checkpoint['model_state_dict'])
        print("Convereted parameters have been loaded successfully!")

        print("Saving converted paramters, loading them, and checking that everything loads fine")
        torch.save(new_checkpoint, self.out_path)
        _ = setup_model.load_checkpoint(
                checkpoint_path=self.out_path,
                model=self.model,
                only_model=True
            )
        print("Everything worked well!")
        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_sa_eval_arguments()
    print("Initializing Evaluator...")
    evaluator = ModelConverter(exp_path=exp_path, checkpoint=args.checkpoint)
    print("Setting up model and loading pretrained parameters")
    evaluator.setup_model()


#
