"""
Benchmarking a model in terms of Throughput, FLOPS, Number of Parameters, ...
"""

import os
import json
import torch

from lib.arguments import get_directory_argument
from lib.config import Config
from lib.logger import Logger, print_, log_function, for_all_methods
import lib.setup_model as setup_model
import lib.utils as utils
import models.model_utils as model_utils
import data


@for_all_methods(log_function)
class Benchmarker:
    """ Class for benchmarking a model """

    def __init__(self, exp_path, checkpoint):
        """ Initializing the benchmarker object """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.checkpoint = checkpoint

        self.models_path = os.path.join(self.exp_path, "models")
        self.results_path = os.path.join(self.exp_path, "benchmark")
        utils.create_directory(self.results_path)

        model_name = self.exp_params["model"]["model_name"]
        self.slot_dim = self.exp_params["model"][model_name]["slot_dim"]
        self.num_slots = self.exp_params["model"][model_name]["num_slots"]
        self.batch_size = self.exp_params["training_prediction"]["batch_size"]
        self.num_context = self.exp_params["training_prediction"]["num_context"]
        is_lstm = self.exp_params["model"]["predictor"]["predictor_name"] == "LSTM"
        self.batch_shape = ( self.num_context, self.slot_dim) if is_lstm else (self.dataset_size, self.num_context, self.num_slots,  self.slot_dim)
        return


    def setup_predictor(self):
        """
        Initializing model, optimizer, loss function and other related objects
        """
        torch.backends.cudnn.fastest = True
        # loading predictor
        predictor = setup_model.setup_predictor(exp_params=self.exp_params)
        utils.log_architecture(predictor, exp_path=self.exp_path)
        predictor = predictor.eval().to(self.device)

        print_(f"Loading pretrained parameters from checkpoint {self.checkpoint}...")
        predictor = setup_model.load_checkpoint(
                checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                model=predictor,
                only_model=True,
            )
        self.predictor = predictor
        return

    def load_data(self):
        """ generate mockdataset"""
        self.batch = torch.rand(*self.batch_shape)
        self.dataset =  torch.utils.data.TensorDataset(
                    self.batch
                )
        return

    @torch.no_grad()
    def benchmark(self):
        """ Benchmarking the model """

        print_("Benchmarking number of learnambe parameters...")
        num_params = model_utils.count_model_params(self.predictor, verbose=True)

        print_("Benchmarking FLOPS and #Activations...")
        total_flops, total_act = model_utils.compute_flops(
                model=self.predictor.cpu(),
                dummy_input= torch.rand(1,64), #self.batch[0,0,:],
                detailed=False,
                verbose=True
            )
        print_("Benchmarking Throughput...")
        throughput, avg_time_per_img = model_utils.compute_throughput(
                model=self.predictor.cpu(),
                dataset=self.dataset,
                device="cpu",
                num_imgs=self.batch_size,
                use_tqdm=True,
                verbose=True
            )

        print_("Saving benchmark results...")
        results = {
            "num_params": num_params,
            "total_flops": total_flops,
            "total_act": total_act,
            "throughput": throughput,
            "avg_time_per_img": avg_time_per_img
        }
        checkpoint_name = self.checkpoint.split(".")[0]
        savepath = os.path.join(self.results_path, f"benchmark_{checkpoint_name}.json")
        with open(savepath, "w") as f:
            json.dump(results, f)

        return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_directory_argument()
    logger = Logger(exp_path=exp_path)
    logger.log_info("Starting benchmarking procedure", message_type="new_exp")

    print_("Initializing Benchmarker...")
    benchmarker = Benchmarker(exp_path=exp_path, checkpoint=args.checkpoint)
    print_("Loading dataset...")
    benchmarker.load_data()
    print_("Setting up model and loading pretrained parameters")
    benchmarker.setup_predictor()
    benchmarker.benchmark()


#
