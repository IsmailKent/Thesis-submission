"""
Postprocessing the prediction metrics so that we can extract the prediction results
for num_preds=M from the precomputed results num_preds=N, when N > M
"""

import os
import json
import numpy as np

from lib.arguments import get_postprocess_results_arguments
import lib.utils as utils


NUM_FRAMES = [5, 15, 30]


def main():
    """
    Postprocessing the prediction metrics so that we can extract the prediction results
    for num_preds=M from the precomputed results num_preds=N, when N > M
    """
    exp_path, checkpoint, name_pred_experiment, num_preds = get_postprocess_results_arguments()
    print("Started process for postprocessing results...")

    # processing paths and finding results file
    checkpoint = checkpoint.split(".")[0] if ".pth" in checkpoint else checkpoint
    results_path = os.path.join(
            exp_path,
            name_pred_experiment,
            "results",
            f"{checkpoint}_NumPreds={num_preds}"
        )
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results path {results_path} does not exist...")
    results_file = os.path.join(results_path, "results.json")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file {results_file} does not exist...")

    # loading results data
    with open(results_file) as f:
        results_data = json.load(f)

    # computing the results for each of the desired horizons, and storing results
    for num_frames in NUM_FRAMES:
        cur_results = {}
        for metric in results_data.keys():
            values = results_data[metric]["framewise"][:num_frames]
            cur_results[metric] = {}
            cur_results[metric]["mean"] = np.mean(values)
            cur_results[metric]["framewise"] = values
        results_file = os.path.join(results_path, f"results_NumFrames={num_frames}.json")
        with open(results_file, "w") as f:
            json.dump(cur_results, f)

    print("Process finished successfully...")
    return


if __name__ == '__main__':
    utils.clear_cmd()
    main()


#
