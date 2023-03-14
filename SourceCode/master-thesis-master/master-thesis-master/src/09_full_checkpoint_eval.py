"""
Given an experiment directory, testing all checkpoints and monitoring the
metrics on the test set.
"""

import os
import json
from matplotlib import pyplot as plt
from lib.arguments import get_sa_eval_arguments
import lib.utils as utils

Evaluate = __import__("03_evaluate")


def get_ordered_models(exp_path):
    """ Getting a list with all saved model checkpoints """
    models_path = os.path.join(exp_path, "models")
    model_names = sorted(os.listdir(models_path))

    # getting all models and the corresponding epoch
    models = {}
    for model_name in model_names:
        words = model_name[:-4].split("_")
        epoch = int(words[-1]) if words[-1].isnumeric() else None
        if epoch is not None:
            models[epoch] = model_name

    # sorting according to epoch
    models = dict(sorted(models.items()))
    return models


def evaluate_models(models, exact_slots):
    """ Iterating through all the models and evaluating them """
    # evaluating all checkpoints
    results = {}
    for i, (epoch, model) in enumerate(models.items()):
        print(f"Evaluatiing model {i+1}/{len(models)}")
        print(f"  --> Checkpoint name {model}")
        print(f"  --> Epoch {epoch}")
        evaluator = Evaluate.Evaluator(
                exp_path=exp_path,
                checkpoint=model,
                exact_slots=exact_slots
            )
        evaluator.load_data()
        evaluator.setup_model()
        evaluator.evaluate(save_results=False)
        results[epoch] = evaluator.results
    return results


def main(exp_path, exact_slots=True):
    """ Main orquestrator for evaluating all saved checkpoints """
    # important paths and files
    plots_path = os.path.join(exp_path, "plots", "metrics")
    utils.create_directory(plots_path)
    results_path = os.path.join(exp_path, "results")
    utils.create_directory(results_path)
    results_file = os.path.join(results_path, f"all_results_exactSlots_{exact_slots}.json")

    # obtaining stored checkpoints
    models = get_ordered_models(exp_path)

    key_pressed = "y"
    if os.path.exists(results_file):
        message = [
                f"Results file {results_file} already exists.",
                "Do you want to recompute the results?"
            ]
        key_pressed = utils.press_yes_or_no(message)

    # computing and saving results
    if key_pressed == "y":
        results = evaluate_models(models, exact_slots)
        with open(results_file, "w") as f:
            json.dump(results, f)

    # loading results to generate plots
    with open(results_file, "r") as f:
        results = json.load(f)

    # generating plots
    epochs = [int(e) for e in list(results.keys())]
    metrics = list(results[str(epochs[0])].keys())
    plt.style.use('seaborn')
    for metric in metrics:
        savepath = os.path.join(plots_path, f"{metric}_over_epochs_exactSlots_{exact_slots}.png")
        values = [results[str(e)][metric] for e in epochs]
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values)
        plt.scatter(epochs, values)
        plt.title(f"{metric} for every checkpoint")
        plt.ylabel("Value")
        plt.xlabel("Epoch")
        T = len(epochs) // 10
        epochs_disp = epochs[::T]
        plt.xticks(epochs_disp)
        plt.savefig(savepath)

    return


if __name__ == "__main__":
    utils.clear_cmd()
    exp_path, args = get_sa_eval_arguments()
    main(exp_path=exp_path, exact_slots=args.exact_slots)
