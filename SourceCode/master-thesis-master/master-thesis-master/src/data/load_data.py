"""
Methods for loading specific datasets, fitting data loaders and other
"""

# from torchvision import datasets
from torch.utils.data import DataLoader
from data import Tetrominoes, CustomMovingMNIST, SpritesDataset, MOVI, MultiDSprites,\
                 OBJ3D, PhysicalConcepts, SynpickVP, SynpickInstances, Sketchy
from CONFIG import CONFIG, DATASETS


def load_data(exp_params, split="train", augmentator=None):
    """
    Loading a dataset given the parameters

    Args:
    -----
    dataset_name: string
        name of the dataset to load
    split: string
        Split from the dataset to obtain (e.g., 'train' or 'test')
    transform: Torch Transforms
        Compose of torchvision transforms to apply to the data

    Returns:
    --------
    dataset: torch dataset
        Dataset loaded given specifications from exp_params
    in_channels: integer
        number of channels in the dataset samples (e.g. 1 for B&W, 3 for RGB)
    """
    # reading dataset parameters from the configuration files
    DATA_PATH = CONFIG["paths"]["data_path"]
    dataset_name = exp_params["dataset"]["dataset_name"]
    if dataset_name == "SpritesMOT":
        dataset = SpritesDataset(
                path=DATA_PATH,
                mode=split,
                rgb=True,
                dataset="spmot"
            )
    elif dataset_name == "VMDS":
        dataset = SpritesDataset(
                path=DATA_PATH,
                mode=split,
                rgb=True,
                dataset="vmds"
            )
    elif dataset_name == "OBJ3D":
        dataset = OBJ3D(
                mode=split,
                sample_length=exp_params["training_prediction"]["sample_length"]
            )
    elif dataset_name == "MultiDSprites":
        dataset = MultiDSprites(
                path=DATA_PATH,
                mode=split
            )
    elif dataset_name == "SynpickVP":
        dataset = SynpickVP(
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                seq_step=2,         # default
                max_overlap=0.25,   # default
                img_size=(64, 112),
                use_segmentation=exp_params["dataset"].get("use_segmentation", True),
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    elif dataset_name == "SynpickInstances":
        dataset = SynpickInstances(
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                seq_step=2,         # default
                max_overlap=0.25,   # default
                img_size=(64, 112),
                use_segmentation=True,
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    elif dataset_name == "Sketchy":
        dataset = Sketchy(
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                seq_step=2,         # default
                max_overlap=0.,     # default
                img_size=(80, 120),
                mode="front_left"
            )
    elif dataset_name == "PhysicalConcepts":
        dataset = PhysicalConcepts(
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                img_size=(64, 64),
                get_masks=False,
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    elif dataset_name == "MoviA":
        dataset = MOVI(
                datapath="/home/nfs/inf6/data/datasets/MOVi/movi_a",
                target=exp_params["dataset"].get("target", "rgb"),
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                img_size=(64, 64),
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    elif dataset_name == "MoviB":
        raise NotImplementedError("Movi-B dataset is not yet supported")
        dataset = MOVI(
                datapath="/home/nfs/inf6/data/datasets/MOVi/movi_b",
                target=exp_params["dataset"].get("target", "rgb"),
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                img_size=(64, 64),
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    elif dataset_name == "MoviC":
        dataset = MOVI(
                datapath="/home/nfs/inf6/data/datasets/MOVi/movi_c",
                target=exp_params["dataset"].get("target", "rgb"),
                split=split,
                num_frames=exp_params["training_prediction"]["sample_length"],
                img_size=(64, 64),
                slot_initializer=exp_params["model"]["SAVi"].get("initializer", "LearnedRandom")
            )
    else:
        raise NotImplementedError(f"""ERROR! Dataset'{dataset_name}' is not available.
            Please use one of the following: {DATASETS}...""")

    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """

    data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"]
        )

    return data_loader


def unwrap_batch_data(exp_params, batch_data):
    """
    Unwrapping the batch data depending on the dataset that we are training on
    """
    initializer_kwargs = {}
    if exp_params["dataset"]["dataset_name"] in ["VMDS", "Sketchy", "OBJ3D"]:
        videos, targets, _ = batch_data
    elif exp_params["dataset"]["dataset_name"] == "PhysicalConcepts":
        videos, targets, all_reps = batch_data
        initializer_kwargs["com_coords"] = all_reps["com_coords"]
        initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
    elif exp_params["dataset"]["dataset_name"] in ["SynpickVP", "MoviA", "MoviB", "MoviC"]:
        videos, targets, all_reps = batch_data
        initializer_kwargs["instance_masks"] = all_reps["masks"]
        initializer_kwargs["com_coords"] = all_reps["com_coords"]
        initializer_kwargs["bbox_coords"] = all_reps["bbox_coords"]
    else:
        dataset_name = exp_params["dataset"]["dataset_name"]
        raise NotImplementedError(f"Dataset {dataset_name} is not supported...")
    return videos, targets, initializer_kwargs


#
