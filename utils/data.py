from datasets import load_dataset, Dataset, concatenate_datasets
from config.train_sft_model import DatasetConfig


def load_datasets(cfg: DatasetConfig, seed: int) -> Dataset:
    """
    Load training and validation datasets based on the configuration.
    Args:
        cfg (DatasetConfig): Configuration containing dataset information.
        seed (int): Random seed for reproducibility.
    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        
    If no datasets are provided or an error occurs, returns None for the respective dataset.
    """
    train_datasets, val_datasets = [], []

    train_cfg = getattr(cfg, "train_datasets", None)
    max_train_examples = getattr(cfg, "max_train_examples", None)
    if train_cfg:
        try:
            for dataset in train_cfg:
                dataset = load_dataset(dataset.name_or_path, split=dataset.split)
                train_datasets.append(dataset)

            if max_train_examples is not None:
                ratios = [dataset.ratio for dataset in train_cfg]
                total_ratio = sum(ratios)
                num_samples = max_train_examples

                if num_samples == -1:
                    for i, dataset in enumerate(train_datasets):
                        train_datasets[i] = dataset.shuffle(seed=seed)
                else:
                    samples_per_dataset = [
                        int(num_samples * ratio / total_ratio) for ratio in ratios
                    ]
                    for i, dataset in enumerate(train_datasets):
                        train_datasets[i] = dataset.shuffle(seed=seed).select(
                            range(samples_per_dataset[i])
                        )
            train_datasets = concatenate_datasets(train_datasets)
            train_datasets = train_datasets.shuffle(seed=seed)
        except Exception as e:
            train_datasets = None
            print("An error occurred while loading training datasets.")
            print(e)
    else:
        train_datasets = None

    eval_cfg = getattr(cfg, "eval_datasets", None)
    max_val_examples = getattr(cfg, "max_val_examples", None)
    if eval_cfg:
        try:
            for dataset in eval_cfg:
                dataset = load_dataset(dataset.name_or_path, split=dataset.split)
                val_datasets.append(dataset)

            if len(val_datasets) == 0:
                return train_datasets, None

            if max_val_examples is not None:
                ratios = [dataset.ratio for dataset in eval_cfg]
                total_ratio = sum(ratios)
                num_samples = max_val_examples

                if num_samples == -1:
                    for i, dataset in enumerate(val_datasets):
                        val_datasets[i] = dataset.shuffle(seed=seed)
                else:
                    samples_per_dataset = [
                        int(num_samples * ratio / total_ratio) for ratio in ratios
                    ]
                    for i, dataset in enumerate(val_datasets):
                        val_datasets[i] = dataset.shuffle(seed=seed).select(
                            range(samples_per_dataset[i])
                        )

            val_datasets = concatenate_datasets(val_datasets)
            val_datasets = val_datasets.shuffle(seed=seed)
        except Exception as e:
            val_datasets = None
            print("An error occurred while loading validation datasets.")
            print(e)
    else:
        val_datasets = None

    return train_datasets, val_datasets
