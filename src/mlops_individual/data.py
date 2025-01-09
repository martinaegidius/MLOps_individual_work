from pathlib import Path
import os
import typer
from torch.utils.data import Dataset, TensorDataset
import torch
from hydra.utils import to_absolute_path #for resolving paths as originally for loading data
import matplotlib.pyplot as plt 

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - tensor.mean()) / tensor.std()

    def preprocess(self, raw_dir: Path, proc_dir: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        train_im_list = [f"{raw_dir}/train_images_{i}.pt" for i in range(6)]
        train_label_list = [f"{raw_dir}/train_target_{i}.pt" for i in range(6)]
        test_im_list = [f"{raw_dir}/test_images.pt"]
        test_label_list = [f"{raw_dir}/test_target.pt"]

        train_images, train_targets = [], []

        for f_im, f_lab in zip(train_im_list, train_label_list):
            ims = torch.load(f_im)
            train_images.append(ims)
            labs = torch.load(f_lab)
            train_targets.append(labs)

        train_images = torch.cat(train_images)
        train_targets = torch.cat(train_targets)

        test_images, test_targets = [], []
        for f_im, f_lab in zip(test_im_list, test_label_list):
            ims = torch.load(f_im)
            test_images.append(ims)
            labs = torch.load(f_lab)
            test_targets.append(labs)

        test_images = torch.cat(test_images)
        test_targets = torch.cat(test_targets)

        train_images = train_images.unsqueeze(1).float()  # add channel dim
        test_images = test_images.unsqueeze(1).float()  # add channel dim
        train_targets = train_targets.long()
        test_targets = test_targets.long()

        train_images = self.normalize(train_images)
        test_images = self.normalize(test_images)
        # print(proc_dir / "train_images.pt")
        torch.save(train_images, proc_dir / "train_images.pt")
        torch.save(train_targets, proc_dir / "train_labels.pt")
        torch.save(test_images, proc_dir / "test_images.pt")
        torch.save(test_targets, proc_dir / "test_labels.pt")
        return


def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    proc_path = "data/processed/"
    
    proc_path = to_absolute_path(proc_path)+"/"
    
    train_images = torch.load(proc_path + "train_images.pt")
    train_labels = torch.load(proc_path + "train_labels.pt")
    test_images = torch.load(proc_path + "test_images.pt")
    test_labels = torch.load(proc_path + "test_labels.pt")

    train = TensorDataset(train_images, train_labels)
    test = TensorDataset(test_images, test_labels)

    return train, test


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)  # / "corruptmnist_v1")
    dataset.preprocess(raw_data_path, output_folder)  # / "corruptmnist_v1")
    return


if __name__ == "__main__":
    typer.run(preprocess)
    