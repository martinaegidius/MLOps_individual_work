import torch
from torch.utils.data import TensorDataset
import typer


def normalize(x: torch.FloatTensor) -> torch.FloatTensor:
    return (x - x.mean()) / x.std()


def preprocess_data(raw_dir: str, proc_dir: str) -> None:
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

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    torch.save(train_images, proc_dir + "/train_images.pt")
    torch.save(train_targets, proc_dir + "/train_labels.pt")
    torch.save(test_images, proc_dir + "/test_images.pt")
    torch.save(test_targets, proc_dir + "/test_labels.pt")
    return


def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    proc_path = "data/processed/"
    train_images = torch.load(proc_path + "train_images.pt")
    train_labels = torch.load(proc_path + "train_labels.pt")
    test_images = torch.load(proc_path + "test_images.pt")
    test_labels = torch.load(proc_path + "test_labels.pt")

    train = TensorDataset(train_images, train_labels)
    test = TensorDataset(test_images, test_labels)

    return train, test


if __name__ == "__main__":
    # import matplotlib.pyplot as plt  # only needed for plotting
    # from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

    # def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    #     """Plot images and their labels in a grid."""
    #     row_col = int(len(images) ** 0.5)
    #     fig = plt.figure(figsize=(10.0, 10.0))
    #     grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    #     for ax, im, label in zip(grid, images, target):
    #         ax.imshow(im.squeeze(), cmap="gray")
    #         ax.set_title(f"Label: {label.item()}")
    #         ax.axis("off")
    #     plt.show()
    # print("... preprocessing data ...")
    # typer.run(preprocess_data)
    # print("\n... finished preprocessing ...\n")
    # print("testing tensordatasets and visualizing")
    # train, test = corrupt_mnist()
    # print("returned train: ")
    # print(train)
    # print("\n with length ",len(train))
    # print("returned test: ")
    # print(test)
    # print("\n with length ",len(test))
    # show_image_and_target(train.tensors[0][:25], train.tensors[1][:25])
    typer.run(preprocess_data)


from pathlib import Path
import os
import torch

import typer
from torch.utils.data import Dataset


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

    def get_data(self) -> dict:
        data_dir = {"train_images": [], "train_target": [], "test_images": [], "test_target": []}

        for file_name in os.listdir(self.data_path):
            filetype = file_name.split(".")[0]
            if filetype[-1].isdigit():
                filetype = filetype[:-2]
            data_dir[filetype].append(torch.load(os.path.join(self.data_path, file_name)))

        return data_dir

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

        data_dir = self.get_data()

        # Concatenate all data:
        train_images = torch.cat(data_dir["train_images"])
        train_targets = torch.cat(data_dir["train_target"])
        test_images = torch.cat(data_dir["test_images"])
        test_targets = torch.cat(data_dir["test_target"])

        # Manipulate data:
        train_images = train_images.unsqueeze(1).float()
        test_images = test_images.unsqueeze(1).float()
        train_targets = train_targets.long()
        test_targets = test_targets.long()

        # Normalize:
        train_images = self.normalize(train_images)
        test_images = self.normalize(test_images)

        # Save intermediate representation:
        torch.save(train_images, output_folder / "train_images.pt")
        torch.save(test_images, output_folder / "test_images.pt")
        torch.save(train_targets, output_folder / "train_targets.pt")
        torch.save(test_targets, output_folder / "test_targets.pt")


def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path / "corruptmnist_v1")
    dataset.preprocess(output_folder / "corruptmnist_v1")


if __name__ == "__main__":
    typer.run(preprocess)
