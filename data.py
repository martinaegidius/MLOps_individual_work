import torch
from torch.utils.data import TensorDataset


DATA_PATH = "s1_development_environment/exercise_files/final_exercise/data/corruptmnist_v1/"

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    train_im_list = [f"{DATA_PATH}train_images_{i}.pt" for i in range(6)]
    train_label_list = [f"{DATA_PATH}train_target_{i}.pt" for i in range(6)]
    test_im_list = [f"{DATA_PATH}test_images.pt"]
    test_label_list = [f"{DATA_PATH}test_target.pt"]

    train_images, train_targets = [],[]
    
    for (f_im,f_lab) in zip(train_im_list,train_label_list):
        ims = torch.load(f_im)
        train_images.append(ims)
        labs = torch.load(f_lab)
        train_targets.append(labs)
    
    train_images = torch.cat(train_images)
    train_targets = torch.cat(train_targets)


    test_images, test_targets = [],[]
    for (f_im,f_lab) in zip(test_im_list,test_label_list):
        ims = torch.load(f_im)
        test_images.append(ims)
        labs = torch.load(f_lab)
        test_targets.append(labs)
    
    test_images = torch.cat(test_images)
    test_targets = torch.cat(test_targets)

    train_images = train_images.unsqueeze(1).float() #add channel dim
    test_images = test_images.unsqueeze(1).float() #add channel dim
    train_targets = train_targets.long()
    test_targets = test_targets.long()

    train = TensorDataset(train_images,train_targets)
    test = TensorDataset(test_images,test_targets)
    
    
    return train, test
    #return train, test

if __name__=="__main__":
    import matplotlib.pyplot as plt  # only needed for plotting
    from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

    def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
        """Plot images and their labels in a grid."""
        row_col = int(len(images) ** 0.5)
        fig = plt.figure(figsize=(10.0, 10.0))
        grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
        for ax, im, label in zip(grid, images, target):
            ax.imshow(im.squeeze(), cmap="gray")
            ax.set_title(f"Label: {label.item()}")
            ax.axis("off")
        plt.show()

    train, test = corrupt_mnist()
    print("returned train: ")
    print(train)
    print("\n with length ",len(train))
    print("returned test: ")
    print(test)
    print("\n with length ",len(test))
    show_image_and_target(train.tensors[0][:25], train.tensors[1][:25])
