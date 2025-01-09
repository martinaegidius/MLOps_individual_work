#import logging before loguru
import os 

import torch
import typer
from mlops_individual.data import corrupt_mnist
from mlops_individual.model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra
from loguru import logger



#log = logging.getLogger(__name__) #before loguru



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../../configs",config_name="config")
def train(cfg: DictConfig):
#def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 2) -> None:
    """Train a model on MNIST."""
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Add a log file to the logger
    logger.add(os.path.join(hydra_path, "my_logger_hydra.log"))
    logger.info(cfg)
    logger.info("Training day and night")
    hyperparameters = cfg._group_.hyperparameters
    lr = hyperparameters.lr
    batch_size = hyperparameters.batch_size
    epochs = hyperparameters.epochs
    seed = hyperparameters.seed

    torch.manual_seed(seed)

    logger.info(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    N_SAMPLES = len(train_set)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    del train_set

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss

            statistics["train_loss"].append(loss.item())
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

        epoch_loss = running_loss.item() / N_SAMPLES
        logger.info(f"Epoch {epoch}: {epoch_loss}")
        statistics["train_loss"].append(epoch_loss)

    logger.info("Finished training")
    torch.save(model.state_dict(), f"{os.getcwd()}/model.pth") #save to hydra output (hopefully using chdir true)
    logger.info(f"Saved model to: f{os.getcwd()}/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    #fig.savefig("reports/figures/training_statistics.png") <- with no hydra configuration
    fig.savefig(f"{os.getcwd()}/training_statistics.png")    
    
if __name__ == "__main__":
    train()
    #typer.run(train)