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
from hydra.utils import to_absolute_path
from dotenv import load_dotenv
    
from loguru import logger
import wandb




#log = logging.getLogger(__name__) #before loguru

#wandb_api_key = os.getenv("WANDB_API_KEY") <- not necessary when already logged in
load_dotenv()
wandb_entity = os.getenv("WANDB_ENTITY")
wandb_project = os.getenv("WANDB_PROJECT")
wandb.login(key=os.getenv("WANDB_API_KEY"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(version_base=None, config_path="../../configs",config_name="config")
def train(cfg: DictConfig):
#def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 2) -> None:
    """Train a model on MNIST."""
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Add a log file to the logger
    hyperparameters = cfg._group_.hyperparameters
    lr = hyperparameters.lr
    batch_size = hyperparameters.batch_size
    epochs = hyperparameters.epochs
    seed = hyperparameters.seed

    logger.add(os.path.join(hydra_path, "my_logger_hydra.log"))
    logger.info(cfg)
    logger.info("Training day and night")
    print(hyperparameters)
    print(type(hyperparameters))
    wandb.init(entity=wandb_entity,project=wandb_project,config=dict(hyperparameters))
    

    torch.manual_seed(seed)

    logger.info(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    N_SAMPLES = len(train_set)
    print(N_SAMPLES)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    del train_set

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            running_loss += loss
            statistics["train_loss"].append(loss.item())
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            #let's plot the first 16 images of the first batch with corresponding predictions
            if(i==0):
                images = images.permute(0,2,3,1).detach().numpy()[:16] #need permute for plt plotting
                labels = labels.detach().numpy()[:16]
                predicted_classes = torch.argmax(logits,dim=1).detach().numpy()[:16]
                fig, axes = plt.subplots(4,4)
                for j, ax in enumerate(axes.flat):
                    ax.imshow(images[j])
                    ax.set_axis_off()
                    ax.text(3,5, f"{predicted_classes[j]}",color="red",fontweight="bold")
                plt.tight_layout(pad=0.0,w_pad=0.0,h_pad=0.0)
                fig.suptitle(f"epoch {epoch}",fontweight="bold")
                wandb.log({"training predictions": wandb.Image(fig)})
                plt.close()
                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads)})

        epoch_loss = running_loss.item() / N_SAMPLES
        logger.info(f"Epoch {epoch}: {epoch_loss}")
        statistics["train_loss"].append(epoch_loss)
        wandb.log({"train_loss":epoch_loss})

    logger.info("Finished training")
    torch.save(model.state_dict(), f"{os.getcwd()}/model.pth") #save to hydra output (hopefully using chdir true)
    artifact = wandb.Artifact(name="model",type="model")
    artifact.add_file(local_path=f"{os.getcwd()}/model.pth",name="model.pth")
    artifact.save()
    logger.info(f"Saved model to: f{os.getcwd()}/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    #fig.savefig("reports/figures/training_statistics.png") <- with no hydra configuration
    fig.savefig(f"{os.getcwd()}/training_statistics.png")   
    wandb.log({"training statistics":wandb.Image(fig)}) #try to log an image 
    
if __name__ == "__main__":
    train()
    #typer.run(train)