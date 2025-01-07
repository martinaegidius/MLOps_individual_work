import torch
import typer
from data_solution import corrupt_mnist
from model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn 
import matplotlib.pyplot as plt 
from tqdm import tqdm

app = typer.Typer()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 2) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # TODO: Implement training loop here
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()
    N_SAMPLES = len(train_set)
    trainloader = DataLoader(train_set,batch_size=batch_size,shuffle=True)
    del train_set 

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = criterion(logits,labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss
            
            statistics["train_loss"].append(loss.item())
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

        epoch_loss = running_loss.item()/N_SAMPLES
        print(f"Epoch {epoch}: {epoch_loss}")
        statistics["train_loss"].append(epoch_loss)

    print("Finished training")
    torch.save(model.state_dict(),"model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")



@app.command()
def evaluate(model_checkpoint: str = "model.pth") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = corrupt_mnist()
    testloader = DataLoader(test_set,batch_size=32,shuffle=False)
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    correct, total = 0,0 
    for img, label in testloader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        pred = model(img)
        correct += (pred.argmax(dim=1) == label).float().sum().item()
        total += label.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    app()
