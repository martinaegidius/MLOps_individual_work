import torch
import typer
from mlops_individual.data import corrupt_mnist
from mlops_individual.model import MyAwesomeModel
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str = "none") -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = corrupt_mnist()
    testloader = DataLoader(test_set, batch_size=32, shuffle=False)
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    correct, total = 0, 0
    for img, label in testloader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        pred = model(img)
        correct += (pred.argmax(dim=1) == label).float().sum().item()
        total += label.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
