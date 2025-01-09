import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mlops_individual.model import MyAwesomeModel
import matplotlib.pyplot as plt
from mlops_individual.data import corrupt_mnist
import typer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# for running without tasks
# app = typer.Typer()
# @app.command()
def visualize(model_checkpoint: str = "none", figure_name: str = "tSNE.png"):
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    print(model)
    # model.layers.classification_head = nn.Identity() #kill classification head  <- one way
    model.layers = model.layers[:-1]
    print(model)
    model.eval()
    _, test_set = corrupt_mnist()
    testloader = DataLoader(test_set, batch_size=32, shuffle=False)

    # feature_extractor = nn.Sequential(*list(model.children())[-1])

    embeddings, labels = [], []
    with torch.inference_mode():  # similar to no_grad, but preferred as long as it doesn't cast an error
        for images, label in testloader:
            embedding = model(images)
            labels.append(label)
            embeddings.append(embedding)

        embeddings = torch.cat(embeddings).numpy()
        labels = torch.cat(labels).numpy()

    if embeddings.shape[-1] > 500:
        print("using tSNE on PCA of embeddings. N PCA components: 100")
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    n_classes = 10
    for i in range(n_classes):
        mask = labels == i  # get the labels corresponding to i
        plt.scatter(
            embeddings[mask, 0], embeddings[mask, 1], label=str(i)
        )  # index in embeddings only plotting tSNE for class = i, first and second component
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


# for running without tasks
# if __name__=="__main__":
#     app()


if __name__ == "__main__":
    typer.run(visualize)
