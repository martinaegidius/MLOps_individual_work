from torch import nn
import torch
from collections import OrderedDict

class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        activation = nn.ReLU()
        conv1 = nn.Conv2d(in_channels=1,out_channels=4,kernel_size=3,stride=1,bias=True)
        conv2 = nn.Conv2d(in_channels=4,out_channels = 32,kernel_size=3,stride=1,bias=True)
        conv3 = nn.Conv2d(in_channels=32,out_channels = 64,kernel_size=3,stride=1,bias=True)
        dropout = nn.Dropout2d(p=0.2)

        self.layers = nn.Sequential(
            OrderedDict(
                [ 
                    ("conv1", conv1),
                    ("Dropout",dropout),
                    ("ReLU",activation),
                    ("MaxPool",maxpool),
                    ("conv2",conv2),
                    ("Dropout",dropout),
                    ("ReLU",activation),
                    ("MaxPool",maxpool),
                    ("conv3",conv3),
                    ("Flatten",nn.Flatten(start_dim=1,end_dim=-1)),
                    ("Classification head",nn.Linear(5184,10))
                ]
            )
        )
    
    def forward(self,x) -> torch.Tensor:
        output = self.layers(x) 
        return output
        
        


if __name__=="__main__":
    import os 
    import torch 
    data_dir = "/home/max/Documents/s194119/MLOps/dtu_mlops/s1_development_environment/exercise_files/final_exercise/data/corruptmnist_v1/"
    sample = "train_images_0.pt"
    f = data_dir+sample
    images = torch.load(f)
    print("images shape: ",images.shape)
    images = images.unsqueeze(1)
    print("images unsqueezed shape: ",images.shape)
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    output = model(images)
    print("output: ",output)
    print("\n with shape: ",output.shape)

