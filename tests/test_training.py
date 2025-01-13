# import pytest
# from unittest.mock import patch, MagicMock
# import torch
# from mlops_individual.model import MyAwesomeModel
# from mlops_individual.data import corrupt_mnist
# from mlops_individual.train import train
# from hydra import initialize, compose
# from hydra.core.hydra_config import HydraConfig
# import hydra
# import sys 
# from omegaconf import DictConfig, OmegaConf


# #from tests import _N_CLASSES 

# @patch("mlops_individual.train.DataLoader") #is a mock
# @patch("mlops_individual.train.MyAwesomeModel") 
# @patch("mlops_individual.train.nn.CrossEntropyLoss")
# def test_training_loss_calculation(mock_loss, mock_model, mock_dataloader):
#     """
#     Test that the loss calculation in the training loop is performed correctly.
#     """

#     # Mock the model to return predictable logits
#     mock_model_instance = MagicMock()
#     mock_logits = torch.randn(32, 10)  # Batch of 32, 10 classes
#     mock_model_instance.return_value = mock_logits
#     mock_model.return_value = mock_model_instance

#     # Mock the loss function - initialize fields - it will be overwritten by train-loop 
#     mock_loss_instance = MagicMock()
#     mock_loss_instance.return_value = torch.tensor(1.0, requires_grad=True)  # Simulated loss value
#     mock_loss.return_value = mock_loss_instance

#     # Mock the data loader to return predictable inputs and labels
#     mock_dataloader.return_value = [(
#         torch.randn(32, 1, 28, 28),  # Batch of images
#         torch.randint(0, 10, (32,))  # Batch of labels
#     )]

#     with initialize(config_path="../configs", version_base=None):
#         cfg = compose(config_name="config")
#         # Simulate sys.argv for @hydra.main
#         sys.argv = ["train.py"]  # Add arguments if needed

#         # Import and run the train function
#         from mlops_individual.train import train
#         train(cfg)  # Call the function directly
#     # Verify the loss function was called with correct arguments
#     mock_loss_instance.assert_called()
#     for call in mock_loss_instance.call_args_list:
#         logits, labels = call[0]
#         assert logits.shape == (32, 10)  # Shape of logits
#         assert labels.shape == (32,)  # Shape of labels

#     # Check that the loss returned requires a gradient
#     assert mock_loss_instance.return_value.requires_grad is True

#     print("Loss calculation test passed!")


# if __name__=="__main__":
#     test_training_loss_calculation()
   


