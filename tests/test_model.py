import torch 

from mlops_individual.model import MyAwesomeModel
from tests import _N_CLASSES
import pytest 


N_CLASSES = _N_CLASSES #should in principle be inherited from dataset config such that it works for all possible datasets
    

# def test_model(): #old
#     model = MyAwesomeModel()
#     single_batch = torch.randn(1,1,28,28)
#     batch = torch.randn(37,1,28,28)
#     assert model(single_batch).shape == (1,N_CLASSES)
#     assert model(batch).shape==(batch.shape[0],N_CLASSES)
#     print("model forward check passed")

@pytest.mark.parametrize("batch_size", [1,32,67])
def test_model(batch_size: int):
    model = MyAwesomeModel()
    x = torch.randn(batch_size,1,28,28)
    y = model(x)
    assert y.shape == (batch_size,N_CLASSES), "Output shape does not match (batch_size x N_CLASSES)"


def NOT_IMPLEMENTED_test_model_wrong_input():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        model(torch.randn(1,2,3))
    print("model forward check passed")

    with pytest.raises(ValueError, match='Expected each sample to have shape [1, 28, 28]'):
        model(torch.randn(1,1,28,29))



if __name__ == "__main__":
    test_model()