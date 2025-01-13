from torch.utils.data import Dataset
import torch 
from mlops_individual.data import corrupt_mnist
from tests import _N_CLASSES, _PATH_DATA
import os.path
import pytest 

@pytest.mark.skipif(not os.path.exists(_PATH_DATA+"processed/"), reason="Data files not found")
def test_my_dataset(): #for  pytest, the method name needs to start with test_*, and the py file needs to start with test_*.py
    """Test the MyDataset class."""
    N_train = 30000
    N_test = 5000
    N_classes = _N_CLASSES
    
    train, test = corrupt_mnist() #this is a dataset
    assert len(train) == N_train 
    assert len(test) == N_test
    for dataset in [train,test]:
        for x, y in dataset:
            assert x.shape == (1,28,28)
            assert y in range(10) #check no labels outside of label definitions
     
    #check that all classes actually exist 
    train_targets = torch.unique(train.tensors[1]) #gets unique values of train. sorted: defaults to True
    assert (train_targets==torch.arange(0,N_classes)).all() #.all(): tests if all elements in the tensor are true.
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets==torch.arange(0,N_classes)).all() #.all(): tests if all elements in the tensor are true.    
    assert isinstance(dataset, Dataset)
    print("test_my_dataset succeeded.")


if __name__=="__main__":
    test_my_dataset()