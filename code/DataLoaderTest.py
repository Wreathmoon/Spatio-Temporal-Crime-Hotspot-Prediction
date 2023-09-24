import pytest
import h5py
import numpy as np
import torch

from DataPreLoader import DataPreLoader
import config



def test_dataloader():
    device = torch.device('cpu')
    prepDatasetsPath = config.PROJECT_DIR + '/Data/PreprocessedDatasets'
    test_data = DataPreLoader(prepDatasetsPath = prepDatasetsPath,
                                device=device,
                                name = 'test')
    
    for i,(x,y) in enumerate(test_data):
        if i == 0:
            assert x.shape == (config.SEQ_LEN, config.CRIME_TYPE_NUM, config.LON_GRIDS, config.LAT_GRIDS)