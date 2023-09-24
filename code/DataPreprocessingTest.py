import pytest
import pandas as pd
import numpy as np
from DataPreprocessing import DataPreprocessing
import config

@pytest.fixture()  
def data_preprocessing():
    return DataPreprocessing(projectDir=config.PROJECT_DIR)

def test_preprocessDataset(data_preprocessing):
    data = data_preprocessing.data
    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {'TYPE', 'date', 'age', 'gender', 'race', 'borough',
                                 'Unnamed: 0', 'precinct', 'type', 'X', 'Y', 'Latitude','Longitude','month_year'}
    
def test_readDataset(data_preprocessing):
    data = data_preprocessing.readDataset() 
    assert isinstance(data, pd.DataFrame)
    
def test_getPivotData(data_preprocessing):
    data_pivot = data_preprocessing.getPivotData()
    assert isinstance(data_pivot, pd.DataFrame)
    assert data_pivot.index.names == ['date', 'type']
    assert data_pivot.shape[1] == 2500  # 50 x 50 grids
    
def test_getFeatureLabel(data_preprocessing):
    features, labels, data_pivot = data_preprocessing.features, data_preprocessing.labels, data_preprocessing.dataPivot
    
    assert isinstance(features, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert isinstance(data_pivot, pd.DataFrame)
    
    assert features.shape[1] == 12 # seq_len
    assert labels.shape[1] == 8