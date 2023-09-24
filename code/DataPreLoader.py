# Luohao Xu edsml-lx122

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

import config
from DataPreprocessing import DataPreprocessing

class DataPreLoader(Dataset):

    def __init__(self, prepDatasetsPath, device, name):
        '''
        Function to initialise data loader

        Input:  prepDatasetsPath <String>: path of the preprocessed datasets.
                device <String> : 'cpu' or 'cuda'
                name <String> : 'train' or 'test' or 'val'
        '''
        self.dp = DataPreprocessing(config.PROJECT_DIR)
        self.device = device
        self.features = self.read_h5(dataDir = prepDatasetsPath+f'/{config.CRIME_TYPE_NUM}_trainvaltest_features.h5', name = name)
        self.labels = self.read_h5(dataDir = prepDatasetsPath+f'/{config.CRIME_TYPE_NUM}_trainvaltest_labels.h5', name = name)
        self.labels = self.labels.reshape(-1,config.CRIME_TYPE_NUM * int(config.LAT_GRIDS * config.LON_GRIDS))

        self.X, self.Y = self.numpy2tensor(self.features, self.labels)
        self.num = self.labels.shape[0]

    def read_h5(self, dataDir, name):
        '''
        Function to read h5 file

        Input: dataDir <String> : h5 file directory
               name <String> : 'train' or 'test' or 'val'
        '''
        hf = h5py.File(dataDir, 'r')
        data = np.array(hf[name][:])
        return data

    def numpy2tensor(self, features, labels):
        '''
        Function to load tensors

        Input: features <array>: features
               labels <array>: labels

        Output: X <array>: array in tensor
                Y <array>: array in tensor
        '''
        X = torch.from_numpy(features).to(self.device)
        Y = torch.from_numpy(labels).to(self.device)
        return X, Y
    
    def __len__(self):
        '''
        Function to get dataset length

        Output: self.num <int>: length of dataset
        '''
        return self.num

    def __getitem__(self, idx):
        '''
        Function to get batched features and binned targets

        Input: idx <int>: index

        Output: self.X[idx].float() <tensor>: batched features in dtype of float
                self.Y[idx].float() <tensor>: batched labels in dtype of float
        '''
        return self.X[idx].float(), self.Y[idx].float()
    