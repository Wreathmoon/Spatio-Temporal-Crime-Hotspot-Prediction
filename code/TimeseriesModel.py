# Luohao Xu edsml-lx122

import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config


class TimeseriesModel():
    def __init__(self, projectDir, dpData):
        """
        Function to initialise timeseries model

        Input: projectDir <String>: project directory
               dpData <DataFrame>: Preprocessed original data
        """
        self.projectDir = projectDir
        self.dpData = dpData
        self.crimeType = [crime.lower() for crime in config.CRIME_TYPE]
        self.lags = [f'{i}-months-lag' for i in range(1, 12)]
        self.datasets = [self.loadDataset(crimeTypeName) for crimeTypeName in self.crimeType]
        self.models = [self.train(crimeTypeName) for crimeTypeName in self.crimeType]
    
    def loadDataset(self, crimeTypeName):
        """
        Function to load dataset for the timeseries model

        Input: crimeTypeName <String>: name of the crime type (in lowercase)

        Output: timeseriesData <DataFrame>: processed timeseries data in every month
        """

        # if there is preprocessed dataset, load it and return
        timeseriesDataDir = self.projectDir + f"/Data/PreprocessedDatasets/timeseries_{crimeTypeName}.csv"
        if (os.path.isfile(timeseriesDataDir)):
            return pd.read_csv(timeseriesDataDir,index_col='date')
        
        print("Loading time series data")
        crimeData = self.dpData
        crimeData = crimeData.pivot_table(index='date', columns='type', aggfunc='size').fillna(0)
        
        # get crime count for every month
        timeseriesData = crimeData.resample('MS').sum()
        # select data for current crime type
        timeseriesData = pd.DataFrame(timeseriesData[crimeTypeName], columns = [crimeTypeName])
        # save dataset
        timeseriesData.to_csv(timeseriesDataDir)
        
        return timeseriesData
    
    def train(self, crimeTypeName):
        """
        Function to load or train the timeseries model

        Input: crimeTypeName<String>: crime type (lowercase)

        Output: model <Object>: trained or loaded model for given crime type
        """
        model_dir = self.projectDir + f'/Data/ModelWeights/TimeSeriesModel_{crimeTypeName}.pkl'
        # check if there is pretrained model, if so, load it and return
        if (os.path.isfile(model_dir)):
            with open(model_dir, 'rb') as file:
                model = pickle.load(file)
                file.close()
            return model
        
        print("Training timeseries model")
        timeseriesData = self.datasets[self.crimeType.index(crimeTypeName)][crimeTypeName]
        # using AutoARIMA to search for parameters with best results
        sarimax = auto_arima(timeseriesData, 
                             start_p = 1, start_q = 1,
                             test='adf', n_jobs=-1,
                            max_p = 3, max_q = 3, m = 12,
                            start_P = 0, seasonal = True,
                            stationary=True,
                            information_criterion='aic',
                            error_action ='ignore',   
                            suppress_warnings = True,  
                            stepwise = False,)       
        # create SARIMAX model with those chosen parameters
        model = SARIMAX(timeseriesData, 
                order = sarimax.order, 
                seasonal_order =sarimax.seasonal_order).fit()
        # save model
        with open(model_dir, 'wb') as file:
            pickle.dump(model, file)
    
        return model
        
    def predict(self, crimeTypeName, date):
        """
        Function to predict monthly crime count for given date and crime type

        Input: crimeTypeName <String>: name of crime type (lowercase)
               date <String>: date to be predicted
        
        Output: crimeCountPredMonthly <float>: predicted monthly crime count for given date and crime type
        """
        # load dataset and trained model foe given crime type
        data = self.datasets[self.crimeType.index(crimeTypeName)]
        model = self.models[self.crimeType.index(crimeTypeName)]
        # change the input date to the first day of that month with the same shape as dataset
        date = date[:-2] + '01'
        # looking for row number of that date in the dataset
        rowNum = np.where(data.index == date)[0][0]

        crimeCountPredMonthly = model.predict(rowNum,rowNum)
        
        return crimeCountPredMonthly
    
    def getTimeseriesFactor(self, crimeTypeName, date):
        """
        Function to get monthly timeseries factor for given date and crime type

        Input: crimeTypeName <String>: name of crime type (lowercase)
               date <String>: date to be predicted
        
        Output: timeseriesFactor <float>: predicted monthly timeseries factor for given date and crime type
        """
        # get predicted monthly crime count
        crimeCountPredMonthly = self.predict(crimeTypeName, date)

        # change the input date to the first date of that month with the same value as the one in dataset
        date = date[:-2] + '01'
        
        # get dataset of given crime type
        data = self.datasets[self.crimeType.index(crimeTypeName)][crimeTypeName]
        
        # get average monthly crime count for the last year
        # if there are no enough months ahead, use the first year as range
        if date <= '2011-01-01':
            avg = data.loc['2010-01-01': '2010-12-01'].mean()
        else:
            # calculate the same date in the last year
            dateLastYear = datetime.strptime(date, '%Y-%m-%d') - pd.to_timedelta(365, unit='d')
            # change the result to the same type as in the dataset
            dateLastYear = dateLastYear.strftime('%Y-%m-%d')
            dateLastYear = dateLastYear[:-2] + '01'
            avg = data.loc[dateLastYear: date].mean()
        
        timeseriesFactor = crimeCountPredMonthly[0] / avg
        return timeseriesFactor
    
    
    
