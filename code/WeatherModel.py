# Luohao Xu edsml-lx122

import pickle
import os
import pandas as pd
import numpy as np
import random

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

from DataPreprocessing import DataPreprocessing

import config

class WeatherModel():
    def __init__(self, projectDir):
        """
        Function to initialize Weather model

        Input: projectDir <String>: directory of the project
        """
        
        self.projectDir =projectDir
        self.modelDir = self.projectDir + f"/Data/ModelWeights/WeatherModel_{config.CRIME_TYPE_NUM}types.pkl"
        
        self.weatherFeatures = projectDir + f"/Data/PreprocessedDatasets/weatherFeatures_{config.CRIME_TYPE_NUM}types.csv"
        self.weatherLabels = projectDir + f"/Data/PreprocessedDatasets/weatherLabels_{config.CRIME_TYPE_NUM}types.csv"
        # if there is no preprocessed data, run preprocessing and save the results.
        if not (os.path.isfile(self.weatherFeatures)):
            self.x, self.y = self.loadDataset()
        # load preprocessed data
        else:
            self.x = pd.read_csv(self.weatherFeatures)
            self.y = pd.read_csv(self.weatherLabels)
            self.x.set_index('Unnamed: 0', inplace=True)
            self.y.set_index('Unnamed: 0', inplace=True)
            # set datetime
            self.x.index = pd.to_datetime(self.x.index,format='%Y-%m-%d',errors='coerce')
            self.y.index = pd.to_datetime(self.y.index,format='%Y-%m-%d',errors='coerce')
        
        # if not trained model, train it
        if not (os.path.isfile(self.modelDir)):
            self.model = self.train()
        # load trained model
        else:
            with open(self.modelDir, 'rb') as file:
                self.model = pickle.load(file)
                file.close()
        
    def loadDataset(self):
        """
        Function to load and preprocess weather data

        Output: x <DataFrame>: weather features
                y <DataFrame>: total crime count per day (labels)
        """
        # load weather data
        weather_dataset_path = self.projectDir + "/Data/Datasets/new york city 2010-01-01 to 2022-12-31.csv"
        weather_data = pd.read_csv(weather_dataset_path)
        # set datetime
        weather_data['datetime'] = pd.to_datetime(weather_data['datetime'],format='%Y-%m-%d',errors='coerce')
        weather_data = weather_data.set_index('datetime')
        
        # load crime data
        dp = DataPreprocessing(self.projectDir)
        crimeData = dp.data
        crimeData = crimeData.pivot_table(index='date', columns='type', aggfunc='size').fillna(0)
        # get total crime count each day with all kinds of crime type
        data = pd.concat([weather_data, crimeData], axis=1)
        
        # change temperature degree to celsius
        data['F'] = (data['temp'] * 1.8) + 32

        # generate HDD and CDD data
        data['CDD'] = (data['F'] - 65).where(data['F'] >= 65, 0).astype(int)
        data['HDD'] = (65 - data['F']).where(data['F'] <  65, 0).astype(int)
        # drop columns that are not needed
        drop_columns = ['preciptype', 'windgust', 'severerisk', 'name', 'sunrise', 'sunset', 'conditions', 'description', 'icon', 'stations', 'F']
        data = data.drop(columns=drop_columns, axis=1)
        data = data.astype(np.float64)
        
        # get features and labels where labels are total crime count each day for all crime type
        crimeType = [crime.lower() for crime in config.CRIME_TYPE]
        y = data[crimeType].sum(axis=1)
        x = data.drop(columns=crimeType)
        
        x.to_csv(self.weatherFeatures)
        y.to_csv(self.weatherLabels)
        
        return x, y
       
    def train(self):
        """
        Function to train the weather model
        
        Output: model <object>: trained model
        """
        
        # split train and test dataset
        x_train,x_test,y_train,y_test = train_test_split(self.x, self.y, test_size=.1, random_state=42)
        
        # # random search for best parameter of the model 
        # model = RandomForestRegressor(random_state = 42)
        # grid_rf = {
        # 'n_estimators': [20, 50, 100, 200, 500, 800, 1000, 1500, 2000],  
        # 'max_depth': [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        # 'min_samples_split': [2, 5, 10], 
        # 'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
        # 'bootstrap': [True, False]
        # }
        
        # rscv = RandomizedSearchCV(estimator=model, 
        #                         param_distributions=grid_rf, 
        #                         cv=3, 
        #                         n_jobs=-1, 
        #                         verbose=2, 
        #                         n_iter=100)

        # rscv_fit = rscv.fit(x_train, y_train)
        # best_parameters = rscv_fit.best_params_
        # print(best_parameters)
        
        # generate random forest regressor model with the selected parameters
        # {'n_estimators': 800, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 50, 'bootstrap': True}
        model = RandomForestRegressor(n_estimators=800, random_state=42, min_samples_split=10, 
                              min_samples_leaf=1, max_depth=50, bootstrap=True)

        model.fit(x_train, y_train)
        # save model
        with open(self.modelDir, 'wb') as file:
            pickle.dump(model, file)
        
        # predict = model.predict(x_test)

        # print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
        # print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
        # print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
        # print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
        # errors = abs(predict - y_test.values)
        # mape = 100 * (errors / y_test.values)
        # accuracy = 100 - np.mean(mape)
        # print('Accuracy:', round(accuracy, 2), '%.') 
            
        return model
    
    def predict(self, date):
        """
        Function to predict total crime count of given date

        Input: date <String>: date for prediction

        Output: crimeCountPred <arr>: prediction result
        """
        # get weather features by searching for that date in the dataset
        weatherFeatures = self.x.loc[date].values
        # modify the datatype to match the original input to the model
        weatherFeatures = np.expand_dims(weatherFeatures, axis=0)

        crimeCountPred = self.model.predict(weatherFeatures)
        
        return crimeCountPred
    
    def getWeatherFactor(self, date):
        """
        Function to calculate weather factor
        
        Input: date <String>: date for prediction

        Output: weatherFactor <float>: factor of how crime count of that date different from the past 1 year's average
        """
        # get predicted value of that given date
        crimeCountPred = self.predict(date)
        
        # get average crime count of the last 365 days. if there is no enough days ahead, use the first year to calculate the average
        if date <= '2011-01-01':
            avg = self.y.loc['2010-01-01': '2010-12-31'].mean()
        else:
            # get the same date in the last year
            dateLastYear = datetime.strptime(date, '%Y-%m-%d') - pd.to_timedelta(365, unit='d')
            # change the date to string which has the same shape with the dataset
            dateLastYear = dateLastYear.strftime('%Y-%m-%d')
            avg = self.y.loc[dateLastYear: date].mean()
        
        # get weather factor based on different device type
        if (config.DEVICE == 'cpu'):
            weatherFactor = (crimeCountPred/avg)
        elif (config.DEVICE == 'cuda'):
            weatherFactor = (crimeCountPred/avg).values[0]

        return weatherFactor
    
    