# Luohao Xu edsml-lx122

import os
import folium
import torch
import pickle
import pandas as pd
import numpy as np
import branca.colormap as cm
from tqdm import tqdm
import streamlit as st
from datetime import datetime, timedelta

import folium.plugins as plugins
from folium import LayerControl
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
from geopandas import GeoDataFrame
from shapely.geometry import Point

import config
from LSTMModel import ConvLSTMModel
from DataPreprocessing import DataPreprocessing
from WeatherModel import WeatherModel
from TimeseriesModel import TimeseriesModel



class VisualizationTool():


    def __init__(self, projectDir=config.PROJECT_DIR):
        self.projectDir = projectDir
        
        self.minus_days = config.SEQ_LEN + 1
        self.start_date = datetime.strptime(config.START_DATE[1:-1], '%Y-%m-%d')
        self.left_limit = self.start_date + timedelta(days=self.minus_days)
        self.right_limit = datetime.strptime(config.END_DATE[1:-1], '%Y-%m-%d')
        self.crimeType = [crime.lower() for crime in config.CRIME_TYPE]
        
        self.device = 'cpu' 
        
        # print(self.left_limit)
        # print(self.right_limit)
        
        self.LSTM_model = self.load_LSTM()
        self.features, self.labels, self.dataPivot, self.crimeData = self.loadDataset()
        self.NYCShape = self.initialize_NYC_shape()
        
        self.weatherModel = WeatherModel(self.projectDir)
        self.Timeseries_model = TimeseriesModel(self.projectDir, self.crimeData)
        
        
    def initialize_NYC_shape(self):

        NYCShapeDir = self.projectDir + '/Data/PreprocessedDatasets/NYCGridsShape.pkl'

        if (os.path.isfile(NYCShapeDir)):
            with open(NYCShapeDir, 'rb') as file:
                NYCShape = pickle.load(file)
                file.close()
                return NYCShape

        else:
            # Get the shape-file for NYC
            boros = GeoDataFrame.from_file(self.projectDir + '/Data/ShapeBorough/geo_export_21e663f4-eeca-4db4-a956-0f0928ed3b37.shp')
            
            NYCShape = []
            x_list = np.linspace(start=0, stop=config.LAT_GRIDS-1, num=config.LAT_GRIDS).astype('int')
            y_list = np.linspace(start=0, stop=config.LON_GRIDS-1, num=config.LON_GRIDS).astype('int')

            print("Initializing NYC map...")
            # latest_iteration = st.empty()
            # bar = st.progress(0)
            for i,x in enumerate(x_list):
                # latest_iteration.text(f'Iteration {2*i+1}')
                # bar.progress(2*i + 1)
                for y in y_list:
                    lat, lon = config.grid2coord(x,y)
                    point = Point(lon, lat)
                    in_NYC_or_Not = np.array([point.within(shape) for shape in boros.geometry]).sum(axis=0)

                    if not in_NYC_or_Not:
                        NYCShape.append((x,y))

            with open(NYCShapeDir, 'wb') as file:
                pickle.dump(NYCShape, file)

        print("NYC map loaded")
        return NYCShape
    
    
    def load_LSTM(self):

        optim_name = 'Adam'
        model_save_path = self.projectDir + '/Data/ModelWeights' + f'/BestModel__bs-({config.TRAIN_BATCH_SIZE})_threshold-({config.CLASS_THRESH})_weights-({config.BCE_WEIGHTS}).pt'
        model = torch.load(model_save_path, map_location=torch.device(self.device) )

        LSTM_model = ConvLSTMModel(input_dim=config.CRIME_TYPE_NUM, hidden_dim=config.HIDDEN_DIM, kernel_size=config.KERNEL_SIZE, bias=True)
        LSTM_model.load_state_dict(model['model'])
        
        print('LSTM model loaded')
        return LSTM_model
        
    
    def loadDataset(self):
        
        dp = DataPreprocessing(self.projectDir)
        features, labels, dataPivot, crimeData = dp.features, dp.labels, dp.dataPivot, dp.data
        
        print("Dataset loaded")
        return features, labels, dataPivot, crimeData, dp
    
    
    def find_data_by_date(self, date):

        dt = datetime.strptime(date[1:-1], '%Y-%m-%d')
        if (dt <= self.left_limit):
            print(f"Please choose date after {self.start_date}.", end=" ")
            print("The crime data before that date is not applied due to limited computing resources.")
            return 0
        elif (dt > self.right_limit):
            print(f"Please choose data before {self.right_limit}.")
            print("Currently the model can not access future data for prediction.(Use data of last 12 days to predict on that day)")
            return 0
        
        minus_days = config.SEQ_LEN + 1
        if (self.dataPivot.query(f"date < {config.START_DATE}").shape[0] == 0):
            startIndex = 0
        else:
            startIndex = int(self.dataPivot.query(f"date < {config.START_DATE}").shape[0] / config.CRIME_TYPE_NUM - minus_days)
        
        found_index = int(self.dataPivot.query(f"date < {date}").shape[0] / config.CRIME_TYPE_NUM - minus_days) - startIndex
        labels_by_date = self.labels[found_index]
        features_by_date = self.features[found_index]
        
        processed_features = torch.from_numpy(features_by_date).to(self.device).unsqueeze(0).float()
        pred_data = self.LSTM_model(processed_features)[0][0]
        
        getWeatherFactor = self.weatherModel.getWeatherFactor(date[1:-1])
        getTimeseriesFactor = [self.Timeseries_model.getTimeseriesFactor(crime_name, date[1:-1]) for crime_name in self.crimeType]
        
        return pred_data, labels_by_date, getWeatherFactor, getTimeseriesFactor
        
    
    def gridmap(self, date, threshold, temporal_factor=False, show_real_data=True):
        
        pred_data, real_data, getWeatherFactor, getTimeseriesFactor = self.find_data_by_date(date)
        
        #Create a empty folium map object
        gridmap = folium.Map(location=[(config.LAT_MIN+config.LAT_MAX)/2, (config.LON_MIN+config.LON_MAX)/2],
                             zoom_start=11,
                             tiles='OpenStreetMap',
                             name='GridMap') 
        # colorbar
        linear = cm.LinearColormap(['green','yellow','red'], vmin=0., vmax=1.)
        
        featureGroupList = [folium.FeatureGroup(name=name) for name in config.CRIME_TYPE]
        # ROBBERY = folium.FeatureGroup(name="Robbery")
        # BURGLARY = folium.FeatureGroup(name="Burglary")

        for x in range(pred_data.shape[1]):
            for y in range(pred_data.shape[2]):
                for type_num in range(config.CRIME_TYPE_NUM):
                    if (((x,y) not in self.NYCShape) or ((x+1,y) not in self.NYCShape) or ((x,y+1) not in self.NYCShape) or ((x+1,y+1) not in self.NYCShape)) and x < config.LAT_GRIDS-1 and y < config.LON_GRIDS-1:
                        if pred_data[type_num][x][y] < threshold:
                            weight = np.float64(pred_data[type_num][x][y]) * config.MULTIPLY_FACTOR
                        else:
                            weight = np.float64(pred_data[type_num][x][y])
                            
                        if temporal_factor:
                            weight = weight * getWeatherFactor * getTimeseriesFactor[type_num]
                            
                        fill_color = linear(weight)
                        
                        rec = folium.Rectangle(bounds=[(config.LAT_BINS[x],config.LON_BINS[y]), (config.LAT_BINS[x+1],config.LON_BINS[y+1])], 
                                                color='red', 
                                                fill=True, 
                                                fill_color=fill_color, 
                                                fill_opacity=0.5,
                                                stroke=False)#.add_to(gridmap)
                        
                        featureGroupList[type_num].add_child(rec)

        for i in range(len(featureGroupList)):   
            gridmap.add_child(featureGroupList[i])
        gridmap.add_child(folium.LayerControl())
        
        plugins.Geocoder().add_to(gridmap)
        
        selected_data = self.crimeData[self.crimeData['date'] == date[1:-1]]
        
        for index, row in selected_data.iterrows():
            #print(row)
            location = row['Latitude'], row['Longitude']
            # icon=folium.Icon(color='purple', icon='glyphicon-cutlery', prefix='glyphicon')
            html = '''Age Group: ''' + row['age'] + '''<br>Gender: ''' + row['gender'] + '''<br>Race: ''' + row['race']
            iframe = folium.IFrame(html, width=150, height=100)
            popup = folium.Popup(iframe, max_width=300)
            marker = folium.Marker(location=location, popup=popup)
            #folium.Popup(popup).add_to(marker)
            
            # if row['type'] == 'robbery':
            #     ROBBERY.add_child(marker)
            # if row['type'] == 'burglary':
            #     BURGLARY.add_child(marker)
            index = config.CRIME_TYPE.index(row['type'].upper())
            featureGroupList[index].add_child(marker)
                
               
        return gridmap
    
    
    def heatmap(self, date, threshold, temporal_factor=False, show_real_data=True):
        
        pred_data, real_data, getWeatherFactor, getTimeseriesFactor = self.find_data_by_date(date)
        
        featureGroupList = [folium.FeatureGroup(name=name) for name in config.CRIME_TYPE]
        # ROBBERY = folium.FeatureGroup(name="Robbery")
        # BURGLARY = folium.FeatureGroup(name="Burglary")
        
        # robbery_lat_lon_weight = []
        # burglary_lat_lon_weight = []
        lat_lon_list_weight = [[] for _ in range(config.CRIME_TYPE_NUM)]
        for x in range(pred_data.shape[1]):
            for y in range(pred_data.shape[2]):
                for type_num in range(config.CRIME_TYPE_NUM):
                    if (((x,y) not in self.NYCShape) or ((x+1,y) not in self.NYCShape) or ((x,y+1) not in self.NYCShape) or ((x+1,y+1) not in self.NYCShape)) and x < config.LAT_GRIDS-1 and y < config.LON_GRIDS-1:
                        if pred_data[type_num][x][y] < threshold:
                            weight = np.float64(pred_data[type_num][x][y]) * config.MULTIPLY_FACTOR
                        else:
                            weight = np.float64(pred_data[type_num][x][y])
                            
                        if temporal_factor:
                            weight = weight * getWeatherFactor * getTimeseriesFactor[type_num]
                        
                        
                        lat = config.LAT_BINS[x] + config.DIFF_LAT
                        lon = config.LON_BINS[y] + config.DIFF_LON
                        
                        # if type_num == 0:
                        #     robbery_lat_lon_weight.append((lat, lon, weight))
                        # elif type_num == 1:
                        #     burglary_lat_lon_weight.append((lat, lon, weight))
                        
                        lat_lon_list_weight[type_num].append((lat, lon, weight))
                            
                        
                            
                        
        heatmap = folium.Map(location=[(config.LAT_MIN+config.LAT_MAX)/2, (config.LON_MIN+config.LON_MAX)/2],
                 zoom_start=11, tiles='OpenStreetMap') 
        
        # #Plot it on the map
        # HeatMap(robbery_lat_lon_weight,
        #         min_opacity=0.2,
        #         max_opacity=1,
        #         radius=20,
        #         name="Robbery").add_to(ROBBERY)
        
        # HeatMap(burglary_lat_lon_weight,
        #         min_opacity=0.2,
        #         max_opacity=1,
        #         radius=20,
        #         name="Burglary").add_to(BURGLARY)
        
        for i in range(config.CRIME_TYPE_NUM):
            HeatMap(lat_lon_list_weight[i],
                min_opacity=0.2,
                max_opacity=1,
                radius=20,
                name=config.CRIME_TYPE[i].lower).add_to(featureGroupList[i])
            
            heatmap.add_child(featureGroupList[i])
        
        # heatmap.add_child(ROBBERY)
        # heatmap.add_child(BURGLARY)
        heatmap.add_child(folium.LayerControl())
        plugins.Geocoder().add_to(heatmap)
        
        selected_data = self.crimeData[self.crimeData['date'] == date[1:-1]]
        
        for index, row in selected_data.iterrows():
            #print(row)
            location = row['Latitude'], row['Longitude']
            # icon=folium.Icon(color='purple', icon='glyphicon-cutlery', prefix='glyphicon')
            html = '''Age Group: ''' + row['age'] + '''<br>Gender: ''' + row['gender'] + '''<br>Race: ''' + row['race']
            iframe = folium.IFrame(html, width=150, height=100)
            popup = folium.Popup(iframe, max_width=300)
            marker = folium.Marker(location=location, popup=popup)
            #folium.Popup(popup).add_to(marker)
            # if row['type'] == 'robbery':
            #     ROBBERY.add_child(marker)
            # if row['type'] == 'burglary':
            #     BURGLARY.add_child(marker)
            index = config.CRIME_TYPE.index(row['type'].upper())
            featureGroupList[index].add_child(marker)
        
        return heatmap
    