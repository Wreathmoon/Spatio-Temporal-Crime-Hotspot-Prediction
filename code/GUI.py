# Luohao Xu edsml-lx122

import os
import torch
import pickle
import pandas as pd
import numpy as np
import branca.colormap as cm
import streamlit as st
import pydeck as pdk
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
from datetime import datetime, timedelta
from streamlit_folium import folium_static
from geopandas import GeoDataFrame
from shapely.geometry import Point

import config
from LSTMModel import ConvLSTMModel
from DataPreprocessing import DataPreprocessing
from WeatherModel import WeatherModel
from TimeseriesModel import TimeseriesModel

device = 'cpu'
projectDir = config.PROJECT_DIR
minus_days = config.SEQ_LEN + 1
start_date = datetime.strptime(config.START_DATE[1:-1], '%Y-%m-%d')
left_limit = start_date + timedelta(days=minus_days)
right_limit = datetime.strptime(config.END_DATE[1:-1], '%Y-%m-%d')
crimeType = [crime.lower() for crime in config.CRIME_TYPE]

# in order to use the cache in streamlit, functions can only be written in isolated function instead of a whole class
@st.cache_data
def loadNYCShape():
    """
    Function to load NYC shape and save it in cache

    Output: NYCShape <list>: list of grids that are not on the map.
    """
    print("Initializing NYC Map")
    NYCShapeDir = projectDir + '/Data/PreprocessedDatasets/NYCGridsShape.pkl'
    # load pickle file if it was trainde before
    if (os.path.isfile(NYCShapeDir)):
        with open(NYCShapeDir, 'rb') as file:
            NYCShape = pickle.load(file)
            return NYCShape

    else:
        # Get the shape-file for NYC
        boros = GeoDataFrame.from_file(projectDir + '/Data/ShapeBorough/geo_export_21e663f4-eeca-4db4-a956-0f0928ed3b37.shp')
        
        NYCShape = []
        x_list = np.linspace(start=0, stop=config.LAT_GRIDS-1, num=config.LAT_GRIDS).astype('int')
        y_list = np.linspace(start=0, stop=config.LON_GRIDS-1, num=config.LON_GRIDS).astype('int')

        print("Generating NYC map...")

        for x in x_list:
            for y in y_list:
                # convert grid number to latitude and longitude
                lat, lon = config.grid2coord(x,y)
                point = Point(lon, lat)
                # determine if current point is in any shape of NYC
                in_NYC_or_Not = np.array([point.within(shape) for shape in boros.geometry]).sum(axis=0)

                if not in_NYC_or_Not:
                    NYCShape.append((x,y))

        with open(NYCShapeDir, 'wb') as file:
            pickle.dump(NYCShape, file)

    return NYCShape

@st.cache_data
def loadDataset():
    """
    Function to load dataset and save it to cache

    Output: features<DataFrame>: features
            labels<DataFrame>: labels
            dataPivot<DataFrame>: crime table
            crimeData<DataFrame>: crime data info
    """
    print("Initilizing Dataset")
    dp = DataPreprocessing(projectDir)
    features, labels, dataPivot, crimeData = dp.features, dp.labels, dp.dataPivot, dp.data
    return features, labels, dataPivot, crimeData

@st.cache_resource
def loadLSTMModel():
    """
    Function to load ConvLSTM model and save it to cache

    Output: LSTM_model<Object>: loaded ConvLSTM model
    """
    print('Loading ConvLSTM Model')
    model_save_path = projectDir + '/Data/ModelWeights' + f'/BestModel__bs-({config.TRAIN_BATCH_SIZE})_threshold-({config.CLASS_THRESH})_weights-({config.BCE_WEIGHTS}).pt'
    model = torch.load(model_save_path, map_location=torch.device(device) )
    LSTM_model = ConvLSTMModel(input_dim=config.CRIME_TYPE_NUM, hidden_dim=config.HIDDEN_DIM, kernel_size=config.KERNEL_SIZE, bias=True)
    LSTM_model.load_state_dict(model['model'])
    return LSTM_model

@st.cache_resource
def loadWeatherModel():
    """
    Function to load Weather model and save it to cache

    Output: WeatherModel<Object>: loaded Weather model
    """
    print('Loading Weather Model')
    return WeatherModel(projectDir)

@st.cache_resource
def loadTimeseriesModel(crimeData):
    """
    Function to load Timeseries model and save it to cache

    Output: Timeseries<Object>: loaded Timeseries model
    """
    print('Loading Timeseries Model')
    return TimeseriesModel(projectDir, crimeData)

def getPredDataByDate(date, LSTMModel, weatherModel, timeseriesModel, dataPivot, features, labels):
    """
    Function to get predicted data by using those three models

    Input: date<String>: selected date
           LSTMModel<Object>: loaded LSTMModel
           weatherModel<Object>: loaded weatherModel
           timeseriesModel<Object>: loaded timeseriesModel
           dataPivot<DataFrame>: crime timetable
           features<DataFrame>: features
           labels<DataFrame>: labels
    """

    dt = datetime.strptime(date[1:-1], '%Y-%m-%d')
    if (dt <= left_limit):
        print(f"Please choose date after {start_date}.", end=" ")
        print("The crime data before that date is not applied due to limited computing resources.")
        return 0
    elif (dt > right_limit):
        print(f"Please choose data before {right_limit}.")
        print("Currently the model can not access future data for prediction.(Use data of last 12 days to predict on that day)")
        return 0
    
    # determine if the input date is valid for prediction
    minus_days = config.SEQ_LEN + 1
    if (dataPivot.query(f"date < {config.START_DATE}").shape[0] == 0):
        startIndex = 0
    else:
        startIndex = int(dataPivot.query(f"date < {config.START_DATE}").shape[0] / config.CRIME_TYPE_NUM - minus_days)
    
    # get input feature and true labels
    found_index = int(dataPivot.query(f"date < {date}").shape[0] / config.CRIME_TYPE_NUM - minus_days) - startIndex
    labels_by_date = labels[found_index]
    features_by_date = features[found_index]
    
    # get pred from ConvLSTM
    processed_features = torch.from_numpy(features_by_date).to(device).unsqueeze(0).float()
    pred_data = LSTMModel(processed_features)[0][0]
    
    getWeatherFactor = weatherModel.getWeatherFactor(date[1:-1])
    getTimeseriesFactor = [timeseriesModel.getTimeseriesFactor(crime_name, date[1:-1]) for crime_name in crimeType]
    
    return pred_data, labels_by_date, getWeatherFactor, getTimeseriesFactor

def getHexagonData(pred_data, getWeatherFactor, getTimeseriesFactor, NYCShape, type_num, threshold, temporal_factor = True):
    """
    Function to get hexagon data

    Input: date<String>: selected date
           getWeatherFactor<float>: weather factor
           getTimeseriesFactor<float>: timeseries factor
           NYCShape<list>: gird number that are not in NYC
           type_num<int>: selected crime type index number
           threshold<float>: threshold for prediction
    
           Output<DataFrame>: latitude longitude list for plot 3d map
    """
    lat_lon_list = []
    for x in range(pred_data.shape[1]):
        for y in range(pred_data.shape[2]):
            if (((x,y) not in NYCShape) or ((x+1,y) not in NYCShape) or ((x,y+1) not in NYCShape) or ((x+1,y+1) not in NYCShape)) and x < config.LAT_GRIDS-1 and y < config.LON_GRIDS-1:
                
                weight = np.float64(pred_data[type_num][x][y])

                if temporal_factor:
                    weight = weight * getWeatherFactor * getTimeseriesFactor[type_num]
                if pred_data[type_num][x][y] < threshold:
                    weight = weight * config.MULTIPLY_FACTOR
                    
                lat = config.LAT_BINS[x] + config.DIFF_LAT
                lon = config.LON_BINS[y] + config.DIFF_LON

                num = int(weight*100)
                for _ in range(num):
                    lat_lon_list.append(np.array([lat, lon]))

                # # make columns more clear and beautiful
                # if (weight > 0.7):
                #     num = int(weight*10)
                #     for _ in range(num):
                #         lat_lon_list.append(np.array([lat, lon]))
                
                # if (weight > 0.9):
                #     num = int(weight*30)
                #     for _ in range(num):
                #         lat_lon_list.append(np.array([lat, lon]))

    df = pd.DataFrame(lat_lon_list, columns=['lat', 'lon'])

    return df

def run():
    """
    Function to run the GUI
    """
    # load NYC shape and datasets
    NYCShape = loadNYCShape()
    features, labels, dataPivot, crimeData = loadDataset()
    # load models
    LSTMModel = loadLSTMModel()
    weatherModel = loadWeatherModel()
    timeseriesModel = loadTimeseriesModel(crimeData)

    # limited by timeseries model
    startDate  = datetime.strptime(config.START_SELECT_DATE[1:-1], '%Y-%m-%d')
    endDate    = datetime.strptime(config.END_DATE[1:-1], '%Y-%m-%d')
    
    # Input parameters in the sidebar
    with st.sidebar:
        st.sidebar.write("Choose Parameters for Prediction")
        with st.form("my_form"):
            # select data in the given range
            dateChosen = st.date_input("Choose date:",  min_value=startDate, max_value=endDate, value=startDate)
            dataChosenStr = dateChosen.strftime('%Y-%m-%d')
            inputDate = f"\'{dataChosenStr}\'"

            # select one of eight crime type
            typeChosen = st.radio("Select crime type:", crimeType)
            type_num = crimeType.index(typeChosen)

            # select threshold for prediction
            threshold = st.select_slider("Adjust threshold:", options=[i/100 for i in range(1,100)])
            # button for start processing prediction
            submitted = st.form_submit_button("Predict")
    # default map when initializing
    if not submitted:
        st.write()
        "👈 👈 👈 Please choose parameters for prediction"
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                longitude = (config.LON_MIN+config.LON_MAX)/2,
                latitude = (config.LAT_MIN+config.LAT_MAX)/2,
                zoom=10,
                pitch=50,
            ),
            layers=[]
        ))

    # plot 3d interactive map by using pydeck_chart
    if submitted:
        st.write()
        # show selected parameters
        'You selected: ', typeChosen, 'on date ', dateChosen, 'with threshold of ', threshold
        'Prediction Results: '
        pred_data, real_data, getWeatherFactor, getTimeseriesFactor = getPredDataByDate(inputDate, LSTMModel, weatherModel, timeseriesModel, dataPivot, features, labels)
        chart_data = getHexagonData(pred_data, getWeatherFactor, getTimeseriesFactor, NYCShape, type_num, threshold)
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                longitude = (config.LON_MIN+config.LON_MAX)/2,
                latitude = (config.LAT_MIN+config.LAT_MAX)/2,
                zoom=10,
                pitch=40,
            ),
            layers=[
                pdk.Layer(
                'HexagonLayer',
                data=chart_data,
                get_position='[lon, lat]',
                radius=400,
                elevation_scale=1,
                elevation_range=[0, 8000],
                auto_highlight = True,
                pickable=True,
                extruded=True,
                )
            ],
        ))

if __name__ == "__main__":
    run()



