# Luohao Xu edsml-lx122

import os
import h5py
import pandas as pd
import numpy as np

import config


class DataPreprocessing():
    def __init__(self, projectDir):
        """
        Function to initialise class variables

        Input: projectDir <String>: directory of the project.

        Output: none.
        """
        self.projectDir = projectDir
        self.datasetDir = self.projectDir +  '/Data/Datasets/NYPD_Arrests_Data__Historic_.csv'
        self.data = self.readDataset()

        self.dataPivotDir = self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_data_pivot.csv'
        if not (os.path.isfile(self.dataPivotDir)):
            self.features, self.labels, self.dataPivot = self.getFeatureLabel()
        else:
            print("Loading pivot data, features and labels")
            self.dataPivot = pd.read_csv(self.dataPivotDir, index_col=[0,1]) 

            hfFeatures = h5py.File(self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_features.h5', 'r')
            self.features = np.array(hfFeatures['features'][:])
            hfLabels= h5py.File(self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_labels.h5', 'r')
            self.labels = np.array(hfLabels['labels'][:])

        self.trainDatasetSaveDir = self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_trainvaltest_features.h5'
        self.trainDatasetSaveDirLable = self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_trainvaltest_labels.h5'
        if not (os.path.isfile(self.trainDatasetSaveDir)):
            self.getTrainValTest()
        if not (os.path.isfile(self.trainDatasetSaveDirLable)):
            self.getTrainValTest()
        
    def preprocessDataset(self, save=True):
        """
        Function to preprocess the original NYPD arrest dataset
        by screened by crime types
        various useless invalid data were eliminated
        data labels and data types were reset

        Input:  save <Boolean>: save the preprocessed dataset to the given directory or not.

        Output: data <DataFrame>: preprocessed dataset.
        """
        
        print("Preporcessing dataset...")
        
        # load original dataset
        rawData = pd.read_csv(self.datasetDir)
        
        # select cirme type from configuration
        data = rawData[rawData['OFNS_DESC'].isin(config.CRIME_TYPE)]
        
        # select columns that are needed
        data = data[['OFNS_DESC', 'ARREST_DATE', 'AGE_GROUP', 'PERP_SEX', 'PERP_RACE', 
                     'Latitude', 'Longitude', 'ARREST_BORO', 'ARREST_PRECINCT']]
        
        # rename and lowercase the column name
        data.rename(columns = {'OFNS_DESC':'TYPE', 'ARREST_DATE':'date', 'AGE_GROUP':'age', 
                               'PERP_SEX':'gender', 'PERP_RACE':'race',
                               'ARREST_BORO': 'borough', 'ARREST_PRECINCT': 'precinct'
                               }, inplace = True)
        
        # add a new type column
        data['type'] = data['TYPE'].str.lower()
        
        # clear data outside age groups
        indexKeepAge = (data['age']=='25-44') | (data['age']=='45-64') | (data['age']=='18-24') | (data['age']=='<18') | (data['age']=='65+')
        data = data[indexKeepAge]

        # clear data with null race
        indexKeepRace = (data['race'] != "UNKNOWN") & (data['race'] != "OTHER")
        data = data[indexKeepRace]

        # set datetime
        data['date'] = pd.to_datetime(data['date'],format='%m/%d/%Y',errors='coerce')
        data = data[data['date'] >= pd.to_datetime('2010-1-1')]
        data = data.dropna()
        
        # add month_year data
        data['month_year'] = pd.DatetimeIndex(data['date']).month_name() + ' ' + pd.DatetimeIndex(data['date']).year.astype('string')

        # sort data by time
        data = data.sort_values(by='date')
        
        ## generate cell coordinates count
        data['X'],data['Y'] = config.coord2grid(data['Latitude'].values, data['Longitude'].values)
        
        # remove data outside latitude and longitude boundaries
        data = data[(data['X'] != -1) & (data['Y'] != -1)]
        
        # save dataset
        if save:
            datasetSaveDir = self.projectDir + '/Data/PreprocessedDatasets'
            if not os.path.exists(datasetSaveDir):
                os.makedirs(datasetSaveDir)
        
            saved_path = datasetSaveDir + '/' + str(config.CRIME_TYPE_NUM) + '_crimes.csv'
            data.to_csv(saved_path)

        return data
        
    def readDataset(self):
        """
        Function that load dataset. If there is no dataset, run preprocessDataset() and load.

        Input: none.

        Output: data <DataFrame>: loaded dataset
        """
        
        print("Loading crime dataset...")
        
        # determine if there is preprocessed dataset
        datasetSaveDir = self.projectDir + '/Data/PreprocessedDatasets/' + str(config.CRIME_TYPE_NUM) + '_crimes.csv'
        if not (os.path.isfile(datasetSaveDir)):
            self.preprocessDataset()
            
        data = pd.read_csv(datasetSaveDir)
        # set datetime
        data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d',errors='coerce')
        data.dropna(inplace=True)
        # sort by time
        data.sort_values(by='date', inplace=True)

        return data
    
    def getPivotData(self):
        """
        Function to create crime timetable. Each row is crime counts of each grid with different crime type.

        Input: none.

        Output: dataPivot <DataFrame>: cirme timetable.
        """
        
        print("Creating crime timetable...")
        
        data = self.data
        # only select type date and location grid count
        data = data[['TYPE','date','type','X','Y']]
        
        # create a pivot table
        dataPivot = data.pivot_table(values='TYPE', index=['date','type'], columns=['X','Y'], aggfunc='count')
            
        # flatten the columns
        dataPivot.columns = dataPivot.columns.to_flat_index()
        
        # get all grids combinations
        xAll = np.arange(1, config.LAT_GRIDS+1, 1)
        yAll = np.arange(1, config.LON_GRIDS+1, 1)
        xyAll = [(x,y) for x in xAll for y in yAll]

        # get all date*crime combinations
        uniqueDates = data['date'].unique()
        uniqueType = data['type'].unique()
        indexAll = [(x,y) for x in uniqueDates for y in uniqueType]

        # reindex with all grid combinations and date-crime combinations
        dataPivot = dataPivot.reindex(indexAll).reindex(columns=xyAll).fillna(0)
        
        # set datatype to int
        dataPivot = dataPivot.astype('int8')

        return dataPivot
    
    def getFeatureLabel(self):
        """
        Function to generate feature and label pair used for training the ConvLSTM model

        Inputs: none.

        Output: features <array>: feature data of 12 days each, 
                labels <array>: label data of 1 day each,
                dataPivot <DataFrame>
        """

        print("Generating features and labels...")
        
        dataPivot = self.getPivotData()
        crimeArr = dataPivot.values
        
        # reshape the array to have 50*50 grids (xxx, type, 50, 50)
        data = crimeArr.reshape((-1,len(config.CRIME_TYPE),config.LAT_GRIDS,config.LON_GRIDS))
        seq_len = config.SEQ_LEN
        
        features = []
        labels = []
        # generate feature-lable pairs
        for i in np.arange(0,data.shape[0]-(seq_len+1)): # minus first day slot
            feature = data[i:i+seq_len]
            features.append(feature)
            
            label = data[i+seq_len+1]
            labels.append(label)

        features = np.array(features)
        labels = np.array(labels)
        # make numbers to int (0 or 1) as most number is one or zero.
        features = (features>0).astype(int)
        labels = (labels>0).astype(int)
        
        minus_days = config.SEQ_LEN + 1
        # set start date of dataset
        if (dataPivot.query(f"date < {config.START_DATE}").shape[0] == 0):
            startIndex = 0
        else:
            startIndex = int(dataPivot.query(f"date < {config.START_DATE}").shape[0] / config.CRIME_TYPE_NUM - minus_days)
        features = features[startIndex: ]
        labels = labels[startIndex: ]

        with h5py.File(self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_features.h5', 'w') as hf:
            hf.create_dataset("features",  data=features)
        with h5py.File(self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_labels.h5', 'w') as hf:
            hf.create_dataset("labels",  data=labels)
        
        dataPivot.to_csv(self.dataPivotDir)
        
        return features, labels, dataPivot
    
    def getTrainValTest(self):
        """
        Function that split features and labels into train validation and test dataset and save them.

        Input: none

        Output: none
        """
        
        print("Spliting Train Val Test dataset...")
        features, labels, dataPivot = self.features, self.labels, self.dataPivot
        
        # get split index by split date
        minus_days = config.SEQ_LEN + 1
        
        if (dataPivot.query(f"date < {config.START_DATE}").shape[0] == 0):
            startIndex = 0
        else:
            startIndex = int(dataPivot.query(f"date < {config.START_DATE}").shape[0] / config.CRIME_TYPE_NUM - minus_days)
        trainValIndex  = int(dataPivot.query(f"date < {config.TRAIN_VAL_DATE}").shape[0] / config.CRIME_TYPE_NUM - minus_days) - startIndex
        valTestIndex   = int(dataPivot.query(f"date < {config.VAL_TEST_DATE}").shape[0] / config.CRIME_TYPE_NUM - minus_days) - startIndex
        
        # Divide features and labels into train and test. 11 years of data is used for training (2010-2020) 
        # 1 year of data for validation (2021) and 1 year of data for testing (2022)
        featuresTrain = features[:trainValIndex]
        featuresVal = features[trainValIndex:valTestIndex]
        featuresTest = features[valTestIndex:]

        labelsTrain = labels[:trainValIndex]
        labelsVal = labels[trainValIndex:valTestIndex]
        labelsTest = labels[valTestIndex:]
        
        print("features train shape: ", featuresTrain.shape)
        print("features val shape: ", featuresVal.shape)
        print("features test shape: ", featuresTest.shape)
        print(" ")
        print("labels train shape: ", labelsTrain.shape)
        print("labels val shape: ", labelsVal.shape)
        print("labels test shape: ", labelsTest.shape)
        print(" ")
        print("all features shape", features.shape)
        print("all labels shape", labels.shape)
 
        # Save the features and labels as pickle files
        with h5py.File(self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_trainvaltest_features.h5', 'w') as hf:
            hf.create_dataset("train",  data=featuresTrain)
            hf.create_dataset("val", data=featuresVal)
            hf.create_dataset("test", data=featuresTest)
            
        with h5py.File(self.projectDir + f'/Data/PreprocessedDatasets/{config.CRIME_TYPE_NUM}_trainvaltest_labels.h5', 'w') as hf:
            hf.create_dataset("train",  data=labelsTrain)
            hf.create_dataset("val",  data=labelsVal)
            hf.create_dataset("test",  data=labelsTest)
        
    
    
    
