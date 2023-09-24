# Luohao Xu edsml-lx122

import os
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.dirname("__file__"))

################################################################### Dataset ###################################################################

# 16 types
# CRIME_TYPE = ['ROBBERY', 'BURGLARY', 'DANGEROUS DRUGS', 'PETIT LARCENY', 'FELONY ASSAULT', 
#                'VEHICLE AND TRAFFIC LAWS', 'DANGEROUS WEAPONS', 'CRIMINAL TRESPASS', 'FRAUDS',
#                'GRAND LARCENY OF MOTOR VEHICLE', 'GRAND LARCENY', 'RAPE',
#                'INTOXICATED & IMPAIRED DRIVING', 'FORGERY', 'GAMBLING', 'HARASSMENT']

# 8 types
CRIME_TYPE = ['ROBBERY', 'BURGLARY', 'PETIT LARCENY', 'DANGEROUS WEAPONS', 'FELONY ASSAULT',
               'VEHICLE AND TRAFFIC LAWS', 'CRIMINAL TRESPASS', 'GRAND LARCENY']

# 2 types
# CRIME_TYPE = ['ROBBERY', 'BURGLARY']

CRIME_TYPE_NUM = len(CRIME_TYPE)

# only includes right big island of NYC
# Latitude range:  1 deg = 110.574 km  (40.92 - 40.54)*110.574 = 42km
# Longitude range: 1 deg = 111.694 km  (74.05 - 74.70)*111.694 = 72km
LAT_MIN = 40.54
LAT_MAX = 40.92
LON_MIN = -74.05
LON_MAX = -73.70

# ~1km length width for each grid
LON_GRIDS = 50
LAT_GRIDS = 50

# length of feature data for ConvLSTM
SEQ_LEN = 12

START_DATE = "'2010-01-01'"  # chosen
START_SELECT_DATE = "'2010-02-01'" # limited by SARIMA model
END_DATE = "'2022-12-31'"   # limited by dataset range
TRAIN_VAL_DATE = "'2021-01-01'"
VAL_TEST_DATE = "'2022-01-01'"

################################################################### Functions ###################################################################

# grid divider
LAT_BINS = np.linspace(start=LAT_MIN, stop=LAT_MAX, num=LAT_GRIDS+1)
LON_BINS = np.linspace(start=LON_MIN, stop=LON_MAX, num=LON_GRIDS+1)

# ength from the center of the grid to the sides
DIFF_LAT = (LAT_BINS[-1] - LAT_BINS[0])/(len(LAT_BINS)-1)/2
DIFF_LON = (LON_BINS[-1] - LON_BINS[0])/(len(LON_BINS)-1)/2

# convert latitude longitude to grid number
def coord2grid(lats,longs):
    cell_x = np.digitize(lats,LAT_BINS,right=True)
    cell_y = np.digitize(longs,LON_BINS,right=True)
    try:
        cell_x_clean = [-1 if i==0 or i==len(LAT_BINS) else i for i in cell_x]
        cell_y_clean = [-1 if i==0 or i==len(LON_BINS) else i for i in cell_y]
        return cell_x_clean, cell_y_clean
    except:
        return cell_x, cell_y
    
# convert grid number to latitude and longitude
def grid2coord(x,y):
    lat = LAT_BINS[x]
    lon = LON_BINS[y]

    return lat, lon

################################################################### LSTM model ###################################################################

# dropout probability
DROP_P = 0.5

# kernel size
KERNEL_SIZE = 4

# hidden dimension in LSTM
HIDDEN_DIM = 64

# kernel size
KERNEL_SIZE = 3

################################################################### Training ###################################################################

DEVICE = 'cpu'

# binary cross entropy loss weights
BCE_WEIGHTS = [1,30]

RANDOM_SEED = 42

TRAIN_BATCH_SIZE = 16

# Learning rate
LEARNING_RATE = 3e-5

# number of epochs
N_EPOCHS = 5

SAVE = True

MODEL_SAVE_PATH = PROJECT_DIR + '/Data/ModelWeights'

CLASS_THRESH = 0.5

MULTIPLY_FACTOR = 0.3
