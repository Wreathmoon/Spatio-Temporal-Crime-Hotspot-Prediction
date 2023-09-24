import pytest
from config import *

def test_crime_types():
    assert len(CRIME_TYPE) == 8
    assert 'ROBBERY' in CRIME_TYPE
    assert 'BURGLARY' in CRIME_TYPE
    
def test_lat_lon_ranges():
    assert LAT_MIN == 40.54
    assert LAT_MAX == 40.92
    assert LON_MIN == -74.05 
    assert LON_MAX == -73.70
    
def test_grid_params():
    assert LAT_GRIDS == 50
    assert LON_GRIDS == 50
    
def test_date_ranges():
    assert START_DATE == "'2010-01-01'"
    assert END_DATE == "'2022-12-31'"
    
def test_model_params():
    assert HIDDEN_DIM == 64
    assert KERNEL_SIZE == 3
    assert DROP_P == 0.5
    
@pytest.mark.parametrize("lat,lon,expected", [
    (40.541, -74.049, ([1], [1])),
    (40.919, -73.701, ([50], [50])) 
])
def test_coord2grid(lat, lon, expected):
    result = coord2grid([lat],[lon])
    assert result == expected