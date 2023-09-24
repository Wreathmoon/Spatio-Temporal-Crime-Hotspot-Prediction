# Luohao Xu edsml-lx122

import mesa
from CrimeModel import *
import warnings
warnings.filterwarnings("ignore")

def agent_portrayal(agent):
    """
    Function to plot agents on the grid with different color

    Input: agent<Object>: agent
    """
    portrayal = {"Shape": "circle", "Filled": "true", "r": "1"}

    # criminals are plotted with red circle dot
    if agent.finalDecision:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 1
    else:
        portrayal["Color"] = "black"
        portrayal["Layer"] = 0
    return portrayal

if __name__ == '__main__':
    # initialise grid
    grid = mesa.visualization.CanvasGrid(agent_portrayal, 100, 100, 500, 500)
    # initialise datacollector and crime rate graph
    chart = mesa.visualization.ChartModule(
        [{"Label": "crime_number", "Color": "Black"}], 
        data_collector_name="datacollector")

    # initialise server with chosen parameters
    server = mesa.visualization.ModularServer(
        CrimeModel, [grid, chart], "Crime Model", 
        {"initialAgents": 1000, "width": 100, "height": 100, "initialDate": '2022-01-01', "crimeType":'robbery'}
    )

    # run simulation
    server.launch(open_browser=True)