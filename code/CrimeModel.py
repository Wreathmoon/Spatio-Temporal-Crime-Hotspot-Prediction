# Luohao Xu edsml-lx122

import mesa
import os
from CrimeAgent import CrimeAgent
from datetime import datetime, timedelta

from TimeseriesModel import TimeseriesModel
from WeatherModel import WeatherModel
from DataPreprocessing import DataPreprocessing
import config

def getCrimeNumber(model):
    """
    Function to calculate total crime number of that day

    Input: model <object>: Crime model

    Output: crimeNumber <int>: total crime number
    """
    crimeNumber = 0
    # iterate every agent to check their final decision (1 for crime)
    for i in range(model.schedule.get_agent_count()):
        if model.schedule.agents[i].finalDecision:
            crimeNumber += 1
    
    return crimeNumber

class CrimeModel(mesa.Model):
    def __init__(self, initialAgents, width, height, initialDate, crimeType):
        """
        Function to initialise crime model

        Input: initialAgents <int>: number of initial agents
               width <int>: width of the model grids
               height <int>: height of the model grids
               initialDate <String>: starting date of the simulation
               crimeType <String>: crime type of the simulation (lowercase)
        """
        
        # set grid height and width
        self.grid = mesa.space.MultiGrid(width, height, False)
        # create scheduler and assign it to the model
        self.schedule = mesa.time.RandomActivation(self)     
        # initialise data collector with given algorithm and output name   
        self.datacollector = mesa.DataCollector(
            model_reporters={"crime_number": getCrimeNumber}, agent_reporters={"agent_crime_prob": "crimeProb"}
            )
        
        self.device = 'cpu'
        self.population = initialAgents 
        # initial unique id
        self.id = 0                                             
        self.running = True
        self.crimeType = crimeType
        self.date = initialDate
        # date counter
        self.dateCounter = 0
        # today's crime probobility
        self.probByDate = 0

        # initialize agents with differen location
        self.addAgentsByLocation(self.population)
        
        # initialize machine learning models
        projectDir = config.PROJECT_DIR
        dp = DataPreprocessing(projectDir)
        self.timeseriesModel = TimeseriesModel(projectDir, dp.data)
        self.weatherModel = WeatherModel(projectDir)
        
    def getCrimeCountByDate(self, date): 
        """
        Function to get today's overall crime count predicted by weather model and timeseries model

        Input: date <String>: today's date

        Output: crimeCountByDate <int>: crime count predicted by weather model and timeseries model
        """

        # get timeseries factor
        timeseriesFactor = self.timeseriesModel.getTimeseriesFactor(self.crimeType, date[:-2]+'01')

        # get crime count predicted by weather data
        weatherCrimeCount = self.weatherModel.predict(date)
        if (self.device == 'cuda'):
            weatherCrimeCount = weatherCrimeCount.values[0]
        
        # combine results of two models
        crimeCountByDate = int(weatherCrimeCount * timeseriesFactor)
        return crimeCountByDate
    
    def getRandomGender(self):
        """
        Function to randomly generate gender of Male or Female according to NYC gender distribution

        Output: gender <char>: randomly generated gender
        """
        # 53% of female in NYC
        gender = 'M' if self.random.randint(1,100) > 53 else 'F'
        return gender
    
    def getRandomAge(self):
        """
        Function to randomly generate age according to NYC age distribution

        Output: age <int>: randomly generated age from 0 to 80
        """
        # same age distribution as citizens in NYC in year 2022 
        ageDice = self.random.randint(1,100)
        if (ageDice<23):
            return self.random.randint(1,17)
        elif (ageDice>=23 and ageDice<30):
            return self.random.randint(18,24)
        elif (ageDice>=30 and ageDice<57):
            return self.random.randint(25,44)
        elif (ageDice>=57 and ageDice<83):
            return self.random.randint(45,64)
        elif (ageDice>=83):
            return self.random.randint(65,80)
        
    def getRandomRace(self):
        """
        Function to randomly generate race according to NYC race distribution

        Output: race <String>: randomly generated race from [native, asian, black, white, hispanic]
        """
        # same race distribution as citizens in NYC in year 2022 
        raceDice = self.random.random()
        if (raceDice<0.0058):
            return 'native'
        elif (raceDice>=0.0058 and raceDice<0.1483):
            return 'asian'
        elif (raceDice>=0.1483 and raceDice<0.3821):
            return 'black'
        elif (raceDice>=0.3821 and raceDice<0.7799):
            return 'white'
        elif (raceDice>=0.7799 and raceDice<1):
            return 'hispanic'
    
    def addAgentsByLocation(self, num):
        """
        Function to add agents with different initial location or default location

        Input: num <int>: number of agents to be added
        """
        for i in range(1, num+1):
            self.id += i
            # 10% possibility of initial previous crime history
            crimeHistory = 1 if self.random.randint(1,10) >= 1 else 0 

            # get random gender, age and race with total distribution same as the ones in NYC
            gender = self.getRandomGender()
            age = self.getRandomAge()
            race = self.getRandomRace()

            placeDice = self.random.randint(1,2)
            # placeDice = 0

            # initialise each agent with thier own crime history, gender, age, race
            ag = CrimeAgent(self.id, self, crimeHistory, gender, age, race, placeDice)
            # Add the agent to the scheduler
            self.schedule.add(ag)  
            
            if (placeDice == 0):
                x = self.random.randrange(0, int(self.grid.width))
                y = self.random.randrange(0, int(self.grid.width))
                self.grid.place_agent(ag, (x, y))
            elif (placeDice == 1):
                # Add the agent to a different initial grid cell
                x = self.random.randrange(0, int(self.grid.width/2))
                y = self.random.randrange(0, int(self.grid.width/2))
                self.grid.place_agent(ag, (x, y))
            elif (placeDice == 2):
                # Add the agent to a different initial grid cell
                x = self.random.randrange(int(self.grid.width/2), self.grid.width)
                y = self.random.randrange(int(self.grid.width/2), self.grid.height)
                self.grid.place_agent(ag, (x, y))

    def addYoungAgents(self, num):
        """
        Function to add young agents to the model grid with random loaction

        Input: num <int>: number of agents to be added
        """

        for i in range(1, num+1):
            self.id += i
            # 40% (higher) possibility of initial previous crime history
            crimeHistory = 1 if self.random.randint(1,10) > 4 else 0 
            
            # get random gender and race with total distribution same as the ones in NYC
            # age is from 18 to 24
            gender = self.getRandomGender()
            age = self.random.randint(18,24)
            race = self.getRandomRace()
            
            ag = CrimeAgent(self.id, self, crimeHistory, gender, age, race, place=0)
            # Add the agent to the scheduler
            self.schedule.add(ag)  
            
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(ag, (x, y))
    
    def step(self):
        """
        Function to step forward one step of the model
        """
        # date + 1
        tomorrow = datetime.strptime(self.date, "%Y-%m-%d") + timedelta(days=1)
        self.date = tomorrow.strftime('%Y-%m-%d')
        self.dateCounter += 1

        # get predicted crime count and crime probobility by date
        self.crimePredCounts = self.getCrimeCountByDate(self.date)
        self.probByDate = self.crimePredCounts / (self.population * 10)
        
        # collect crime data and write
        self.datacollector.collect(self)

        # make every agent step forward
        self.schedule.step()
        
        # # simulation of young people come to NYC
        # if (self.dateCounter == 30):
        #     self.addYoungAgents(100)   

        print(f'crime events is {getCrimeNumber(self)}')
        print(f"crimePredCounts: {self.crimePredCounts}")

        df = self.datacollector.get_model_vars_dataframe()

        outname = 'simulation_migration.csv'
        outdir = os.path.join(config.PROJECT_DIR, "/Outputs")
        # if not os.path.exists(outdir):
        #     os.mkdir(outdir)

        fullname = os.path.join(outdir, outname)    

        df.to_csv(fullname)