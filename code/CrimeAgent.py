# Luohao Xu edsml-lx122

import mesa
import random
import copy

class CrimeAgent(mesa.Agent):
    def __init__(self, unique_id, model, crimeHistory, gender, age, race, place):
        super().__init__(unique_id, model)
        """
        Function to initialise agent with its own crime history, gender, age, race, initial place
        """

        self.crimeHistory = crimeHistory
        self.gender = gender                                  
        self.age = age                                        
        self.race = race
        # initial crime probobility is 50% if someone has crime history
        self.crimeProb = 0 if self.crimeHistory==0 else 0.5 
        self.place = place 
        # initialise the final decision to false
        self.finalDecision = False 
    
    def move(self):
        """
        Function to make movement to the agent
        """
        # get the neighbor cells
        possibleGrids = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,  #  Moore (includes all 8 surrounding squares), and Von Neumann(only up/down/left/right).
            include_center=True) # include center which means no move
        
        x, y = self.pos
        step=10
        if (self.place == 0):
            pass
        # left bottom agent move to the other side
        elif (self.place == 1):
            if (x+step < self.model.grid.width and y+step<self.model.grid.height):
                possibleGrids.append((x+step,y+step))#.append((x+step,y+step))
        # right top agent move to the other side
        elif (self.place ==2):
            if (x-step > 0 and y-step >0):
                possibleGrids.append((x-step,y-step))#.append((x+step,y+step))
        # choose one new place from the possibleGrids
        new_position = self.random.choice(possibleGrids)
        # make movement by placing new location
        self.model.grid.move_agent(self, new_position)

    def probByGender(self, prob):
        """
        Function to get crime probability affected by gender

        Input: prob <float>: total probability predicted by two machine learning model on given date

        Output: probByGender <float>: crime probability of this agent according to its gender
        """
        # from 2010 to 2022, there were 167202 male criminals and 20257 female criminals with proportion of 47% and 53% respectively  
        # (167202/47) / (20257/53)
        m2fRatio = 9.30774  
        x = prob / (0.53 + 0.47*m2fRatio)
        probFemale = x
        probMale = x*m2fRatio

        if (self.gender == 'M'):
            return probMale
        else:
            return probFemale
        
    def probByRace(self, prob):
        """
        Function to get crime probability affected by race

        Input: prob <float>: total probability predicted by two machine learning model on given date

        Output: probByRace <float>: crime probability of this agent according to its race
        """
        # from 2010 to 2022, there were 256, 5419, 106821, 14228, 60735 criminals with race of native, asian, black, white, hispanic
        # with proportion of 0.58%, 14.25%, 23.38%, 39.78%, 22.01% respectively  
        a2nRatio = 0.8615734649122807  # (5419/14.25)/(256/0.58)
        b2nRatio = 10.351425497219847  # (106821/23.38)/(256/0.58)
        w2nRatio = 0.8103396807440925  # (14228/39.78)/(256/0.58)
        h2nRatio = 6.251828004316219  # (60735/22.01)/(256/0.58)
        # get native probability
        x = prob/(0.0058 + 0.1425 * a2nRatio + 0.2338 * b2nRatio + 0.3978 * w2nRatio + 0.2201 * h2nRatio)
        probNative = x
        probAsian = x * a2nRatio
        probBlack = x * b2nRatio
        probWhite = x * w2nRatio
        probHispanic = x * h2nRatio

        if(self.race == 'native'):
            return probNative
        elif(self.race == 'asian'):
            return probAsian
        elif(self.race == 'black'):
            return probBlack
        elif(self.race == 'white'):
            return probWhite
        elif(self.race == 'hispanic'):
            return probHispanic
    
    def probByAge(self, prob):
        """
        Function to get crime probability affected by age

        Input: prob <float>: total probability predicted by two machine learning model on given date

        Output: probByRace <float>: crime probability of this agent according to its age
        """
        # from 2010 to 2022, there were 40169, 51537, 69774, 25301, 678 criminals 
        # with age group of <18, 19-24, 25-44, 45-64, >64
        # with proportion of 23.2%, 6.3%, 27%, 26%, 17.5% respectively  
        age_10_ratio = 4.7247142726741425  # (51537/6.3)/(40169/23.2)
        age_20_ratio = 1.4925428951568511  # (69774/27)/(40169/23.2)
        age_40_ratio = 0.5620323364553991  # (25301/26)/(40169/23.2)
        age_60_ratio = 0.0223763172026758  # (678/17.5)/(40169/23.2)
        # get crime probability of age group of <18
        x = prob/(0.232 + 0.063 * age_10_ratio + 0.27 * age_20_ratio + 0.26 * age_40_ratio + 0.175 * age_60_ratio)
        prob_0 = x
        prob_1 = x * age_10_ratio
        prob_2 = x * age_20_ratio
        prob_4 = x * age_40_ratio
        prob_6 = x * age_60_ratio

        if(self.age <= 18):
            return prob_0
        elif(self.age>18 and self.age<=24):
            return prob_1
        elif(self.age>24 and self.age<=44):
            return prob_2
        elif(self.age>44 and self.age<=64):
            return prob_4
        elif(self.age>64):
            return prob_6

    def decision_tree(self, probByDate):
        """
        Function to calculate agent's final crime decision of this day by its age, race, gender and today's total crime probability

        Input: probByDate <float>: totay's total crime probability
        """
        # get agents near me
        near_agents = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False) # except the one on the current cell

        # according to near repeat principle, simulate that if there are agents with crime history near this agent, its crime probability increase.
        prob_near = copy.deepcopy(probByDate)
        if len(near_agents) > 0:
            # search how many agents with crime history near this agent
            near_criminals = [near_agents[i] for i in range(len(near_agents)) if near_agents[i].crimeHistory == 1]
            num_near_criminals = len(near_criminals)
            # more agents with crime history near this agent, higher its crime probability will be.
            if (num_near_criminals > 0):
                prob_near = (num_near_criminals+1) * probByDate

        # get its crime probabilty according to agent's gender, race and age
        probByGender = self.probByGender(prob_near)
        probByRace = self.probByRace(probByGender)
        probByAge = self.probByAge(probByRace)
        
        # get final crime decision
        dice = random.random()
        self.finalDecision = True if dice < probByAge else False
        self.crimeHistory = 1 if self.finalDecision else 0
            
    def step(self):
        """
        Function to step forward of this day by making movement and crime decision
        """
        # agent make movement
        self.move()

        # # aging society simulation
        # self.age += 1  

        # # simulation of gender change
        # if self.model.dateCounter == 30:
        #     self.gender = 'F'

        # # simulation of race change
        # if self.model.dateCounter == 30:
        #     race_list = ['native', 'asian', 'white', 'black', 'hispanic']
        #     self.race = random.choice(race_list)

        # get crime decision of this day
        self.decision_tree(self.model.probByDate)