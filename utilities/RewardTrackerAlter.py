#Billy Matthews
#wmatthe1@stumail.northeaststate.edu


import time

class RewardTracker:
    """Class for tracking various reward values to help debug long running issues with the reward weights and how the model uses them to learn

    Returns:
        _type_: _description_
    """
    #This class saves the model and training logs for the stablebaselines model
    
    def __init__(self, rank, modelSettings):
        self.rank = rank
        self.dateTime = time.localtime()
        self.settings = modelSettings
        
        self.battlesFought=0
        self.battlesLost=0
        self.battlesWon=0
        self.battlesDrawn=0
        self.attacksMade=0
        self.tilesFound =0
        self.totalReward=0
        self.explorationReward=0
        self.battleReward=0
        self.modifier =0
        
        
        self.bestReset = 0
        
        self.currentRun = {
            "Battles Fought": 0, 
            "Battles Lost" : 0,
            "Battles Won" : 0,
            "Battles Drawn" : 0,
            "Attacks Made" : 0,
            "Tiles Found" : 0,
            "Total Reward" : 0,
            "Exploration Reward" : 0,
            "Battle Reward" : 0,
            "Reward Modifier" : 0
        }
        
        self.averageRun = {
            "Battles Fought": 0, 
            "Battles Lost" : 0,
            "Battles Won" : 0,
            "Battles Drawn" : 0,
            "Attacks Made" : 0,
            "Tiles Found" : 0,
            "Total Reward" : 0,
            "Exploration Reward" : 0,
            "Battle Reward" : 0,
            "Reward Modifier" : 0
        }
        self.firstRun = {
            "Battles Fought": 0, 
            "Battles Lost" : 0,
            "Battles Won" : 0,
            "Battles Drawn" : 0,
            "Attacks Made" : 0,
            "Tiles Found" : 0,
            "Total Reward" : 0,
            "Exploration Reward" : 0,
            "Battle Reward" : 0,
            "Reward Modifier" : 0
        }
        self.bestRun = {
            "Battles Fought": 0, 
            "Battles Lost" : 0,
            "Battles Won" : 0,
            "Battles Drawn" : 0,
            "Attacks Made" : 0,
            "Tiles Found" : 0,
            "Total Reward" : 0,
            "Exploration Reward" : 0,
            "Battle Reward" : 0,
            "Reward Modifier" : 0
        }
        self.mapStepCount = 0
        self.mapTotals = {
            0 : 0,
            1: 0,
            2: 0,
            3: 0,
            12: 0,
            13: 0,
            14: 0,
            37: 0,
            41: 0,
            49: 0,
            50: 0,
            51: 0,
            54: 0,
            58: 0,
            59: 0,
            60: 0,
            61: 0}
        self.mapNames = {
            0 : "Pallet Town",
            1 : "Viridian City",
            2 : "Pewter City",
            3 : "Cerulean City",
            12 : "Route 1",
            13 : "Route 2",
            14 : "Route 3",
            37 : "Red's House: F1",
            41 : "Pokemon Center: Viridian",
            49 : "Gate: Route 2",
            50 : "Gate: Route 2, North",
            51 : "Viridian Forest",
            54 : "Pewter Gym",
            58 : "Pokemon Center: Pewter",
            59 : "Mt. Moon, Entrance",
            60 : "Mt. Moon, Somewhere",
            61 : "Mt. Moon, Somewhere else",}
        
    def WriteMapTotals(self, log):
        log.write("\nMap Time Percentages")
        for k, v in self.mapTotals.items():
                log.write("\n"+str(self.mapNames[k]) + ": "+ str((self.mapTotals[k] *100) / self.mapStepCount) +"%")
        log.write("\n---------------------------------------------------------")
    
    def TrackerReset(self, resets):
        """Resets single reset values after writing tracked reward to a text file

        Args:
            resets (_type_): _description_
        """
        self.GetValues(resets)
        progressLog = open("progressLogProcess" + str(self.rank) + ".txt", "w")
        progressLog.write("\n---------------------------------------------------------")
        progressLog.write("\n---------------------------------------------------------")
        progressLog.write("\nProcess Number: " + str(self.rank))
        progressLog.write("\nDate/time = " + str(self.dateTime))
        self.WriteMapTotals(progressLog)
        progressLog.write("\nCurrent Reset: " + str(resets))
        progressLog.write("\nBest Run Reset: " + str(self.bestReset))
        progressLog.write("\n---------------------------------------------------------")
        progressLog.write("\nModel Settings:\n")
        self.RewardWriter(progressLog,self.settings)
        progressLog.write("\n\nCurrent Run Stats:\n")
        self.RewardWriter(progressLog,self.currentRun)
        progressLog.write("\nFirst Run Stats:\n")
        self.RewardWriter(progressLog,self.firstRun)
        progressLog.write("\nBest Run Stats:\n")
        self.RewardWriter(progressLog,self.bestRun)
        progressLog.write("\nAverage Run Stats:\n")
        self.RewardWriter(progressLog,self.averageRun)
        progressLog.write("\n---------------------------------------------------------")
        #not sure why this isn't being added to git
        for k, v in self.mapTotals.items():
                self.mapTotals[k] = 0
                
        
    
    
    def RewardWriter(self,log,dictionary):
        for k, v in dictionary.items():
            log.write("\n"+str(k) + ": "+ str(v))
        log.write("\n---------------------------------------------------------")
            
    def GetValues(self, resets):
        self.currentRun = {
            "Battles Fought": self.battlesFought, 
            "Battles Lost" : self.battlesLost,
            "Battles Won" : self.battlesWon,
            "Battles Drawn" : self.battlesDrawn,
            "Attacks Made" : self.attacksMade,
            "Tiles Found" : self.tilesFound,
            "Total Reward" : self.totalReward,
            "Exploration Reward" : self.explorationReward,
            "Battle Reward" : self.battleReward,
            "Reward Modifier" : self.modifier
            }
        
        self.bestRunReward = False
        
        if self.currentRun["Total Reward"] > self.bestRun["Total Reward"]:
            self.bestReset = resets
            for k, v in self.firstRun.items():
                self.bestRun[k] = self.currentRun[k]
                
        if resets == 1:
            self.fvisitedMaps = self.visitedMaps
            for k, v in self.firstRun.items():
                self.firstRun[k] = self.currentRun[k]
        
        for k, v in self.averageRun.items():
                self.averageRun[k] = self.Averager(self.averageRun[k], self.currentRun[k], resets)
        
    
    def Averager(self, averageReward, currentReward, reset):
        """Averages collected reward values

        Args:
            averageReward (_type_): _description_
            currentReward (_type_): _description_
            reset (_type_): _description_

        Returns:
            _type_: _description_
        """
        if reset == 1:
            return currentReward
        return ((averageReward * (reset - 1)) +currentReward) / reset
        