import time

class RewardTracker:
    
    #This class saves the model and training logs for the stablebaselines model
    
    def __init__(self, rank):
        self.rank = rank
        self.dateTime = time.localtime()
        self.mapProgress = 0
        self.processNumber = rank
        self.totalRewardAllResets = 0
        self.totalRewardThisReset = 0
        self.caughtPokemon = 0
        self.attacksPerformed = 0
        self.totalAttacksPerformed = 0
        self.knockedOut = 0
        self.progressing = 0
        
        
        self.bestRunResets = 0
        self.bestRunTotalRewardThisReset = 0
        self.bestRunProgressing = 0
        self.bestAttacksPerformed =0
        self.bestKnockedOut = 0
        
        self.highestMapProgress = 0
        self.hmReset =0
        self.highestAttacksPerformed = 0
        self.haReset =0
        self.highestKnockedOut = 0
        self.hkReset = 0
        
        self.rewardTrackerMapExploration = 0
        self.rewardTrackerWorldProgression = 0
        self.rewardTrackerPokemonCaught =0
        self.rewardTrackerPokemonLevels =0
        self.rewardTrackerDamageDealt =0
        self.rewardTrackerDamageReceived =0
        self.rewardTrackerFlagsReached =0
        
        self.fRewardTrackerMapExploration = 0
        self.fRewardTrackerWorldProgression = 0
        self.fRewardTrackerPokemonCaught = 0
        self.fRewardTrackerPokemonLevels = 0
        self.fRewardTrackerDamageDealt = 0
        self.fRewardTrackerDamageReceived = 0
        self.fRewardTrackerFlagsReached = 0
        self.resets = 1
        
        
    def TrackerAdd(self, reward, tracker):
        if tracker == "ME":
            self.rewardTrackerMapExploration += reward
        elif tracker == "WP":
            self.rewardTrackerWorldProgression += reward
        elif tracker =="PC":
            self.rewardTrackerPokemonCaught += reward
        elif tracker =="PL":
            self.rewardTrackerPokemonLevels += reward
        elif tracker =="DD":
            self.rewardTrackerDamageDealt += reward
        elif tracker =="DR":
            self.rewardTrackerDamageReceived += reward
        elif tracker =="FR":
            self.rewardTrackerFlagsReached += reward
    def TrackerReset(self, resets):
        if resets == 1:
            self.fRewardTrackerMapExploration = self.rewardTrackerMapExploration
            self.fRewardTrackerWorldProgression = self.rewardTrackerWorldProgression
            self.fRewardTrackerPokemonCaught = self.rewardTrackerPokemonCaught
            self.fRewardTrackerPokemonLevels = self.rewardTrackerPokemonLevels
            self.fRewardTrackerDamageDealt = self.rewardTrackerDamageDealt
            self.fRewardTrackerDamageReceived = self.rewardTrackerDamageReceived
            self.fRewardTrackerFlagsReached = self.rewardTrackerFlagsReached
            
        self.totalRewardAllResets += self.totalRewardThisReset
        self.bestRun = False
        
        self.totalAttacksPerformed += self.attacksPerformed
            
        if self.totalRewardThisReset > self.bestRunTotalRewardThisReset:
            self.bestRun = True
            self.bestRunResets = self.resets
            self.bestRunTotalRewardThisReset = self.totalRewardThisReset
            self.bestRunProgressing = self.mapProgress
            self.bestAttacksPerformed = self.attacksPerformed
            self.bestKnockedOut = self.knockedOut
        if self.mapProgress > self.highestMapProgress:
            self.hmReset = self.resets
            self.highestMapProgress = self.mapProgress
        if self.attacksPerformed > self.highestAttacksPerformed:
            self.haReset = self.resets
            self.highestAttacksPerformed = self.attacksPerformed
        if self.knockedOut > self.highestKnockedOut:
            self.hkReset = self.resets
            self.highestKnockedOut = self.knockedOut
            
        
        
        progressLog = open("progressLogProcess" + str(self.rank) + ".txt", "w")
        progressLog.write("\n")
        progressLog.write("\n---------------------------------------------------------")
        progressLog.write("\n---------------------------------------------------------")
        progressLog.write("\nProcess number = " + str(self.processNumber))
        progressLog.write("\nTotal Reward all resets = " +str(self.totalRewardAllResets)) 
        progressLog.write("\nTotal caught Pokemon = "+str(self.caughtPokemon))
        progressLog.write("\nTotal attacks performed all resets = "+str(self.totalAttacksPerformed))
        progressLog.write("\n")
        progressLog.write("\nReset Number: " +str(self.resets))
        progressLog.write("\n")
        progressLog.write("\nBest Run? = " +str(self.bestRun)) 
        progressLog.write("\nTotal Reward this reset = " +str(self.totalRewardThisReset))
        progressLog.write("\nMap Progress = "+str(self.mapProgress))
        progressLog.write("\nAttacks performed = "+str(self.attacksPerformed))
        progressLog.write("\nTimes own Pokemon knocked out = "+str(self.knockedOut))
        progressLog.write("\n")
        progressLog.write("\nBest run stats:")
        progressLog.write("\n")
        progressLog.write("\nBest run reset count: " + str(self.bestRunResets))
        progressLog.write("\nBest run total reward: " + str(self.bestRunTotalRewardThisReset))
        progressLog.write("\nBest run map progress: " + str(self.bestRunProgressing))
        progressLog.write("\nBest run damaging attacks performed: " + str(self.bestAttacksPerformed))
        progressLog.write("\nBest run self ko total = "+str(self.bestKnockedOut))
        progressLog.write("\n")
        progressLog.write("\nHighest ever totals:")
        progressLog.write("\n")
        progressLog.write("\nHighest map Progress: " + str(self.highestMapProgress))
        progressLog.write("\nOccurred at reset: " + str(self.hmReset))
        progressLog.write("\nHighest amount of attacks: " + str(self.highestAttacksPerformed))
        progressLog.write("\nOccurred at reset: " + str(self.haReset))
        progressLog.write("\nHighest amount self ko: " + str(self.highestKnockedOut))
        progressLog.write("\nOccurred at reset: " + str(self.hkReset))
        progressLog.write("\n")
        progressLog.write("\nReward Details:")
        progressLog.write("\n")
        progressLog.write("\nRun Reward Totals First/Current/Change:")
        progressLog.write("\n")
        progressLog.write(self.TrackerRewardStringBuild("Map exploration", self.fRewardTrackerMapExploration, self.rewardTrackerMapExploration))
        progressLog.write(self.TrackerRewardStringBuild("World progression", self.fRewardTrackerWorldProgression, self.rewardTrackerWorldProgression))
        progressLog.write(self.TrackerRewardStringBuild("Pokemon caught", self.fRewardTrackerPokemonCaught, self.rewardTrackerPokemonCaught))
        progressLog.write(self.TrackerRewardStringBuild("Pokemon levels", self.fRewardTrackerPokemonLevels, self.rewardTrackerPokemonLevels))
        progressLog.write(self.TrackerRewardStringBuild("Damage dealt", self.fRewardTrackerDamageDealt, self.rewardTrackerDamageDealt))
        progressLog.write(self.TrackerRewardStringBuild("Damage received", self.fRewardTrackerDamageReceived, self.rewardTrackerDamageReceived))
        progressLog.write(self.TrackerRewardStringBuild("Flags reached", self.fRewardTrackerFlagsReached, self.rewardTrackerFlagsReached))
        progressLog.write("\n---------------------------------------------------------")
        progressLog.write("\n---------------------------------------------------------")
        progressLog.close()
        
        self.totalRewardThisReset = 0
        self.attacksPerformed = 0
        self.mapProgress = 0
        self.knockedOut = 0
        self.levelTotal = 0
        
        
        self.rewardTrackerMapExploration = 0
        self.rewardTrackerWorldProgression = 0
        self.rewardTrackerPokemonCaught =0
        self.rewardTrackerPokemonLevels =0
        self.rewardTrackerDamageDealt =0
        self.rewardTrackerDamageReceived =0
        self.rewardTrackerFlagsReached =0
    
    
    def TrackerRewardStringBuild(self,name, firstReward, currentReward):
        rewardPercentage = 0
        if firstReward > 0:
            rewardPercentage = (currentReward *100) / firstReward
        
        return("\n"+str(name) +" reward total: " + str(firstReward) + "/"+ str(currentReward) + "/"+str(rewardPercentage))