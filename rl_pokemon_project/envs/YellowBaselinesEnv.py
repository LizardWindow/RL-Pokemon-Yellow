#Billy Matthews
#wmatthe1@stumail.northeaststate.edu

import gymnasium as gymnasium
import sys
import time
import datetime
import numpy as np
from gymnasium import Env, spaces
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from gymnasium.utils import seeding
from skimage.transform import resize
class YellowEnv(Env):
    def __init__(self,config,rank):
        
        super(YellowEnv,self).__init__()
        
        
        
        
        #Creates an array of valid actions
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            #WindowEvent.PRESS_BUTTON_START,
            #WindowEvent.PASS
        ]
        #Sets gymnasium's action space to the list of valid actions
        #self.action_space = spaces.Discrete(len(self.valid_actions))
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        #Releases the held arrow key
        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
            
        ]
        #releases the held buttons
        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]
        #sets the window size/resolution/colors of the pyboy window
        #self.output_shape = (144,160, 3)
        self.output_shape = (36, 40, 3)
        self.output_full_shape = (144,160,3)
        self.observation_space = spaces.Box(low=0,high=255,shape=self.output_shape,dtype=np.uint8)
        
        #the ammount of time between actions taken
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        
        #variables needed for reward
        rank +=1
        #self.rewardMultiplier = rank * 0.1
        self.rewardMultiplier = 0.1
        self.step_count = 0
        self.exploredMaps = [(0,(0,0))]
        self.revisitedMaps = [(0,(0,0))]
        self.lastCoordinates = (0,(0,0))
        self.caughtPokemon = []
        self.gotParcel = False
        self.battleInProgress = False
        self.turnsInBattle = 0
        self.enemyLowestHP = 0
        self.mapProgress = 0
        self.newBattlePokemon = False
        self.newEnemy = False
        self.levelTotal = 0
        
        #Variables to track specifics on how well model is performing
        self.rank = rank
        self.dateTime = time.localtime()
        self.processNumber = rank
        self.totalRewardAllResets = 0
        self.totalRewardThisReset = 0
        self.gotPikachuCounter = 0
        self.attacksPerformed = 0
        self.totalAttacksPerformed = 0
        self.knockedOut = 0
        self.resets = -2
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
        
        



        
        #simple directmedia layer 2, used by pyboy to access graphics/controls/etc
        
        head='SDL2'
        if self.rank > 0:
            head='headless'
        
        #instantiates a pyboy emulator object to run yellow
        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )
        
        self.init_state = config['init_state']
        with open(self.init_state, 'rb') as f:
            self.pyboy.load_state(f)
        
        #sets screen variable to pyboy's screen manager
        self.screen = self.pyboy.botsupport_manager().screen()
        
        self.agent_stats = []
        self.total_reward=np.array(int)
    
    
    
    
    
    
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
    def TrackerReset(self, isFirst = False):
        if self.resets == 1:
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
        progressLog.write("\nExploration Reward Multiplier = "+str(self.rewardMultiplier))
        progressLog.write("\nTotal Reward all resets = " +str(self.totalRewardAllResets)) 
        progressLog.write("\nTotal got Pikachu = "+str(self.gotPikachuCounter))
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
        MEPercentage = 0
        if self.fRewardTrackerMapExploration > 0:
            MEPercentage = (self.rewardTrackerMapExploration *100) / self.fRewardTrackerMapExploration
        progressLog.write("\nMap exploration reward total: " + str(self.fRewardTrackerMapExploration) + "/"+ str(self.rewardTrackerMapExploration)+ "/" +str(MEPercentage))
        WPPercentage = 0
        if self.fRewardTrackerWorldProgression > 0:
            WPPercentage = (self.rewardTrackerWorldProgression *100) / self.fRewardTrackerWorldProgression
        progressLog.write("\nWorld progression reward total: " + str(self.fRewardTrackerWorldProgression) + "/"+ str(self.rewardTrackerWorldProgression) + "/"+str(WPPercentage))
        PCPercentage = 0
        if self.fRewardTrackerPokemonCaught > 0:
            PCPercentage = (self.rewardTrackerPokemonCaught *100) / self.fRewardTrackerPokemonCaught
        progressLog.write("\nPokemon caught reward total: " + str(self.fRewardTrackerPokemonCaught) + "/"+ str(self.rewardTrackerPokemonCaught) + "/"+str(PCPercentage))
        PLPercentage = 0
        if self.fRewardTrackerPokemonLevels > 0:
            PLPercentage = (self.rewardTrackerPokemonLevels *100) / self.fRewardTrackerPokemonLevels
        progressLog.write("\nPokemon levels reward total: " + str(self.fRewardTrackerPokemonLevels) + "/"+ str(self.rewardTrackerPokemonLevels) + "/"+str(PLPercentage))
        DDPercentage = 0
        if self.fRewardTrackerDamageDealt > 0:
            DDPercentage = (self.rewardTrackerDamageDealt *100) / self.fRewardTrackerDamageDealt
        progressLog.write("\nDamage dealt reward total: " + str(self.fRewardTrackerDamageDealt)  + "/"+ str(self.rewardTrackerDamageDealt)  + "/"+str(DDPercentage))
        DRPercentage = 0
        if self.fRewardTrackerDamageReceived > 0:
            DRPercentage = (self.rewardTrackerDamageReceived *100) / self.fRewardTrackerDamageReceived
        progressLog.write("\nDamage received reward total: " + str(self.fRewardTrackerDamageReceived) +"/"+ str(self.rewardTrackerDamageReceived) +"/"+str(DRPercentage))
        FRPercentage = 0
        if self.fRewardTrackerFlagsReached > 0:
            FRPercentage = (self.rewardTrackerFlagsReached *100) / self.fRewardTrackerFlagsReached
        
        progressLog.write("\nFlags reached reward total: " + str(self.fRewardTrackerFlagsReached) + "/"+ str(self.rewardTrackerFlagsReached) + "/"+str(FRPercentage))
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
    
    
    
    
    
    def render(self):
        #used to get pixel data back from the screen variable
        game_pixels_render = self.screen.screen_ndarray()
        
        #This ensures that the render size, and the observation shape of the model match when the size is being compressed
        game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
        return game_pixels_render

# had to have seed = none and options = none to get it to work
#, seed=None, options=None
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
            self.pyboy.set_emulation_speed(0)
        self.step_count = 0
        self.exploredMaps = [(0,(0,0))]
        self.revisitedMaps = [(0,(0,0))]
        self.lastCoordinates = (0,(0,0))
        self.caughtPokemon = []
        self.gotParcel = False
        self.battleInProgress = False
        self.turnsInBattle = 1
        self.enemyLowestHP = 0
        self.newBattlePokemon = False
        self.newEnemy = False
        self.total_reward = np.array(int)
        self.resets += 1
        if self.resets > 0:
            self.TrackerReset()
        
        
        

        
        
        return self.render(), {}

    def step(self,action):
        
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.act_freq):

            if i == 8:
                if action < 4:
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    self.pyboy.send_input(self.release_button[action -4])
                if self.valid_actions[action] == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)
                    
            if i == self.act_freq-1:
                self.pyboy._rendering(True)
            self.pyboy.tick()

        
        self.turnsInBattle = self.pyboy.get_memory_value(0XCCD4)
        obs_memory = self.render()
        new_reward = self.reward()
        
        
        terminated = False
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            
        self.step_count += 1
        #return obs_memory, new_reward, terminated, truncated, {}
        return obs_memory, new_reward, terminated, truncated, {}
    
    def reward(self):
        #can create a list of x/y pos for each map, and if the current isn't in the list, give a reward
        reward = 0
        reward += self.rewardPosition()
        #need to prioritize catching a pidgey/rattata/something that can surf
        reward += self.rewardPokemon()
        #reward based off correct map progression
        reward += self.rewardProgress()
        #must prioritize badges
        #reward += self.rewardTrainers()
        #must deprioritize getting knocked out
        reward += self.rewardKO()
        #prioritize doing higher damage?
        reward += self.rewardDamage()
        #deprioritize talking to pikachu?
        #reward += self.rewardNoPikachu()
        #prioritize gaining levels
        reward += self.rewardPartyLevels()
        #prioritize teaching hms?
        #reward += self.rewardHMs()
        #deprioritize using moves when out of pp
        #reward += self.rewardPP()
        #prioritize flags
        #reward += self.rewardFlags()
        if reward > 0:
            idc = False
        if reward < 0:
            idc = False
        #return reward *0.1
        self.totalRewardThisReset += reward
        return reward
    def rewardPosition(self):
        #use this to generate reward based off the x/y coordinates of the current map
        #this will convince it to explore new areas instead of stay in the current area
        #instead of checking through this single list of touples, turn it into a multidimensional list to save on 
        #processing time as the model gets further into the game
        
        currentMapAddress = 0XD35D
        X_POS_ADDRESS, Y_POS_ADDRESS = 0XD361,0XD360
        x_pos = self.pyboy.get_memory_value(X_POS_ADDRESS)
        y_pos = self.pyboy.get_memory_value(Y_POS_ADDRESS)
        mapCoordinates = x_pos, y_pos
        currentMap = self.pyboy.get_memory_value(currentMapAddress)
        currentLocation = (currentMap,(mapCoordinates))
        
        if currentLocation != self.lastCoordinates:
            for i in range(len(self.exploredMaps)):
                if self.exploredMaps[i] == currentLocation:    
                    for j in range(len(self.revisitedMaps)):
                        if self.revisitedMaps[j] == currentLocation:
                            self.lastCoordinates = currentLocation
                            reward = -1 * self.rewardMultiplier
                            self.TrackerAdd(reward,"ME")
                            return reward
                    self.lastCoordinates = currentLocation
                    return 0
            self.exploredMaps.append(currentLocation)
            self.lastCoordinates = currentLocation
            
            reward = 1 * self.rewardMultiplier
            self.TrackerAdd(reward,"ME")
            return reward
        return 0
    
    def rewardProgress(self):
        reward = 0
        #mapTarget = [37,0,12,1,42,12,0,40,0,12,1,13,51,13,2,54]
        mapTarget = [37,0,12,1,13,51,13,2,54]
        currentMapAddress = 0XD35D
        currentMap = self.pyboy.get_memory_value(currentMapAddress)
        if currentMap == mapTarget[self.mapProgress]:
            reward = 5
            self.mapProgress += 1
            reward = reward * self.mapProgress
            self.TrackerAdd(reward,"WP")
            return reward
        return reward
        #completionPercent = self.step_count * 100 / self.max_steps 
        #rewardModifier = 100 - completionPercent
        #return (reward * rewardModifier) /100
        
        
    def rewardPokemon(self):
        #use this to generate reward based off of specific pokemon being caught and in the party
        #this is helpful to give it a chance at future progression requiring hms
        #currently is set to reward based only off ammount of pokemon in the party
        
        #sets variables to be used
        reward = 0
        teamAddresses = []
        
        #memory addresses needed
        pokemon1Address =0XD163
        pokemon2Address =0XD164
        pokemon3Address =0XD165
        pokemon4Address =0XD166
        pokemon5Address =0XD167
        pokemon6Address =0XD168
        
        #obtains values from all memory addresses
        pokemon1 = self.pyboy.get_memory_value(pokemon1Address)
        pokemon2 = self.pyboy.get_memory_value(pokemon2Address)
        pokemon3 = self.pyboy.get_memory_value(pokemon3Address)
        pokemon4 = self.pyboy.get_memory_value(pokemon4Address)
        pokemon5 = self.pyboy.get_memory_value(pokemon5Address)
        pokemon6 = self.pyboy.get_memory_value(pokemon6Address)
        
        #checks if memory address is empty, if not, add it to the list to be checked
        if pokemon1 != 0 & pokemon1!= 255 & self.contains(teamAddresses,pokemon1):
            teamAddresses.append(pokemon1)
        if pokemon2 != 0 & pokemon2!= 0 & self.contains(teamAddresses,pokemon2):
            teamAddresses.append(pokemon2)
        if pokemon3 != 0 & pokemon3!= 0 & self.contains(teamAddresses,pokemon3):
            teamAddresses.append(pokemon3)
        if pokemon4 != 0 & pokemon4!= 0 & self.contains(teamAddresses,pokemon4):
            teamAddresses.append(pokemon4)
        if pokemon5 != 0 & pokemon5!= 0 & self.contains(teamAddresses,pokemon5):
            teamAddresses.append(pokemon5)
        if pokemon6 != 0 & pokemon6!= 0 & self.contains(teamAddresses,pokemon6):
            teamAddresses.append(pokemon6)
        
        #if ammount of pokemon has increased, give reward and overwrite caughtPokemon list, otherwise, lower reward and overwrite
        if len(teamAddresses) > len(self.caughtPokemon):
            self.caughtPokemon = teamAddresses
            self.gotPikachuCounter +=1
            #he can abuse this by storing pokemon and taking them out of the pc. Honestly wondering if it can figure that out.
            reward = 2 + self.gotPikachuCounter
        elif len(teamAddresses) < len(self.caughtPokemon):
            reward -= 3
            self.caughtPokemon = teamAddresses
        self.TrackerAdd(reward,"PC")
        return reward
    
    def contains(self, list, variable):
        for i in list:
            if i == variable:
                return False
        return True
    def rewardTrainers(self):
        #use this to generate reward based off specific battles that need to be fought to progress the game
        #this can help guide the model through the game by getting badges from gyms
        
        #for badges, all you have to do is check if the value has changed since last time, if so, reward it
        #there is no way to lose badges, so every change is when you gain a badge
        #D356 = Badges (Binary Switches)
        return
    def rewardKO(self):
        #use this to generate negative reward if a pokemon is knocked out, and further negative if ai wipes out
        #this is one piece of teaching it how to battle

        reward = 0
        
        pokemonMaxHPBattleAddress1 = 0XD022
        pokemonMaxHPBattleAddress2 = 0XD023
        pokemonMaxHPAddress1 = 0XD18C
        pokemonMaxHPAddress2 = 0XD18D
        
        pokemonHPAddress1 = 0XD014
        pokemonHPAddress2 = 0XD015
        pokemonStatusAddress = 0XD017
        
        
        pokemonHP1 = self.pyboy.get_memory_value(pokemonHPAddress1)
        pokemonHP2 = self.pyboy.get_memory_value(pokemonHPAddress2)
        pokemonMaxHPBattle1 = self.pyboy.get_memory_value(pokemonMaxHPBattleAddress1)
        pokemonMaxHPBattle2 = self.pyboy.get_memory_value(pokemonMaxHPBattleAddress2)
        pokemonMaxHP1 = self.pyboy.get_memory_value(pokemonMaxHPAddress1)
        pokemonMaxHP2 = self.pyboy.get_memory_value(pokemonMaxHPAddress2)
        pokemonStatus = self.pyboy.get_memory_value(pokemonStatusAddress)
        
        pokemonHP = pokemonHP1 + pokemonHP2
        pokemonMaxHPBattle = pokemonMaxHPBattle1 + pokemonMaxHPBattle2
        pokemonMaxHP = pokemonMaxHP1 + pokemonMaxHP2
        
        #this is currently being seen as 0 even after battle
        if self.turnsInBattle == 0:
            self.battleInProgress = True
        
        if pokemonMaxHP > 0 and pokemonHP > 0:
            self.newBattlePokemon = True
        
        if self.battleInProgress == True:
            if pokemonMaxHP > 0 or pokemonMaxHPBattle > 0: 
                if pokemonHP == 0 and self.newBattlePokemon == True:
                    reward -= 0.5
                    self.knockedOut += 1
                    self.newBattlePokemon = False
                    self.battleInProgress = False
        #if self.battleInProgress == True:
            #if pokemonStatus > 0:
                #reward -= 10
        self.TrackerAdd(reward,"DR")
        return reward
    def rewardDamage(self):
        #use this to generate reward based off how much damage is done to the opposing pokemon?
        #can also be used to reward knockouts on the opposing side
        
        
        reward = 0
        
        enemyMaxHPAddress1 = 0XCFF3
        enemyMaxHPAddress2 = 0XCFF4
        
        
        enemyHPAddress1 = 0XCFE5
        enemyHPAddress2 = 0XCFE6
        enemyStatusAddress = 0XCFE8
        
        
        enemyMaxHP1 = self.pyboy.get_memory_value(enemyMaxHPAddress1)
        enemyMaxHP2 = self.pyboy.get_memory_value(enemyMaxHPAddress2)
        
        enemyHP1 = self.pyboy.get_memory_value(enemyHPAddress1)
        enemyHP2 = self.pyboy.get_memory_value(enemyHPAddress2)
        enemyStatus = self.pyboy.get_memory_value(enemyStatusAddress)
        
        enemyHP = enemyHP1 + enemyHP2
        enemyMaxHP = enemyMaxHP1 + enemyMaxHP2
        
        if self.turnsInBattle == 0 & enemyStatus == 0 & self.battleInProgress == False:
            self.battleInProgress = True
            #reward +=1 why was this here????
        
        if enemyMaxHP > 0 and enemyHP > 0:
            self.newEnemy = True
        if enemyHP > self.enemyLowestHP:
            self.enemyLowestHP = enemyHP
        
        if self.battleInProgress == True:
            #had to check for maxhp to avoid constant reward loop due to default enemy current hp set at 0
            if enemyMaxHP > 0:
                if enemyHP == 0 and self.newEnemy == True:
                    reward += 0.5
                    self.attacksPerformed +=1
                    self.newEnemy = False
                    self.battleInProgress = False
                elif enemyHP > 0 and enemyHP < self.enemyLowestHP and self.newEnemy == True:
                    reward += 0.25
                    self.attacksPerformed +=1
                    self.enemyLowestHP = enemyHP
        #if self.battleInProgress == True:
         #   if enemyStatus > 0:
                    
          #     reward += 2
        self.TrackerAdd(reward,"DD")
        return reward
        
    def rewardNoPikachu(self):
        #use this to keep the ai from constantly talking to pikachu
        #only add in if it becomes a problem
        return
    def rewardPartyLevels(self):
        #use this to generate reward based off highest ever party combined levels
        #make sure not to set by current or might run into problem of boxing pokemon
        reward = 0
        
        slot1Address =0xD18B
        slot2Address =0xD1B7
        slot3Address =0xD1E3
        slot4Address =0xD20F
        slot5Address =0xD23B
        slot6Address =0xD267
        
        slot1 = self.pyboy.get_memory_value(slot1Address)
        slot2 = self.pyboy.get_memory_value(slot2Address)
        slot3 = self.pyboy.get_memory_value(slot3Address)
        slot4 = self.pyboy.get_memory_value(slot4Address)
        slot5 = self.pyboy.get_memory_value(slot5Address)
        slot6 = self.pyboy.get_memory_value(slot6Address)
        
        levels = slot1+slot2+slot3+slot4+slot5+slot6
        
        if levels > self.levelTotal:
            reward += 1
            self.levelTotal = levels
        self.TrackerAdd(reward,"PL")
        return reward
        completionPercent = self.step_count * 100 / self.max_steps 
        rewardModifier = 100 - completionPercent
        return (reward * rewardModifier) /100
    def rewardHMs(self):
        #use this to generate reward for teaching required hms to pokemon
        #possibly try to ensure that it's only taught once
        return
    def rewardPP(self):
        #use this to generate negative reward for trying to use moves that run out of pp
        #can be helpful to let it know that pp exists and not to keep slamming tackle 50 million times in a row
        #D02D 	PP (First Slot) 
        #D02E 	PP (Second Slot) 
        #D02F 	PP (Third Slot) 
        #D030 	PP (Fourth Slot) 
        return
    def rewardFlags(self):
        #use this to generate reward based off certain ingame flags being triggered
        #this will hopefully be useful for learning to beeline to specific points in the game
        
        #D5F3 - Have Town map?

        
        reward = 0
        oaksParcelAddress = 0XD60C
        
        if self.gotParcel == False:
            oaksParcel = self.pyboy.get_memory_value(oaksParcelAddress)
            if oaksParcel > 0:
                
                self.gotParcel = True
                reward += 50
                self.exploredMaps = [(0,(0,0))]
                self.revisitedMaps = [(0,(0,0))]
        self.TrackerAdd(reward,"FR")
        return reward
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def close(self):
        self.pyboy.stop()
        super().close()
