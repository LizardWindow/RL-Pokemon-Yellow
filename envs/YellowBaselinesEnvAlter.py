#Billy Matthews
#wmatthe1@stumail.northeaststate.edu

import sys #Used by Pyboy
import gymnasium as gymnasium #Contains many supporting methods/classes/wrappers for setting up environments for RL models
import numpy as np #Used for many methods in gymnasium
from gymnasium import Env, spaces #Used by gymnasium
from pyboy import PyBoy #Python based Gameboy Emulator
from pyboy.utils import WindowEvent #Used for Pyboy functionality
from gymnasium.utils import seeding #Used to generate a random seed
from skimage.transform import resize #Used to resize pixel data in Render()
from utilities.RewardTracker import RewardTracker #Tracks reward data for debugging
class YellowEnv(Env):
    """Model for a Pokemon Yellow Gymnasium environment

    Args:
        Env (_type_): _description_
    """
    def __init__(self,config,rank):
        """Parameterized constructor

        Args:
            config (_type_): _Configuration file for setting up the environment_
            rank (_type_): _Identification number for env when using multiple environments_
        """
        super(YellowEnv,self).__init__()
        
        #Creates an array of valid actions
        #had to temporarily remove the start button and pass to speed up learning
        #start will be required further into the game to teach hms and use items, but it is not required for the proof of concept leg
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
        #by setting output shape to a smaller number, we can compress the image and help the model learn a bit faster
        #by setting the third number to one, we can grayscale the image and speed things up even further
        self.output_shape = (36, 40, 1)
        self.output_full_shape = (144,160,3)
        self.observation_space = spaces.Box(low=0,high=255,shape=self.output_shape,dtype=np.uint8)
        
        #the ammount of time between actions taken
        self.act_freq = config['action_freq']
        self.max_steps = config['max_steps']
        
        #variables needed for reward
        self.rank = rank +1
        self.explorationMultiplier = config['expl_mult']
        self.battleMultiplier = config['batl_mult']
        self.step_count = 0
        self.exploredMaps = [(0,(0,0))]
        self.revisitedMaps = [(0,(0,0))]
        self.lastCoordinates = (0,(0,0))
        self.caughtPokemon = []
        self.gotParcel = False
        self.battleReset = False
        self.battleWon = False
        self.battlelost = False
        self.enemyLowestHP = 0
        self.mapProgress = 0
        self.newBattlePokemon = False
        self.newEnemy = False
        self.levelTotal = 0
        self.resets = -2
        self.PP0StepCount = 1
        self.viridianPokeCenter = False
        self.pewterPokeCenter = False
        self.viridianForest = False
        self.pewterGym = False
        self.highestSeenLevel = 1
        self.flagsReached = 0
        
        self.levels = 0
        self.discoveredMaps = []
        self.selfKOCount = 0
        self.enemyKOCount = 0
        self.ranAway = 0
        self.teamAddresses = []
        self.enemyLevel = 1
        
        
        
        #used to track reward totals to better discover what the model is prioritizing
        self.rewardTracker = RewardTracker(rank)
        
        
    
        
        #simple directmedia layer 2, used by pyboy to access graphics/controls/etc
        #can switch to headless to not display game to screen
        head='SDL2'
        if self.rank > 1:
            head='headless'
        
        #instantiates a pyboy emulator object to run yellow
        self.pyboy = PyBoy(
            config['gb_path'],
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window='--quiet' in sys.argv,
        )
        
        #loads save state chosen in config file
        self.init_state = config['init_state']
        with open(self.init_state, 'rb') as f:
            self.pyboy.load_state(f)
        
        #sets screen variable to pyboy's screen manager
        self.screen = self.pyboy.botsupport_manager().screen()
    
    #used to retrieve pixel data to render the environment
    def render(self):
        #used to get pixel data back from the screen variable
        game_pixels_render = self.screen.screen_ndarray()
        
        #This ensures that the render size, and the observation shape of the model match when the size is being compressed
        game_pixels_render = (255*resize(game_pixels_render, self.output_shape)).astype(np.uint8)
        return game_pixels_render

# had to have seed = none and options = none to get it to work
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        self.rewardTracker.TrackerAdd((len(self.exploredMaps) - len(self.revisitedMaps)),"ME")
        self.rewardTracker.TrackerAdd(self.ranAway, "RA")
        self.rewardTracker.TrackerAdd(len(self.discoveredMaps)* len(self.discoveredMaps),"WP")
        self.rewardTracker.TrackerAdd(-self.selfKOCount,"DR")
        self.rewardTracker.TrackerAdd(self.enemyKOCount,"DD")
        self.rewardTracker.TrackerAdd(self.levels,"PL")
        self.rewardTracker.TrackerAdd(self.flagsReached,"FR")
        self.rewardTracker.rewardTrackerPP = self.rewardTracker.rewardTrackerPP / self.PP0StepCount
                
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
            self.pyboy.set_emulation_speed(0)
        self.step_count = 0
        self.exploredMaps = [(0,(0,0))]
        self.revisitedMaps = [(0,(0,0))]
        self.lastCoordinates = (0,(0,0))
        self.caughtPokemon = []
        self.gotParcel = False
        self.battleReset = False
        self.enemyLowestHP = 0
        self.newBattlePokemon = False
        self.newEnemy = False
        self.resets += 1
        if self.resets > 0:
            self.rewardTracker.mapProgress = self.mapProgress
            self.rewardTracker.TrackerReset(self.resets)
        self.levelTotal = 0
        self.mapProgress = 0
        self.discoveredMaps = []
        self.selfKOCount = 0
        self.enemyKOCount = 0
        self.ranAway = 0
        self.levels = 0
        self.flagsReached = 0
        self.PP0StepCount = 1
        self.teamAddresses = []
        self.highestSeenLevel = 1
        self.enemyLevel = 1
        
        return self.render(), {}
    
    def step(self,action):
        """Moves the environment forward using an action from the RL model, and returns outcome to the model

        Args:
            action (_type_): Agent's chosen action

        Returns:
            _type_: Observation data, Reward, Whether the environment is being reset, Whether environment is being terminated
        """
        
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

        obs_memory = self.render()
        self.GetAddressValues()
        new_reward = self.reward()
        terminated = False
        truncated = False
        
        #resets env if it has reached the max step count
        if self.step_count >= self.max_steps:
            truncated = True
            
        self.step_count += 1
        #return obs_memory, new_reward, terminated, truncated, {}
        return obs_memory, new_reward, terminated, truncated, {}
    
    def GetAddressValues(self):
        #memory addresses
        self.pokemon1Address =self.pyboy.get_memory_value(0XD163) #checks what pokemon is in slot 1-6
        self.pokemon2Address =self.pyboy.get_memory_value(0XD164) #
        self.pokemon3Address =self.pyboy.get_memory_value(0XD165) #
        self.pokemon4Address =self.pyboy.get_memory_value(0XD166) #
        self.pokemon5Address =self.pyboy.get_memory_value(0XD167) #
        self.pokemon6Address =self.pyboy.get_memory_value(0XD168) #
        self.audioBankAddress = self.pyboy.get_memory_value(0xC0EF) #contains currently loaded audio bank
        self.currentMapAddress = self.pyboy.get_memory_value(0XD35D) #contains currently loaded map
        self.xAddress = self.pyboy.get_memory_value(0XD361)
        self.yAddress = self.pyboy.get_memory_value(0XD360) #contains player x/y coordinates
        self.gymMusicPlayingAddress = self.pyboy.get_memory_value(0xD05B) #contains value of whether gym leader battle music is playing or not
        self.pokemonMaxHPBattle1Address = self.pyboy.get_memory_value(0XD022) # contains max hp of users current pokemon in battle
        self.pokemonMaxHPBattle2Address = self.pyboy.get_memory_value(0XD023) #
        self.pokemonMaxHP1Address = self.pyboy.get_memory_value(0XD18C) #contains values of max and current hp of pokemon in slot 1 of the party
        self.pokemonMaxHP2Address = self.pyboy.get_memory_value(0XD18D) #
        self.pokemonHP1Address = self.pyboy.get_memory_value(0XD014) #
        self.pokemonHP2Address = self.pyboy.get_memory_value(0XD015) #
        self.pokemonStatusAddress = self.pyboy.get_memory_value(0XD017) #contains value of status condition inflicted on user's pokemon in slot 1
        self.enemyMaxHP1Address = self.pyboy.get_memory_value(0XCFF3) # contains max/current hp of enemy's active pokemon
        self.enemyMaxHP2Address = self.pyboy.get_memory_value(0XCFF4) #
        self.enemyHP1Address = self.pyboy.get_memory_value(0XCFE5) #
        self.enemyHP2Address = self.pyboy.get_memory_value(0XCFE6) #
        self.enemyStatusAddress = self.pyboy.get_memory_value(0XCFE8) #contains value of status condition inflicted on enemy's active pokemon
        self.pokemonSlot1LevelAddress =self.pyboy.get_memory_value(0xD18B) #contains value for current levels of the pokemon in the party
        self.pokemonSlot2LevelAddress =self.pyboy.get_memory_value(0xD1B7) #
        self.pokemonSlot3LevelAddress =self.pyboy.get_memory_value(0xD1E3) #
        self.pokemonSlot4LevelAddress =self.pyboy.get_memory_value(0xD20F) #
        self.pokemonSlot5LevelAddress =self.pyboy.get_memory_value(0xD23B) #
        self.pokemonSlot6LevelAddress =self.pyboy.get_memory_value(0xD267) #
        self.PPSlot1Address = self.pyboy.get_memory_value(0xD02C) 
        self.PPSlot2Address = self.pyboy.get_memory_value(0xD02D) 
        self.PPSlot3Address = self.pyboy.get_memory_value(0xD02E )
        self.PPSlot4Address = self.pyboy.get_memory_value(0xD02F) 
        self.move1Address = self.pyboy.get_memory_value(0xD01B)
        self.move2Address = self.pyboy.get_memory_value(0xD01C)
        self.move3Address = self.pyboy.get_memory_value(0xD01D)
        self.move4Address = self.pyboy.get_memory_value(0xD01E)
        self.oaksParcelAddress = self.pyboy.get_memory_value(0XD60C) #contains flag for if player has obtained oak's parcel or not
        self.enemyLevel = self.pyboy.get_memory_value(0xCFF2)
    
    def reward(self):
        reward = 0
        if self.IsBattleOver():
            if self.rewardPosition() and self.rewardProgress():
                reward = 1
            else: 
                reward = 0
        else:
            if self.BattleHandler():
                reward = 1
            else:
                reward = 0
        #TalkToNPCs
        #   Not sure how to do this yet
        self.rewardPartyLevels()
        #highest level pokemon seen
        
        if self.enemyLevel > self.highestSeenLevel:
            self.highestSeenLevel = self.enemyLevel
        reward = reward * (self.highestSeenLevel / 100)
        
        return reward
        
        
    
            
    
    def rewardPosition(self):
        """use this to generate reward based off the x/y coordinates of the current map
        \nthis will convince it to explore new areas instead of stay in the current area
        Returns:
            int: Reward
        """
        
        #TODO instead of checking through this single list of touples, turn it into a multidimensional list to save on processing time as the model gets further into the game
        mapCoordinates = self.xAddress, self.yAddress
        currentLocation = (self.currentMapAddress,(mapCoordinates))
        
        
        #Checks if agent has been on this tile before
        if self.contains(self.exploredMaps, currentLocation) != True:
            self.exploredMaps.append(currentLocation)
            return True
        return False
    
    
    def rewardProgress(self):
        """Rewards map progress in the game based off time taken to reach there 
        \nhelps point models exploration in a designated direction
        
        Returns:
            int: Reward
        """
        validMaps = [0,1,2,3,12,13,14,37,41,49,50,51,54,58,59,60,61]
        #Checks if agent is currently in the next map target
        if self.contains(validMaps, self.currentMapAddress):
            return True
        return False
        
    
        
    def rewardPokemon(self):
        """generates reward when catching a pokemon and additional reward if it is one of a few specific pokemon
        \nthis is helpful to give it a chance at future progression requiring hms
        
        Returns:
            int: Reward
        """
        
        
        
        #checks if memory address is empty, if not, add it to the list to be checked. 0 = empty, 255 = empty but next to be filled
        if self.pokemon1Address != 0 & self.pokemon1Address!= 255 & self.contains(self.teamAddresses,self.pokemon1Address) != True:
            self.teamAddresses.append(self.pokemon1Address)
        if self.pokemon2Address != 0 & self.pokemon2Address!= 255 & self.contains(self.teamAddresses,self.pokemon2Address)!= True:
            self.teamAddresses.append(self.pokemon2Address)
        if self.pokemon3Address != 0 & self.pokemon3Address!= 255 & self.contains(self.teamAddresses,self.pokemon3Address)!= True:
            self.teamAddresses.append(self.pokemon3Address)
        if self.pokemon4Address != 0 & self.pokemon4Address!= 255 & self.contains(self.teamAddresses,self.pokemon4Address)!= True:
            self.teamAddresses.append(self.pokemon4Address)
        if self.pokemon5Address != 0 & self.pokemon5Address!= 255 & self.contains(self.teamAddresses,self.pokemon5Address)!= True:
            self.teamAddresses.append(self.pokemon5Address)
        if self.pokemon6Address != 0 & self.pokemon6Address!= 255 & self.contains(self.teamAddresses,self.pokemon6Address)!= True:
            self.teamAddresses.append(self.pokemon6Address)
        
        
    
    
    
    def rewardTrainers(self):
        """use this to generate reward based off specific battles that need to be fought to progress the game
        \nthis can help guide the model through the game by getting badges from gyms
        \ncurrently just set to reward the model while gym battle music is playing
        Returns:
            int: Reward
        """
        #TODO add badge reward
        #for badges, all you have to do is check if the value has changed since last time, if so, reward it
        #there is no way to lose badges, so every change is when you gain a badge
        #D356 = Badges (Binary Switches)
        reward = 0
        #checks to see if agent is currently in a gym leader battle. This reward is far too high for long term growth
        #as it is given every step, but will be fine for my current purpose of getting the agent to the gym
        if self.gymMusicPlayingAddress > 0:
            reward +=1
        self.rewardTracker.TrackerAdd(reward, "T")
        return reward
    
    def BattleHandler(self):
        if self.newBattlePokemon == False or self.newEnemy == False:
            return False
        if self.rewardEnemyDamage() or self.rewardEnemyKO():
            return True
        if self.rewardSelfKO():
            return False
    
    def rewardEnemyKO(self):
        if self.enemyHP1Address + self.enemyHP2Address > 0:
            self.newEnemy == True
        if self.enemyHP1Address + self.enemyHP2Address == 0 and self.newEnemy == True:
            self.newEnemy = False
            self.battlelost = False
            self.battleWon = True
            return True
        return False
    def rewardSelfKO(self):
        #Checks if own Pokemon's current HP is greater than 0
        
        if self.pokemonHP1Address + self.pokemonHP2Address > 0:
            self.newBattlePokemon = True
        
        if self.pokemonHP1Address + self.pokemonHP2Address == 0 and self.newBattlePokemon:
            self.selfKOCount +=1
            self.rewardTracker.knockedOut += 1
            self.battlelost = True
            self.battleWon = False
            self.newBattlePokemon = False
            return True
        return False
    
    def rewardEnemyDamage(self):
        enemyHP = self.enemyHP1Address + self.enemyHP2Address
        if enemyHP > self.enemyLowestHP:
            self.enemyLowestHP = enemyHP
            return False
        if enemyHP < self.enemyLowestHP:
            self.enemyLowestHP = enemyHP
            return True
        return False
        
    
    def IsBattleOver(self):
        if self.audioBankAddress == 8:
            False   
        else:
            return True
    
    
    def rewardConversation(self):
        return
    def rewardHealing(self):
        return
    
        
    def rewardPartyLevels(self):
        """Generates reward based off highest ever party combined levels
        Returns:
            int: Reward
        """
        reward = 0

        self.levels = self.pokemonSlot1LevelAddress+self.pokemonSlot2LevelAddress+self.pokemonSlot3LevelAddress+self.pokemonSlot4LevelAddress
        +self.pokemonSlot5LevelAddress+self.pokemonSlot6LevelAddress
        reward = self.levels
        
        return reward
    
    def rewardPP(self):
        """use this to generate negative reward for trying to use moves that run out of pp
        
        Returns:
            int: Reward
        """
        reward = 0
        
        
        
        if self.move2Address == 0:
            self.PPSlot2Address =1
        if self.move3Address == 0:
            self.PPSlot3Address =1
        if self.move4Address == 0:
            self.PPSlot4Address =1
        
        
        
        if (self.PPSlot1Address == 0 or self.PPSlot2Address == 0 or self.PPSlot3Address == 0 or self.PPSlot4Address == 0):
             reward -=1
             self.PP0StepCount +=1
        self.rewardTracker.TrackerAdd(reward, "PP")
        
        return reward
    #TODO fix/rebuild flags reward
    def rewardFlags(self):
        """use this to generate reward based off certain ingame flags being triggered
        Returns:
            int: Reward
        """
        reward = 0
        totalFlags = 0

        if self.oaksParcelAddress > 0:
            totalFlags += 1
        if totalFlags > self.flagsReached:
            self.flagsReached = totalFlags
        reward = totalFlags
        return reward

    def _seed(self, seed=None):
        """Creates a random seed for the environment

        Args:
            seed (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: Seed number
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def close(self):
        """Closes pyboy emulator
        """
        self.pyboy.stop()
        super().close()
    def contains(self, list, variable):
        """Checks to see if a variable is contained in a list

        Args:
            list (_type_): _description_
            variable (_type_): _description_

        Returns:
            _type_: _description_
        """
        for i in list:
            if i == variable:
                return True
        return False
