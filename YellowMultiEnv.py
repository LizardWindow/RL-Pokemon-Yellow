#Billy Matthews
#wmatthe1@stumail.northeaststate.edu


#https://medium.com/@ym1942/create-a-gymnasium-custom-environment-part-2-1026b96dba69
#Used for Yellow env starter code

#https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map
#Used to find Memory Addresses for Pokemon Yellow

#https://gymnasium.farama.org/tutorials/gymnasium_basics/vector_envs_tutorial/#
#used code for A2C model

#https://github.com/PWhiddy/PokemonRedExperiments

#Renotte, Nicholas. “Build an Mario AI Model with Python | Gaming Reinforcement Learning.” YouTube, 
# YouTube, 23 Dec. 2021, www.youtube.com/watch?v=2eeYqJ0uBKE.
#used as starter code for the code in YellowSingleEnv

#https://glitchcity.wiki/wiki/List_of_maps_by_index_number_(Generation_I)
#used for index numbers of maps for rewardMapProgress

#import vectorization wrappers
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
# import os for file path management
import os
#import ppo for algos
from stable_baselines3 import PPO
#import base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
#import register to register pokemon yellow as a custom gym environment
from gymnasium.envs.registration import register, make
from rl_pokemon_project.envs.YellowBaselinesEnv import YellowEnv

#This method was taken from the red rl repo linked above. This was one of the keys to my multiprocessing problem, mixed with discovering the api compatibility
#problem linked below

#This method is used to create an environment with a different seed based off a passed in index to be used for each seed
def make_env(env_config, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        
        env = YellowEnv(env_config, rank)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init

#Needed to avoid some issues with multiprocessing once I get back to trying to run multiple environments
if __name__ == '__main__':
    

    #Registers env with gym so that A2C can see it. (You wouldn't believe how long I spent just trying to find the exact file structure and naming to make this work)
    #Turns out, while this was very useful for creating a single gym environment, it was counterproductive to creating multiple environments.
    #Since I'm only ever going to be looking at one environment, it is better to declare the environment in a custom constructor that can be looped through
    #On top of this, the api compatibility while helping me solve an earlier problem with constructing a single environment, completely broke compatibility with
    #creating multiple environments by expecting different information to be returned from the step method. 
    register(
        id="Yellow-v0",
        entry_point="rl_pokemon_project.envs.YellowBaselinesEnv:YellowEnv",
    )
    fileChoice = 4
    if fileChoice == 0:
        stateFile = "CatchingTutorial.gb.state"
    elif fileChoice == 1:
        stateFile = "FirstBattle.state"
    elif fileChoice == 2:
        stateFile = "LevelingUp.state"
    elif fileChoice == 3:
        stateFile = "GotOaksParcel.state"
    elif fileChoice == 4:
        stateFile = "StartWithPokeballs.state"
    else:
        stateFile = "PokemonYellowVersion.gb.state"
    #setup directories
    ROM_PATH = "PokemonYellowVersion.gb" #File location of Pokemon Yellow
    INIT_STATE_FILE_PATH = stateFile #File location of the starting state of the rom
    PROGRESS_LOG = './logs/progressLogs/'
    ep_length = 2048 *10
    #sets up configurations for the environment
    env_config = {
        'action_freq': 24, 'init_state': INIT_STATE_FILE_PATH,
        'gb_path': ROM_PATH, 'max_steps': ep_length,
        'progressLogs': PROGRESS_LOG
    }
    
    num_cpu = 32
    #Use DummyVecEnc whenever you need to troubleshoot, similar requirements but subproc is a lot more vague on exceptions
    env = SubprocVecEnv([make_env(env_config,i) for i in range(num_cpu)])

    #saves ai learning model and logs
    class TrainAndLoggingCallback(BaseCallback):
        def __init__(self, check_freq, save_path, verbose=1):
            super(TrainAndLoggingCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.save_path = save_path
            
        def _init_callback(self):
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
        
        def _on_step(self):
            if self.n_calls % self.check_freq == 0:
                #model_path = os.path.join(self.save_path, 'best_model')
                model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
                self.model.save(model_path)
                
            return True
    #setup directories for saving training data
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    

    ep_length = 2048 *10
    #setup model saving callback
    callback = TrainAndLoggingCallback(check_freq=ep_length, save_path=CHECKPOINT_DIR)
    
    #create reinforcement learning model
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR,n_steps=ep_length, batch_size=128, n_epochs=3, gamma=0.998, learning_rate=0.00001,device="cuda",)

    #model= PPO.load('./train/best_model_ViridianTourist.zip', env=env, device="cuda")
    model.learn(total_timesteps=(ep_length) *num_cpu*5000,callback = callback)
    #model.load('./train/best_model_55000.zip')


    
        

