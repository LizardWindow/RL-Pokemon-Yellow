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

#import gym for handling the environment
import gym
#import grayscaling wrapper
from gym.wrappers import  GrayScaleObservation
#import vectorization wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
# import os for file path management
import os
#import ppo for algos
from stable_baselines3 import PPO
#import base callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
#import register to register pokemon yellow as a custom gym environment
from gym.envs.registration import register
import time

#Needed to avoid some issues with multiprocessing once I get back to trying to run multiple environments
if __name__ == '__main__':
    
    #Used for vectorized environments
    def make_env(env_id, rank, seed=0):
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """

        def _init():
            env = gym.make(env_id)
            # use a seed for reproducibility
            # Important: use a different seed for each environment
            # otherwise they would generate the same experiences
            env.reset(seed=seed + rank)
            return env

        set_random_seed(seed)
        return _init

    #Registers env with gym so that A2C can see it. (You wouldn't believe how long I spent just trying to find the exact file structure and naming to make this work)
    register(
        id="Yellow-v0",
        entry_point="rl_pokemon_project.envs.YellowBaselinesEnv:YellowEnv"
    )

    #setup directories
    ROM_PATH = "PokemonYellowVersion.gb" #File location of Pokemon Yellow
    INIT_STATE_FILE_PATH = "PokemonYellowVersion.gb.state" #File location of the starting state of the rom

    #sets up configurations for the environment
    env_config = {
        'action_freq': 24, 'init_state': INIT_STATE_FILE_PATH,
        'gb_path': ROM_PATH
    }
    
    #instantiates the Pokemon Yellow environment
    env = gym.make('Yellow-v0',config=env_config)
    #grayscales the environment to lower amount of variables that the agent needs to track
    env = GrayScaleObservation(env, keep_dim = True)
    #wrap inside dummy environment, can be used to run multiple environments at once...somehow
    env = DummyVecEnv([lambda: env])


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
                model_path = os.path.join(self.save_path, 'best_model')
                #model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
                self.model.save(model_path)
                
            return True
    #setup directories for saving training data
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    #setup model saving callback
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    #create reinforcement learning model
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001,n_steps=512)

    #model.load('./train2/best_model_100_highstep.zip')
    #model.load('./train2/best_model_100_lowstep.zip')
    model.learn(total_timesteps=10000000,callback = callback)
    #model.load('./train2/best_model_100_highstep.zip')
    #train the ai model
    #model.learn(total_timesteps=1000,callback = callback)


    
        

