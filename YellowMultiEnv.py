#Billy Matthews
#wmatthe1@stumail.northeaststate.edu


#import vectorization wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
#import ppo for algos
from stable_baselines3 import PPO
#import register to register pokemon yellow as a custom gym environment
from gymnasium.envs.registration import register
from envs.YellowBaselinesEnvAlter import YellowEnv
import psutil
import os

from utilities.BaselinesCallback import TrainAndLoggingCallback

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
    
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    #Registers env with gym so that A2C can see it. (You wouldn't believe how long I spent just trying to find the exact file structure and naming to make this work)
    #Turns out, while this was very useful for creating a single gym environment, it was counterproductive to creating multiple environments.
    #Since I'm only ever going to be looking at one environment, it is better to declare the environment in a custom constructor that can be looped through
    #On top of this, the api compatibility while helping me solve an earlier problem with constructing a single environment, completely broke compatibility with
    #creating multiple environments by expecting different information to be returned from the step method. 
    register(
        id="Yellow-v0",
        entry_point="rl_pokemon_project.envs.YellowBaselinesEnv:YellowEnv",
    )
    
    #Variable to choose starting state of the game
    fileChoice = 8
    if fileChoice == 0:
        stateFile = "./states/CatchingTutorial.gb.state"
    elif fileChoice == 1:
        stateFile = "./states/FirstBattle.state"
    elif fileChoice == 2:
        stateFile = "./states/LevelingUp.state"
    elif fileChoice == 3:
        stateFile = "./states/GotOaksParcel.state"
    elif fileChoice == 4:
        stateFile = "./states/StartWithPokeballs.state"
    elif fileChoice == 5:
        stateFile = "./states/TeamCaught.state"
    elif fileChoice == 6:
        stateFile = "./states/TeamCaughtLeveled.state"
    elif fileChoice == 7:
        stateFile = "./states/noBaldMen.state"
    elif fileChoice == 8:
        stateFile = "./states/NoPikachu.state"
    else:
        stateFile = "./states/PokemonYellowVersion.gb.state"
    #setup directories
    ROM_PATH = "PokemonYellowVersion.gb" #File location of Pokemon Yellow
    INIT_STATE_FILE_PATH = stateFile #File location of the starting state of the rom
    PROGRESS_LOG = './logs/progressLogs/'
    
    ep_length = 2048*10 #episode length of training before env is truncated, needs to be a multiple of batch size, making a multiplier better for changing length
    
    #sets up configurations for the environment
    model_config = {
        "Policy" : "CnnPolicy",
        "Verbose" : 1, #Level of debug messages to be shown in the console
        "Number of Steps": ep_length, #Number of steps before triggering the gradient
        "Batch size" : 128, #Size of the gradient
        "Number of Epochs" : 3, #Number of passes through the gradient
        "Gamma" : 0.998, #Discount Factor used by some calculations in 
        "Learning Rate" : 0.0003 #Speed at which model learns
    }
    
    env_config = {
        'action_freq': 24, 'init_state': INIT_STATE_FILE_PATH,
        'gb_path': ROM_PATH, 'max_steps': ep_length,
        'progressLogs': PROGRESS_LOG, 'batl_mult' : 1,
        'expl_mult': 1, "head": 1, "model_config": model_config
    }
    
    
    max = psutil.cpu_count()
    num_cpu = 1
    #Use DummyVecEnc whenever you need to troubleshoot, similar requirements but subproc is a lot more vague on exceptions
    env = SubprocVecEnv([make_env(env_config,i) for i in range(num_cpu)])
    #env = DummyVecEnv([make_env(env_config,i) for i in range(num_cpu)])

    #setup directories for saving training data
    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'
    
    #setup model saving callback
    callback = TrainAndLoggingCallback(check_freq=ep_length, save_path=CHECKPOINT_DIR)
    
    #create reinforcement learning model
    
    #'''
    model = PPO( #Proximal Policy Optimization, Reinforcement learning algorithm
        model_config["Policy"], 
        env, 
        verbose=model_config["Verbose"], 
        tensorboard_log=LOG_DIR,
        n_steps=model_config["Number of Steps"], 
        batch_size=model_config["Batch size"], 
        n_epochs=model_config["Number of Epochs"], 
        gamma=model_config["Gamma"], 
        learning_rate=model_config["Learning Rate"])
    ''' 
    model= PPO.load('./train/model37.zip', env=env)
    #''' 
    #model= PPO.load('./train/model60.zip', env=env)
    model.learn(total_timesteps=(ep_length ) *num_cpu*5000,callback = callback)