#Billy Matthews
#wmatthe1@stumail.northeaststate.edu



#import vectorization wrappers
from gymnasium import Env, spaces
from stable_baselines3.common.utils import set_random_seed
import numpy as np
#import ppo for algos
from stable_baselines3 import PPO
#import register to register pokemon yellow as a custom gym environment
from gymnasium.envs.registration import register
from envs.YellowBaselinesEnv import YellowEnv
from IPython.display import clear_output


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
    fileChoice = 6
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
    else:
        stateFile = "./states/PokemonYellowVersion.gb.state"
    #setup directories
    ROM_PATH = "PokemonYellowVersion.gb" #File location of Pokemon Yellow
    INIT_STATE_FILE_PATH = stateFile #File location of the starting state of the rom
    PROGRESS_LOG = './logs/progressLogs/'
    ep_length = 2048 *10
    #sets up configurations for the environment
    env_config = {
        'action_freq': 24, 'init_state': INIT_STATE_FILE_PATH,
        'gb_path': ROM_PATH, 'max_steps': ep_length,
        'progressLogs': PROGRESS_LOG, 'batl_mult' : 1,
        'expl_mult': 1
    }
    
    
    env = YellowEnv(env_config,0)
    states = []
    rewards = []
    
    #Variable to select what model to test out: I have made more changes to reward leading to my desktop currently training on the new model
    #I have not yet gone through these individual models to figure out most of their behavior. noticeableSuccess and ScaredOfBaldMen are examples
    #of the model learning with small crippling problems. IE, ScaredOfBaldMen was the first model trained after adding a punishment for fleeing battles
    #However, in the second town there is a npc that puts you in a mandatory catching tutorial which triggered as a fled battle upon completion
    #This punished the model and it learned to avoid that npc, making it scared of bald men.
    
    #ComputerNerd/MallWalker both learned destructive behavior due to a reward value leak.
    modelChoice = 15
    
    if modelChoice == 0:
        env.output_shape = (36, 40, 3) # this and following line are needed to fix where the output shape was grayscaled in recent generations
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8) # causing the model and current env to be incompatible
        model= PPO.load('./train/noticeableSuccess.zip', env=env)
    elif modelChoice ==1:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/Fred.zip', env=env)
    elif modelChoice == 2:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/ScaredOfBaldMen.zip', env=env)
    elif modelChoice == 3:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/ComputerNerd.zip', env=env)
    elif modelChoice == 4:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/CurrentProject.zip', env=env)
    elif modelChoice == 5:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/IAmBlind.zip', env=env)
    elif modelChoice == 6:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/LikesToFight.zip', env=env)
    elif modelChoice == 7:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/MallWalker.zip', env=env)
    elif modelChoice == 8:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/OopsAllSingle.zip', env=env)
    elif modelChoice == 9:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/PunishedMovement.zip', env=env)
    elif modelChoice == 10:
        env.output_shape = (36, 40, 1)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/TestBest1.zip', env=env)
    elif modelChoice == 11:
        env.output_shape = (36, 40, 1)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/TestBest2.zip', env=env)
    elif modelChoice == 12:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/TooHighLevelsReward.zip', env=env)
    elif modelChoice == 13:
        env.output_shape = (36, 40, 1)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/BlindCheck.zip', env=env)
    elif modelChoice == 14:
        env.output_shape = (36, 40, 1)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/7hours-1.zip', env=env)
    elif modelChoice == 15:
        env.output_shape = (36, 40, 1)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/7hours.zip', env=env)
    elif modelChoice == 16:
        env.output_shape = (36, 40, 1)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/7hours+1.zip', env=env)
        
    else:
        env.output_shape = (36, 40, 3)
        env.observation_space = spaces.Box(low=0,high=255,shape=env.output_shape,dtype=np.uint8)
        model= PPO.load('./train/ViridianTourist.zip', env=env)
    
    state, [] = env.reset()
    while True:
        action, state = model.predict(state)
        state,reward,truncated,terminated, stuff = env.step(action)
        env.render()

   



    
        

