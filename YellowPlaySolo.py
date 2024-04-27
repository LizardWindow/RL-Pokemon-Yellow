#Billy Matthews
#wmatthe1@stumail.northeaststate.edu

import sys
import gymnasium as gym
import numpy as np
from pyboy import PyBoy
from gym import Env, spaces
from pyboy.utils import WindowEvent
from IPython.display import clear_output
class YellowGymEnv(Env):
    def __init__(self,config):
        
        super(YellowGymEnv,self).__init__()
        
        #Creates an array of valid actions
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PASS
        ]
        #Sets gym's action space to the list of valid actions
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
        
        
        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP
            
        ]
        
        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B
        ]
        
        self.output_shape = (144,160, 1)
        self.output_full_shape = (144,160,3)
        self.observation_space = spaces.Box(low=0,high=255,shape=self.output_full_shape,dtype=np.uint8)
        
        
        self.act_freq = config['action_freq']
        
        head='SDL2'
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
        
        self.screen = self.pyboy.botsupport_manager().screen()
        
        
        self.agent_stats = []
        self.total_reward=0
    
    def render(self):
        game_pixels_render = self.screen.screen_ndarray()
        return game_pixels_render

    def reset(self):
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        
        self.total_reward = 0
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
        X_POS_ADDRESS, Y_POS_ADDRESS = 0XD361,0XD360
        LEVELS_ADDRESSES = [0xD18B, 0xD1B7, 0xD1E3, 0xD20F,0xD23B,0xD267]
        x_pos = self.pyboy.get_memory_value(X_POS_ADDRESS)
        y_pos = self.pyboy.get_memory_value(Y_POS_ADDRESS)
        levels = [self.pyboy.get_memory_value(a) for a in LEVELS_ADDRESSES]
        self.agent_stats.append({
            'x': x_pos, 'y': y_pos, 'levels': levels
        })
        
        obs_memory = self.render()
        new_reward = levels
        
        
        terminated = False
        truncated = False
        
        return obs_memory, new_reward, terminated, truncated, {}

    def close(self):
        self.pyboy.stop()
        super().close()
ROM_PATH = "PokemonYellowVersion.gb"
INIT_STATE_FILE_PATH = "./states/TeamCaughtLeveled.state"

env_config = {
    'action_freq': 24, 'init_state': INIT_STATE_FILE_PATH,
    'gb_path': ROM_PATH
}
env = YellowGymEnv(env_config)
env.reset()
states = []
rewards = []

try:
    for i in range(3000000000):
        observation,reward,terminated,truncated, _ =env.step(7)
        states.append(observation)
        rewards.append(reward)
        
        clear_output(wait=True)
        #plt.imshow(env.render())
        #plt.show()
finally:
    env.close()
