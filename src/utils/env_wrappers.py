import numpy as np
import gymnasium.spaces as spaces

class BaseEnvWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.unwrapped = env.unwrapped


    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

class FlappyBirdWrapper(BaseEnvWrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        return obs, reward, done, info
    
class TetrisWrapper(BaseEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        board_shape = (162, 81, 1)     
        next_block_shape = (25, 33, 1)  

        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=255, shape=board_shape, dtype=np.uint8),
            spaces.Box(low=0, high=255, shape=next_block_shape, dtype=np.uint8)
        ))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._process_obs(obs)

    def _process_obs(self, obs):
        obs = obs.mean(axis=2, keepdims=True).astype(obs.dtype)
        binary_obs = (obs > 0).astype(obs.dtype)

        board_area = binary_obs[47:209, 95:176, :]    
        next_block_area = binary_obs[115:140, 191:224, :]  


        return board_area, next_block_area

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info        
