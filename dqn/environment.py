"""Environment with some preprocessing"""
import numpy as np

import gym
from gym import wrappers


class NonRGBImage(Exception):
    """Raised when a RGB image is expected but not provided"""

    pass


class ProcessedEnvironnement:
    """Wrapper around a OpenAI environment to add some pre-processing"""

    def __init__(self, env_id, outdir=None, wrappers_cond=False):
        """Init

        :param env_id: id of the OpenAI environment
        :param outdir: output directory
        :param wrappers_cond: boolean, whether or not to call wrappers.Monitor
        """

        self.env = gym.make(env_id)
        if wrappers_cond:
            self.env = wrappers.Monitor(self.env, directory=outdir, force=True)
        self.action_space = self.env.action_space

    def close(self):
        """Close the environment"""

        self.env.close()

    def reset(self):
        """Reset the environment

        :return: initial preprocessed observation
        """

        ob = self.env.reset()

        return ob

    def seed(self, int):
        """Set the random seed of the environment"""

        self.env.seed(int)

    def step(self, action):
        """Take a step in the environment

        :param action: int, action id
        :return: tuple with observed preprocessed observation, reward, boolean
         whether the episode is done or not, info
        """

        ob, reward, done, info = self.env.step(action)

        if done:
            reward = -1
        reward = np.clip(reward, -1, 1)

        return ob, reward, done, info
