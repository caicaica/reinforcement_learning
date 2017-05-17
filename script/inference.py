import argparse 
import logging
import sys

import numpy as np

import gym

from environment import ProcessedEnvironnement
from agent import DQNAgent
from convnet import ConvNet
from history import History


def get_logger():
    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.
    logger.setLevel(logging.INFO)

    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='SpaceInvaders-v0',
                        help='Select the environment to run')
    args = parser.parse_args()

    logger = get_logger()

    env = ProcessedEnvironnement(args.env_id,
                                 outdir='/home/matthieu/temp/random-agent-results',
                                 wrappers_cond=True)
    env.seed(0)

    weight_fname = '/home/matthieu/temp/test.h5'
    use_actions = False
    nbr_obs = 4
    nbr_past_actions = 0
    ob = env.reset()
    input_shape = (ob.shape[0], ob.shape[1], nbr_obs*ob.shape[2])
    network = ConvNet(input_shape=input_shape, nbr_action=env.action_space.n,
                      use_actions=use_actions, weight_fname=weight_fname,
                      nbr_previous_action=nbr_obs + nbr_past_actions)
    history = History(obs_shape=ob.shape, nbr_obs=nbr_obs, nbr_past_actions=nbr_past_actions,
                      use_actions=network.use_actions, buffer_size=10)
    agent = DQNAgent(
        action_space=env.action_space, network=network,
        obs_shape=ob.shape, nbr_obs=nbr_obs,
        nbr_past_actions=nbr_past_actions, buffer_size=6,
        use_actions=use_actions, epsilon=0.05, decay=0.0,
        epsilon_min=0.0
    )

    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        done = False
        counter = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            import matplotlib.pyplot as plt 
            plt.imshow(agent.history.past_obs[-1][...,0], cmap='gray') 
            plt.axis('off') 
            plt.savefig('{}.png'.format(counter))
            plt.imshow(agent.history.past_obs_original[-1][...,0], cmap='gray')
            plt.axis('off')
            plt.savefig('_{}.png'.format(counter))
            counter += 1
            if done:
                break

    # Close the env and write monitor result info to disk
    env.close()
    # Upload to the scoreboard
    #gym.upload(outdir)
