import argparse
import logging
import sys

import gym
from gym import wrappers

from agent import DQNAgent
from network import Network
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

    env = gym.make(args.env_id)

    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    weight_fname = None
    use_actions = False
    nbr_obs = 2
    ob = env.reset()
    input_shape = (ob.shape[0], ob.shape[1], nbr_obs*ob.shape[2])
    network = Network(input_shape=input_shape, nbr_action=env.action_space.n,
                      use_actions=use_actions, weight_fname=weight_fname)
    history = History(obs_shape=ob.shape, nbr_obs=nbr_obs, nbr_past_actions=0,
                      use_actions=network.use_actions, buffer_size=10)
    agent = DQNAgent(env.action_space, network, history)

    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    # for i in range(100):
    #     test = agent.history.get_training_input(4)
    #     print(test.shape)

    # Close the env and write monitor result info to disk
    env.close()
    # Upload to the scoreboard
    #gym.upload(outdir)