"""Inference"""
from dqn.environment import ProcessedEnvironnement
from dqn.agent import DQNAgent
from dqn.cnn import ConvNet


def main():
    """Main"""

    env_id = 'SpaceInvaders-v0'
    weight_fname = '/home/matthieu/temp/test.h5'

    env = ProcessedEnvironnement(
        env_id, outdir='/home/matthieu/temp/random-agent-results',
        wrappers_cond=True
    )
    env.seed(0)
    network = ConvNet(
        input_shape=(84, 84, 1), nbr_action=env.action_space.n,
        weight_fname=weight_fname
    )
    agent = DQNAgent(
        action_space=env.action_space, network=network,
        obs_shape=(84, 84, 1), buffer_size=6, decay=0.0, epsilon=0.9
    )
    episode_count = 1
    reward = 0
    action_repetition_rate = 4
    action = 0
    for i in range(episode_count):
        ob = env.reset()
        done = True
        counter = 0
        while True:
            if counter % action_repetition_rate == 0:
                action = agent.act(ob, reward, done)
                print(action)
            ob, reward, done, _ = env.step(action)
            counter += 1
            if done:
                break

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':

    main()
