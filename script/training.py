"""Training"""

from trainer import DQNLearning


dqlearning = DQNLearning(weight_fname='/Users/matthieule/temp/test.h5',
                         use_actions=True, nbr_past_actions=10,
                         env_id='SpaceInvaders-v0', episode_count=10000000,
                         update_freq=10000)

dqlearning.train()
