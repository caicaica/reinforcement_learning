"""Training"""

from trainer import DQNLearning


dqlearning = DQNLearning(weight_fname='/Users/matthieule/temp/test.h5',
                         use_actions=True, nbr_obs=4, nbr_past_actions=2,
                         env_id='SpaceInvaders-v0', episode_count=1,
                         buffer_size=10)

dqlearning.train()
