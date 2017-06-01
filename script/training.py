"""Training"""

from dqn.trainer import DQNLearning

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

dqlearning = DQNLearning(weight_fname='/home/matthieu/temp/test.h5',
                         use_actions=False, nbr_past_actions=0, nbr_obs=4,
                         env_id='SpaceInvaders-v0', episode_count=1000000,
                         update_freq=10000, buffer_size=100000, batch_size=32,
                         decay=1e-6)

dqlearning.train()
