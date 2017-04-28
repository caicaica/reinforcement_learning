"""Training"""

from trainer import DQNLearning

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

dqlearning = DQNLearning(weight_fname='/home/matthieu/temp/test.h5',
                         use_actions=True, nbr_past_actions=10,
                         env_id='SpaceInvaders-v0', episode_count=10000000,
                         update_freq=10000)

dqlearning.train()
