"""Training"""
import os

from dqn.trainer import DQNLearning


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

dqlearning = DQNLearning(
    env_id='SpaceInvaders-v0', weight_fname='/home/matthieu/temp/test.h5'
)

dqlearning.train()
