"""Training"""
import os

from dqn.dqn_learning import DQNLearning


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    """Main"""

    dqlearning = DQNLearning(
        env_id='SpaceInvaders-v0', weight_fname='/home/matthieu/temp/test.h5'
    )
    dqlearning.train()


if __name__ == '__main__':

    main()
