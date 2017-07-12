## WIP

This is a work in progress: the code seems to run somewhat smoothly, 
but I have not taken the time to get 1 week worth of GPU to train it
entirely :crying_cat_face:

## Description

This is an implementation of Deep Q learning [1] based on OpenAI Gym's
environment.

## Installation

Clone this repository:

```bash
git clone https://github.com/matthieule/reinforcement_learning.git
cd reinforcement_learning
conda env create -f environment.yml
```

Activate the environment:

```bash
source activate openai
```

Export the path to the repository:
```bash
export PYTHONPATH={path_to_reinforcement_learning}:$PYTHONPATH
```

You *might* need to install the following:
```bash
pip install gym'[all]'
conda install -c conda-forge tensorflow
```

## Code Organization

This code revolves around the 4 classes:
- **dqn.agent.DQNAgent**: an agent acting on the environment, and holding 
a history of the past actions, observations, and rewards.
- **dqn.cnn.ConvNet**: a deep Q network, fairly close to the one described
in [1], but with the ability to add an history of the past actions to
the one before last fully connected layer.
- **dqn.dqn_learning.DQNLearning**: the learning class which instantiate
the agent, and the target network and train the agent.
- **dqn.environment.ProcessedEnvironnement**: a simple wrapper around the
Gym environment which highlights which methods are necessary. It should
be easily extended to any other 2D problems.
- **dqn.history.History**: a buffer of the past actions, observations, and
rewards. It is also responsible for pre-processing the observations.

## Organization of observations / actions / rewards

The History class holds a buffer of the events in the attributes 
past_obs, past_actions, past_rewards, and past_done. As such, the tuple
(past_obs[i], past_rewards[i], past_done[i], past_actions[i]) holds:
- the current observation
- the reward the agent got leading to the current observation
- done indicate whether the previous observation was the end of an 
episode
- the action taken on this state which will lead to a new observation

## Example Usage

Training

```bash
cd reinforcement_learning
python script/training.py
```

Inference

```bash
python script/inference.py
```

This should create a video in the 
`/home/matthieu/temp/random-agent-results` folder to visualize the 
results (you might wanna change the folder name :grimacing:).

Look at the notebook `Exploratory Analysis.ipynb` for simple examples
of usage of the different classes.

## References

[1] Mnih, Volodymyr, et al. "[Human-level control through deep reinforcement learning.](http://www.davidqiu.com:8888/research/nature14236.pdf)" Nature 518.7540 (2015): 529-533.]

