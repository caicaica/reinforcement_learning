## Description

Implementation of [1]

Object oriented implementation of the deep Q-network (DQN) for 2D inputs

## Install

Install [OpenAI gym](https://github.com/openai/gym)

```bash
mkdir openai
cd openai
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

Clone this repository

```bash
cd ..
git clone https://github.com/matthieule/reinforcement_learning.git
```

Install the conda environment

```bash
cd reinforcement_learning
conda create -n openai --file exported_packages.txt
```

Activate the environment

```bash
source activate openai
```

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

This should create a video in the `/tmp/` folder to visualize the results.

## References

[1] Mnih, Volodymyr, et al. "[Human-level control through deep reinforcement learning.](http://www.davidqiu.com:8888/research/nature14236.pdf)" Nature 518.7540 (2015): 529-533.]

