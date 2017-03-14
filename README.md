Network class which predict for each action, the future reward given the
observation

Agent class which takes a decision based on the observation and given the
Network

An history class which remembers the last N observations and the last N+P
actions taken

At training, randomly take a batch of size B of S consecutive observations and S+P
consecutive actions. The observations are fed to the convnet, the actions are
added at the fully connected layer
