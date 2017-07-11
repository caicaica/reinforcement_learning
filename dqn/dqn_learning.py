"""Trainer"""
import keras.backend as K

from .agent import DQNAgent
from .environment import ProcessedEnvironnement
from .cnn import ConvNet

from keras.optimizers import Adam


def clipped_l2_loss(y_true, y_pred):

    return K.mean(K.clip(K.square(y_pred - y_true), -10, 10))


class DQNLearning:

    def __init__(self, env_id, weight_fname, use_actions=False, new_size=84, nbr_obs=4,
                 episode_count=1000000, buffer_size=50000, nbr_past_actions=0,
                 update_freq=10000, batch_size=32, gamma=0.99,
                 optimizer=Adam(lr=2.5e-4, clipnorm=1.),
                 loss=clipped_l2_loss, epsilon=1.0, decay=1e-6,
                 epsilon_min=0.1, action_repetition_rate=4,
                 no_op_max=10, no_op_action=0):
        """Init

        :param env_id: id of the gym environment
        :param weight_fname: filename of the weights
        :param use_actions: boolean, whether or not to use the action history
         as input to the network
        :param new_size: int, new size of the input
        :param nbr_obs: int, number of observation to feed during a forward
         network pass (which are stacked in the channel dimension)
        :param episode_count: number of episode to go through during training
        :param buffer_size: size of the history buffer, set it to 0 for
         inference
        :param nbr_past_actions: int, number of past action (past the last
         frame given as input to the network) use for each forward pass
        :param update_freq: frequency at which to update the target model
         which is also used to save the model
        :param batch_size: int, batch size
        :param gamma: float, gamma parameter of the reward decay
        :param optimizer: optimizer instance of the Keras optimizers
        :param loss: loss for the model
        :param epsilon: epsilon parameter of the policy
        :param decay: decay rate of epsilon
        :param epsilon_min: minimum epsilon value
        :param action_repetition_rate: rate at which action should be repeated
         for computational efficiency
        :param no_op_max: int, maximum number of no_op action to perform at the
         start of an episode
        :param no_op_action: index of the action which results in not doing
         anything
        """

        self.env = ProcessedEnvironnement(env_id)
        self.env.seed(0)
        self.nbr_action = self.env.action_space.n
        self.batch_size = batch_size
        self.counter = 0
        self.episode_count = episode_count
        self.gamma = gamma
        self.update_freq = update_freq
        self.use_actions = use_actions
        self.weight_fname = weight_fname
        self.action_repetition_rate = action_repetition_rate
        self.no_op_max = no_op_max
        self.no_op_action = no_op_action
        self.buffer_size = buffer_size

        _ = self.env.reset()
        input_shape = (new_size, new_size, nbr_obs)

        network_agent = ConvNet(
            input_shape=input_shape, nbr_action=self.nbr_action,
            use_actions=use_actions, print_model=False,
            nbr_previous_action=nbr_obs + nbr_past_actions
        )
        self.network = ConvNet(
            input_shape=input_shape, nbr_action=self.nbr_action,
            use_actions=use_actions,
            nbr_previous_action=nbr_obs + nbr_past_actions
        )
        self.network.model.compile(optimizer=optimizer, loss=loss)
        self.agent = DQNAgent(
            action_space=self.env.action_space, network=network_agent,
            obs_shape=(new_size, new_size, 1), nbr_obs=nbr_obs,
            nbr_past_actions=nbr_past_actions, buffer_size=buffer_size,
            use_actions=use_actions, epsilon=epsilon, decay=decay,
            epsilon_min=epsilon_min
        )
        self._save_network()

    def _learn(self):
        """Fit the model to a batch of data"""

        features, labels = self.agent.get_training_data(
                self.batch_size, self.gamma, self.network
        )
        verbose = 0
        if self.counter % 50 == 0:
            print(self.counter)
            verbose = 1
        self.network.model.fit(
            features, labels, nb_epoch=1, batch_size=self.batch_size,
            verbose=verbose
        )
        if self.counter % self.update_freq == 0:
            self._save_network()
        self.counter += 1

    def _save_network(self):
        """Save network and update target"""

        self.network.save_weights(self.weight_fname)
        self.agent.network.load_weights(self.weight_fname)

    def train(self):
        """Train the model"""

        action = 0
        reward = 0
        warm_up_counter = 0
        while warm_up_counter < self.buffer_size:
            action_rep_counter = 0
            ob = self.env.reset()
            done = True
            while True:
                if action_rep_counter % self.action_repetition_rate == 0:
                    action = self.agent.act(
                        ob, reward, done, random=True, no_op_max=self.no_op_max,
                        no_op_action=self.no_op_action
                    )
                    warm_up_counter += 1
                ob, reward, done, _ = self.env.step(action)
                action_rep_counter += 1
                if done:
                    break
        for i in range(self.episode_count):
            action_rep_counter = 0
            ob = self.env.reset()
            done = True
            while True:
                if action_rep_counter % self.action_repetition_rate == 0:
                    action = self.agent.act(
                        ob, reward, done, no_op_max=self.no_op_max,
                        no_op_action=self.no_op_action
                    )
                    self._learn()
                ob, reward, done, _ = self.env.step(action)
                action_rep_counter += 1
                if done:
                    break
