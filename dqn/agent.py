"""Agent"""

import numpy as np

from .history import History
from .convnet import ConvNet

np.random.seed(0)


class DQNAgent(object):
    """Agent"""

    def __init__(self, action_space, network, obs_shape, nbr_obs=4,
                 nbr_past_actions=0, buffer_size=1000000, use_actions=False,
                 epsilon=1.0, decay=1e-6, epsilon_min=0.1):
        """Init

        :param action_space: action_space attribute of an OpenAI gym
         environment
        :param network: ConvNet instance
        :param obs_shape: tuple of length 2 or 3 providing the input shape
        :param nbr_obs: int, number of observation to feed during a forward
         network pass (which are stacked in the channel dimension)
        :param nbr_past_actions: int, number of past action (past the last
         frame given as input to the network) use for each forward pass
        :param buffer_size: size of the history buffer, set it to 0 for
         inference
        :param use_actions: boolean, whether or not to use the action history
         as input to the network
        :param epsilon: epsilon parameter of the policy
        :param decay: decay rate of epsilon
        :param epsilon_min: minimum epsilon value
        """

        assert isinstance(network, ConvNet)
        assert isinstance(obs_shape, tuple)
        assert isinstance(nbr_obs, int)
        assert isinstance(nbr_past_actions, int)
        assert isinstance(buffer_size, int)
        assert isinstance(use_actions, bool)
        assert isinstance(epsilon, float)
        assert nbr_obs > 0
        assert nbr_past_actions >= 0
        assert buffer_size >= 0
        assert 0 <= epsilon <= 1

        self.action_space = action_space
        self.network = network
        self.history = History(
            obs_shape=obs_shape, nbr_obs=nbr_obs,
            nbr_past_actions=nbr_past_actions, buffer_size=buffer_size,
            use_actions=use_actions
        )
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.use_actions = use_actions
        self.nbr_action = self.action_space.n

    def _evaluate_max_repeat_condition(self, no_op_max, no_op_action=0):
        """Return True if all the previous no_op_max action are no op"""

        index_min = min(len(self.history.past_actions) - no_op_max, 0)
        considered_actions = np.array(self.history.past_actions[index_min:])

        return np.all(considered_actions == no_op_action)

    def _format_action_input(self, action):
        """Format the action input to feed to the network

        :param action: array (1, nbr_obs+nbr_past_actions) of action
        :return: array (1, self.action_space.n*len(action)) with the actions
         encoded as 1-hot vectors and flattened
        """

        input_action = np.zeros((self.action_space.n, len(action)))
        for act in np.squeeze(action):
            if 0 < act < self.nbr_action:
                input_action[act] = 1
        input_action = np.reshape(input_action, (1, -1))

        return input_action

    def _format_input(self, network_input):
        """Format the input to the network

        :param network_input: input to the network, either an array of size
         (None, *obs_shape), or a list
         [array(None, *obs_shape), (None, nbr_obs+nbr_past_actions)]
        :return: inference input formatted to feed to the network
        """

        if self.use_actions:
            assert isinstance(network_input, list)
            assert len(network_input) == 2
            action = network_input[1]
            input_action = self._format_action_input(action)
            return [network_input[0], input_action]
        else:
            return network_input

    def _predict(self, obs, action, network):
        """Predict the Q value using the network

        :param obs: observation from the gym environment
        :param action: action history
        :param network: network to use
        :return: predicted Q value
        """

        if self.use_actions:
            input_action = self._format_action_input(action)

        model_input = [obs, input_action] if self.use_actions else obs
        qval = network.q_value(model_input)[0]
        return qval

    def _update_policy(self):
        """Update the policy followed by the agent"""

        self.epsilon = max(self.epsilon - self.decay, self.epsilon_min)

    def act(self, obs, reward, done, random=False, no_op_max=30,
            no_op_action=0, action_to_take=None):
        """Act on the observation

        :param obs: observation from the gym environment
        :param reward: current reward
        :param done: boolean, if true the episode ended with the previous
         action
        :param random: boolean, if true, act as a random agent
        :param no_op_max: int, maximum number of no_op action to perform at the
         start of an episode
        :param no_op_action: index of the action which results in not doing
         anything
        :param action_to_take: if not None, the agent will take this action
        :return: int, action to take in the environment
        """
        if action_to_take is not None:
            action = action_to_take
        elif random:
            action = self.action_space.sample()
        else:
            max_repeat_condition = self._evaluate_max_repeat_condition(
                no_op_max, no_op_action
            )
    
            if np.random.rand() <= self.epsilon or max_repeat_condition:
                action = self.action_space.sample()
            else:
                inference_input = self.history.get_inference_input(obs)
                inference_input = self._format_input(inference_input)
                q_value = self.network.model.predict(inference_input)[0]
                print(q_value)
                action = np.argmax(q_value)
            self._update_policy()

        self.history.update_history(obs, action, float(reward), done)

        return action

    def get_training_data(self, batch_size, gamma, current_network):
        """Get the data used for a batch update

        :param batch_size: int, batch size
        :param gamma: float, gamma parameter of the reward decay
        :param current network: current network being trained
        :return: tuple X, Y with the training data
        """

        assert isinstance(batch_size, int)
        assert isinstance(gamma, float)
        assert batch_size > 0
        assert 0 <= gamma <= 1

        training_data_list = self.history.get_training_data(batch_size)

        Y_list = []
        X_list_obs = []
        X_list_action = []

        for training_data_dict in training_data_list:
            old_qval = self._predict(
                training_data_dict['obs'][None, ..., :-1],
                training_data_dict['action_taken'][..., :-1],
                current_network
            )
            new_qval = self._predict(
                training_data_dict['obs'][None, ..., 1:],
                training_data_dict['action_taken'][..., 1:],
                self.network
            )
            best_action = np.argmax(new_qval)
            maxQ = new_qval[best_action]
            Y = np.zeros((1, self.nbr_action))
            Y[:] = old_qval[:]
            done = training_data_dict['done']
            update = training_data_dict['reward'] + (gamma * maxQ)*(1-done)
            Y[0][best_action] = update
            Y_list.append(Y)
            X_list_obs.append(training_data_dict['obs'])
            X_list_action.append(training_data_dict['action_taken'])

        Y = np.concatenate(Y_list)

        input_obs = np.stack(X_list_obs)
        if self.use_actions:
            input_action = [
                self._format_action_input(action)
                for action in X_list_action
            ]
            input_action = np.concatenate(input_action)
            X = [input_obs, input_action]
        else:
            X = input_obs

        return X, Y
