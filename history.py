"""History class remembering the past observations and actions"""

import numpy as np


class History:
    """History class

    Holds a buffer with current observation, action taken, and past reward
    """

    def __init__(self, obs_shape, nbr_obs=4, nbr_past_actions=0,
                 buffer_size=1000000, use_actions=False):
        """Init

        :param obs_shape: tuple of length 2 or 3 providing the input shape
        :param nbr_obs: int, number of observation to feed during a forward
         network pass (which are stacked in the channel dimension)
        :param nbr_past_actions: int, number of past action (past the last
         frame given as input to the network) use for each forward pass
        :param buffer_size: size of the history buffer, set it to 0 for
         inference
        :param use_actions: boolean, whether or not to use the action history
         as input to the network
        """
        
        assert isinstance(obs_shape, tuple)
        assert isinstance(nbr_obs, int)
        assert isinstance(nbr_past_actions, int)
        assert isinstance(buffer_size, int)
        assert isinstance(use_actions, bool)
        assert nbr_obs > 0
        assert nbr_past_actions >= 0
        assert buffer_size >= 0

        self.obs_shape = obs_shape
        self.nbr_observations = nbr_obs
        self.nbr_actions = nbr_obs + nbr_past_actions
        self.buffer_size = nbr_obs + nbr_past_actions + buffer_size
        self.use_actions = use_actions

        self.past_obs = []
        self.past_obs_original = []
        self.past_actions = []
        self.past_rewards = []
        self.past_done = []

    def _get_sample(self, index, nbr_observations, nbr_actions):
        """Get a sample of the stored data

        It provides a list of nbr_observations where the last one is the index
        one, and a list of nbr_actions where the last one is the index-1 one
        (i.e. the action taken to get to observation index)

        :param index: index where to get the data from the history
        :param nbr_observations: number of observations to stack
        :param nbr_actions: number of actions to stack
        :return: tuple with a sample (as an array) of observation and action
        """

        padding = max(nbr_observations - index, 0)
        index_min = max(index-nbr_observations, 0)
        sample_obs = np.concatenate(
            padding*[np.zeros(self.obs_shape)]+self.past_obs[index_min:index],
            axis=-1
        )
        index_min = max(index - 1 - nbr_actions, 0)
        padding = max(
            nbr_actions - len(self.past_actions[index_min:index-1]), 0
        )
        sample_action = np.array(
            padding*[-1]+self.past_actions[index_min:index-1]
        )

        return sample_obs, sample_action

    def _update_actions(self, action):
        """Update the action buffer

        :param action: int, action
        """

        assert isinstance(action, int)
        self._update_buffer(self.past_actions, action)

    def _update_end_episode(self, done):
        """Update the done buffer

        :param done: boolean, if true the episode ended with the previous
         action
        """

        assert isinstance(done, bool)
        self._update_buffer(self.past_done, done)

    def _update_buffer(self, current_buffer, element):
        """General buffer update function

        :param current_buffer: list, buffer to update
        :param element: element to add to the buffer
        """
        
        assert isinstance(current_buffer, list)

        if len(current_buffer) == self.buffer_size:
            current_buffer.pop(0)
        current_buffer.append(element)

    def _update_observations(self, obs):
        """Add observation to the history buffer
        
        :param obs: ndarray of shape self.obs_shape to add to the history
        """

        assert isinstance(obs, np.ndarray)
        assert obs.shape == self.obs_shape
        self._update_buffer(self.past_obs_original, obs)
        obs_processed = np.max(np.array(self.past_obs_original[-3:]), axis=0)
        self._update_buffer(self.past_obs, obs_processed)

    def _update_rewards(self, reward):
        """Add reward to the history buffer

        :param reward: int indicating the action
        """

        assert isinstance(reward, float)
        self._update_buffer(self.past_rewards, reward)

    def get_inference_input(self, obs):
        """Get the input necessary for inference"""

        n = len(self.past_obs)
        inference_input_obs, inference_input_action = self._get_sample(
                n, self.nbr_observations-1, self.nbr_actions-1
        )
        inference_input_obs = np.concatenate(
                [inference_input_obs, obs],
                axis=-1
        )
        inference_input_obs = inference_input_obs[None, ...]
        inference_input_action = np.concatenate(
                [inference_input_action, [self.past_actions[-1]]],
                axis=-1
        )
        if self.use_actions:
            return [inference_input_obs, inference_input_action]
        else:
            return inference_input_obs

    def get_training_data(self, batch_size):
        """Get a batch of training data

        :param batch_size: int, size of the batch
        :return: list of dictionary with data used to train the model
        """

        assert isinstance(batch_size, int)
        assert batch_size > 0

        training_data_list = []

        for sample in range(batch_size):
            random_idx = np.random.randint(1, len(self.past_obs)-1)
            obs, action_taken = self._get_sample(
                    random_idx, self.nbr_observations, 
                    self.nbr_actions
            )
            new_obs, new_action_taken = self._get_sample(
                    random_idx+1, self.nbr_observations,
                    self.nbr_actions
            )
            reward = self.past_rewards[random_idx+1]
            done = self.past_done[random_idx+1]
            training_data_dict = {
                'obs': obs,
                'action_taken': action_taken,
                'new_action_taken': new_action_taken,
                'new_obs': new_obs,
                'reward': reward,
                'done': done
            }
            training_data_list.append(training_data_dict)

        return training_data_list

    def update_history(self, obs, action, reward, done):
        """Update the different history buffer

        :param obs: observation from the gym environment
        :param action: int, action taken
        :param reward: float, reward
        :param done: boolean, if true the episode ended with the previous
         action
        """

        self._update_observations(obs)
        self._update_actions(action)
        self._update_rewards(reward)
        self._update_end_episode(done)
