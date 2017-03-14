"""History class remembering the past observations and actions"""

class History:
    """History class"""

    def __init__(self, obs_shape, nbr_obs, nbr_past_actions):
        """Init
        
        :param obs_shape: shape of the input observations
        :param nbr_obs: number of observations to keep in the history
        :param nbr_past_actions: number of actions in the past to keep in the
         history on top of the past observations
        """
        
        assert isinstance(obs_shape, tuple)
        assert isinstance(nbr_obs, int)
        assert isinstance(nbr_past_actions, int)
        assert nbr_obs > 0
        assert nbr_past_actions >= 0

        self.nbr_observations = n_observations
        self.nbr_actions = n_observations + nbr_past_actions
        self.past_obs = []
        self.past_actions = []
        self.past_rewards = []

    @staticmethod
    def _update_buffer(current_buffer, buffer_max_size, element):
        """Update the considered buffer"""
        
        assert isinstance(current_buffer, list)
        assert isinstance(buffer_max_size, int)
        assert buffer_max_size > 0

        if len(current_buffer) == buffer_max_size:
            current_buffer.pop(0)
        current_buffer.append(element)

    def _update_observations(self, obs):
        """Add observation to the history buffer
        
        :param obs: ndarray of shape self.obs_shape to add to the history
        """

        assert isinstance(obs, np.array)
        assert obs.shape == self.obs_shape
        self._update_buffer(self.past_obs, self.nbr_observations, obs)

    def _update_actions(self, action):
        """Add action to the history buffer

        :param action: int indicating the action
        """

        assert isinstance(action, int)
        self._update_buffer(self.past_actions, self.nbr_actions, action)

    def _update_rewards(self, reward):
        """Add reward to the history buffer

        :param reward: int indicating the action
        """

        assert isinstance(reward, float)
        self._update_buffer(self.past_rewards, self.nbr_observations, reward)

    def update_history(self, obs, action, reward):

        self._update_observations(obs)
        self._update_actions(action)
        self._update_rewards(reward)

