"""Network"""
from keras import initializations
from keras.layers import (
    Conv2D, MaxPooling2D, Input, merge,
    Activation, Flatten, Dense
)
from keras.models import Model


my_init = (
    lambda shape, dim_ordering=None,
    name=None: initializations.normal(shape, scale=0.01)
)


class ConvNet:
    """ConvNet"""

    def __init__(self, input_shape, nbr_action, use_actions=False,
                 nbr_previous_action=0, weight_fname=None, print_model=True):
        """Init

        :param input_shape: tuple, input shape of the network
        :param nbr_action: int, nbr of actions in the environment
        :param use_actions: boolean, whether or not to use a second input to
         the network to feed the action history
        :param nbr_previous_action: nbr of previous action to feed to the
         network
        :param weight_fname: filename to potentially initialize the weights
        :param print_model: boolean, whether or not to print the model
         summary
        """
        
        assert isinstance(input_shape, tuple)
        assert isinstance(nbr_action, int)
        assert isinstance(use_actions, bool)
        assert isinstance(nbr_previous_action, int)
        assert isinstance(print_model, bool)
        assert weight_fname is None or isinstance(weight_fname, str)
        assert len(input_shape) > 0
        assert nbr_action > 0
        assert nbr_previous_action >= 0

        self.input_shape = input_shape
        self.nbr_action = nbr_action
        self.use_actions = use_actions
        self.nbr_previous_action = nbr_previous_action
        self.weight_fname = weight_fname
        self.print_model = print_model

        self.model = self._build_model()
        if self.weight_fname:
            self.load_weights(self.weight_fname)

    def _build_model(self):
        """Build the model"""

        model_input = Input(shape=self.input_shape)
        layer = model_input

        layer = Conv2D(32, 8, 8, subsample=(4, 4), init=my_init)(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(64, 4, 4, subsample=(2, 2), init=my_init)(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(64, 3, 3, subsample=(1, 1), init=my_init)(layer)
        layer = Activation('relu')(layer)
        layer = Flatten()(layer)
        layer = Dense(512, init=my_init)(layer)
        layer = Activation('relu')(layer)

        # Add the action history if specified
        if self.use_actions:
            action_input = Input(
                shape=(self.nbr_previous_action * self.nbr_action,)
            )
            layer = merge([layer, action_input], mode='concat')
            model_input = [model_input, action_input]

        output_layer = Dense(self.nbr_action, init=my_init)(layer)
        model = Model(input=model_input, output=output_layer)
        if self.print_model:
            model.summary(line_length=115)

        return model

    def load_weights(self, weight_fname):
        """Load weights

        :param weight_fname: filename of the weights
        """

        self.model.load_weights(weight_fname)

    def q_value(self, model_input):
        """Predict Q values

        :param model_input: input to the model
        :return: return prediction
        """

        return self.model.predict(model_input)

    def save_weights(self, weight_fname):
        """Save weights

        :param weight_fname: filename of the weights
        """

        self.model.save_weights(weight_fname)
