"""Network"""
from keras import initializations
from keras.layers import (
    Conv2D, MaxPooling2D, Input, merge, BatchNormalization,
    Activation, Flatten, Dropout, Dense
)
from keras.models import Model


my_init = lambda shape, dim_ordering=None, name=None: initializations.normal(shape, scale=0.01)


class ConvNet:
    """ConvNet"""

    def __init__(self, input_shape, nbr_action, nbr_filters=32,
                 nbr_bottleneck=2, nbr_pooling=4, kernel_size=3,
                 projection_scale=4, nbr_fc_neurons=128, dropout=0.0,
                 use_actions=False, nbr_previous_action=10,
                 weight_fname=None):
        """Init

        :param input_shape: tuple, input shape of the network
        :param nbr_action: int, nbr of actions in the environment
        :param nbr_filters: int, nbr of filters of the first layer
        :param nbr_bottleneck: int, number of bottleneck per module
        :param nbr_pooling: nbr of downsampling layers
        :param kernel_size: size of the kernel
        :param projection_scale: projection scale at which the number of
         feature map is reduced inside a bottleneck
        :param nbr_fc_neurons: nbr of neuron in the first fully connected
         layer
        :param dropout: dropout probability between the fully connected layers
        :param use_actions: boolean, whether or not to use a second input to
         the network to feed the action history
        :param nbr_previous_action: nbr of previous action to feed to the
         network
        :param weight_fname: filename to potentially initialize the weights
        """
        
        assert isinstance(input_shape, tuple)
        assert isinstance(nbr_action, int)
        assert isinstance(nbr_filters, int)
        assert isinstance(nbr_bottleneck, int)
        assert isinstance(nbr_pooling, int)
        assert isinstance(kernel_size, int)
        assert isinstance(projection_scale, int)
        assert isinstance(nbr_fc_neurons, int)
        assert isinstance(dropout, float)
        assert isinstance(use_actions, bool)
        assert isinstance(nbr_previous_action, int)
        assert weight_fname is None or isinstance(weight_fname, str)
        assert len(input_shape) > 0
        assert nbr_action > 0
        assert nbr_filters > 0
        assert nbr_bottleneck > 0
        assert nbr_pooling > 0
        assert kernel_size > 0
        assert projection_scale > 0
        assert nbr_fc_neurons > 4
        assert 0 <= dropout <= 1
        assert nbr_previous_action >= 0
        valid_pooling_cond = input_shape[0]/2**nbr_pooling
        assert valid_pooling_cond >= 1

        self.input_shape = input_shape
        self.nbr_action = nbr_action
        self.nbr_filters = nbr_filters
        self.nbr_bottleneck = nbr_bottleneck
        self.nbr_pooling = nbr_pooling
        self.kernel_size = kernel_size
        self.projection_scale = projection_scale
        self.nbr_fc_neurons = nbr_fc_neurons
        self.dropout = dropout
        self.use_actions = use_actions
        self.nbr_previous_action = nbr_previous_action
        self.weight_fname = weight_fname

        self.model = self._build_model()
        if self.weight_fname:
            self.load_weights(self.weight_fname)

    def _bottleneck(self, input_layer, nbr_filters):
        """Bottleneck module

        :param input_layer: input layer
        :param nbr_filters: nbr of filters
        :return: output layers
        """

        projected_nbr_filters = nbr_filters // self.projection_scale
        assert projected_nbr_filters > 0

        output_layer = Conv2D(projected_nbr_filters, 1, 1, 
                              border_mode='same'
                              )(input_layer)
        output_layer = Activation('relu')(output_layer)
        output_layer = Conv2D(projected_nbr_filters, self.kernel_size,
                              self.kernel_size, border_mode='same'
                              )(output_layer)
	output_layer = Activation('relu')(output_layer)
	output_layer = Conv2D(nbr_filters, 1, 1,
			      border_mode='same'
                              )(output_layer)
        current_nbr_filters = int(input_layer.get_shape()[-1])
        if current_nbr_filters == nbr_filters:
            output_layer = merge([output_layer, input_layer], mode='sum')

        return output_layer

    def _build_model(self):
        """Build the model"""

        model_input = Input(shape=self.input_shape)
	layer = model_input
        
        #layer = Conv2D(
        #    32, 8, 8, subsample = (4, 4), 
        #    init=my_init)(layer)
        #layer = Activation('relu')(layer)
        #layer = Conv2D(
        #    64, 4, 4, subsample = (2, 2),
        #    init=my_init)(layer)
	#layer = Activation('relu')(layer)
        #layer = Conv2D(
        #    64, 3, 3, subsample = (1, 1),
        #    init=my_init)(layer)
        #layer = Activation('relu')(layer)
	#layer = Flatten()(layer)
	#layer = Dense(512, init=my_init)(layer)
	#layer = Activation('relu')(layer)
	#output_layer = Dense(self.nbr_action, init=my_init)(layer)

        for idx_pooling in range(self.nbr_pooling):
            for idx_bottleneck in range(self.nbr_bottleneck):
                layer = self._bottleneck(layer,
                                         self.nbr_filters * 2 ** idx_pooling)
            pool_layer = MaxPooling2D()(layer)
            layer = Conv2D(self.nbr_filters * 2 ** idx_pooling, 2, 2,
                           subsample=(2,2))(layer)
            layer = merge([layer, pool_layer], mode='sum')
        # Dense layer
        layer = Flatten()(layer)
        layer = Dropout(self.dropout)(layer)
        layer = Dense(self.nbr_fc_neurons)(layer)
        layer = Activation('relu')(layer)

        # Add the action history if specified
        if self.use_actions:
            action_input = Input(
                shape=(self.nbr_previous_action * self.nbr_action,)
            )
            layer = merge([layer, action_input], mode='concat')
            model_input = [model_input, action_input]

        layer = Dropout(self.dropout)(layer)
        output_layer = Dense(self.nbr_action)(layer)
        model = Model(input=model_input, output=output_layer)
        model.summary(line_length=115)

        return model

    @staticmethod
    def _convolution(input_layer, kernel_size, nbr_filters):
        """Convolution with batchnormalization and non linearity

        :param input_layer: input layer
        :param kernel_size: size of the kernel
        :param nbr_filters: number of filters
        :return: output layer
        """
        
        assert kernel_size > 0
        assert nbr_filters > 0
        output_layer = Conv2D(nbr_filters, kernel_size, kernel_size,
                              border_mode='same')(input_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('relu')(output_layer)
        return output_layer

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
