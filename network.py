"""Network"""

from keras.layers import (
    Conv2D, MaxPooling2D, Input, merge, BatchNormalization,
    Activation
)

class Network:
    """Network"""

    def __init__(self, input_shape, nbr_action, nbr_filters, nbr_bottleneck, 
                 nbr_pooling, kernel_size, projection_scale, nbr_fc_neurons, use_actions=False):
        """Init"""
        
        assert isinstance(input_shape, tuple)
        assert isinstance(nbr_action, int)
        assert isinstance(nbr_filters, int)
        assert isinstance(nbr_bottleneck, int)
        assert isinstance(nbr_pooling, int)
        assert isinstance(kernel_size, int)
        assert isinstance(projection_scale, int)
	assert isinstance(nbr_fc_neurons, int)
        assert isinstance(use_actions, bool)
        assert len(input_shape) > 0
        assert nbr_action > 0
        assert nbr_filters > 0
        assert nbr_bottleneck > 0
        assert nbr_pooling > 0 
        assert kernel_size > 0
        assert projection_scale > 0
	assert nbr_fc_neurons > 4
        assert (
            input_shape[0]/2**nbr_pooling == float(input_shape[0])/float(2**nbr_pooling)
        )

        self.input_shape = input_shape
        self.nbr_action = nbr_action
        self.nbr_filters = nbr_filters
        self.nbr_bottleneck = nbr_bottleneck
        self.nbr_pooling = nbr_pooling
        self.kernel_size = kernel_size
        self.projection_scale = projection_scale
	self.nbr_fc_neurons = nbr_fc_neurons
        self.use_actions = use_actions

    @staticmethod
    def _convolution(input_layer, kernel_size, nbr_filters):
        """Convolution"""
        
        assert kernel_size > 0
        assert nbr_filters > 0
        output_layer = Conv2D(nbr_filters, kernel_size, kernel_size,
                              border_mode='same')(input_layer)
        output_layer = BatchNormalization()(output_layer)
        output_layer = Activation('relu')(output_layer)
        return output_layer
    
    @staticmethod
    def _pooling(input_layer):
        """Pooling"""
        
        output_layer = MaxPooling2D(pool_size=(2, 2))(input_layer)
        return output_layer

    def _bottleneck(self, input_layer, nbr_filters):
        """Bottleneck"""
        
        projected_nbr_filters = nbr_filters // self.projection_scale
        assert projected_nbr_filters > 0

        output_layer = self._convolution(input_layer, 1, projected_nbr_filters)
        output_layer = self._convolution(output_layer, self.kernel_size, projected_nbr_filters)
        output_layer = self._convolution(output_layer, 1, nbr_filters)

        current_nbr_filters = int(input_layer.get_shape()[-1])
	if current_nbr_filters = nbr_filters:
	    output_layer = merge([output_layer, input_layer], mode='sum')

        return output_layer

    def build(self):
        """Build the network"""

        input_layer = Input(shape=self.input_shape)

        layer = self._convolution(input_layer, self.kernel_size,
                                  self.nbr_filters)
        for idx_pooling in range(self.nbr_pooling):
            for idx_bottleneck in range(self.nbr_bottleneck):
                layer = self._bottleneck(layer,
                                         self.nbr_filters*2**idx_pooling)
            layer = self._pooling(layer)

	:
