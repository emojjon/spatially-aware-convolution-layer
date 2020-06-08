import tensorflow as tf
import numpy as np

class SAConvolution2D(tf.keras.layers.Layer): # Spatially Aware Convolution Layer

    def __init__(self, filters, kernel_size, *args, **kwargs):
        trainable = kwargs['trainable'] if 'trainable' in kwargs else True
        name = kwargs['name'] if 'name' in kwargs else None
        dtype = kwargs['dtype'] if 'dtype' in kwargs else None
        dynamic = kwargs['dynamic'] if 'dynamic' in kwargs else False
        
        allowed = {'input_shape', 'batch_input_shape', 'batch_size', 'weights', 'activity_regularizer', 'autocast'}
        superkwargs = {k:kwargs[k] for k in set(kwargs).intersection(allowed)}
        super(SAConvolution2D, self).__init__(trainable, name, dtype, dynamic, **superkwargs)
        self.layers = []
        self.filters = filters
        self.kernel_size = kernel_size
        if 'input_shape' in kwargs:
            input_shape = kwargs.pop('input_shape')
            self.build(input_shape, *args, **kwargs)
        else:
            self.args=args
            self.kwargs=kwargs

    def build(self, input_shape, *args, **kwargs):
        self.x = tf.linspace(-1.0, 1.0, num=input_shape[-3])[:,np.newaxis,np.newaxis]*tf.ones(input_shape[-3:-1]+(1,))
        self.y = tf.linspace(-1.0, 1.0, num=input_shape[-2])[np.newaxis,:,np.newaxis]*tf.ones(input_shape[-3:-1]+(1,))
        if 'kwargs' in dir(self):
            kwargs = {**self.kwargs, **kwargs}
        if 'args' in dir(self):
            if len(self.args) > len(args):
                args += self.args[len(args):]        
        kwargs['input_shape'] = input_shape[-3:-1] + (input_shape[-1] + 2,)
        kwargs['name'] = '{}/convolute2d'.format(self.name)
        self.layers = [
                        tf.keras.layers.Concatenate(input_shape=input_shape[-3:-1]+(None,), name = '{}/concatenate'.format(self.name)),
                        tf.keras.layers.Conv2D(self.filters, self.kernel_size,  *args, **kwargs)
                      ]

    def call(self, inputs):
        return self.layers[1](self.layers[0]([inputs, tf.ones_like(inputs[:,:,:,0:1])*self.x, tf.ones_like(inputs[:,:,:,0:1])*self.y]))

