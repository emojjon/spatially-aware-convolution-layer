import tensorflow as tf
import numpy as np
import re

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
            self.kwargs.update(kwargs)
        else:
            self.kwargs = kwargs
        if 'args' in dir(self):
            if len(self.args) > len(args):
                args += self.args[len(args):]
        self.args = args
        convkwargs = self.kwargs.copy()
        kwargs['input_shape'] = input_shape
        convkwargs = self.kwargs.copy()
        convkwargs['input_shape'] = input_shape[-3:-1] + (input_shape[-1] + 2,)
        convkwargs['name'] = '{}/conv2d'.format(self.name)
 
        self.layers = [
                        tf.keras.layers.Concatenate(input_shape=input_shape[-3:-1]+(None,), name = '{}/concatenate'.format(self.name)),
                        tf.keras.layers.Conv2D(self.filters, self.kernel_size,  *args, **kwargs)
                      ]
        self.layers[1].build((None,) + convkwargs['input_shape']) # needed so that weights can be set right after building self

    def call(self, inputs):
        return self.layers[1](self.layers[0]([inputs, tf.ones_like(inputs[:,:,:,0:1])*self.x, tf.ones_like(inputs[:,:,:,0:1])*self.y]))

    def get_config(self):
        config = self.kwargs.copy()
        config['filters'] = self.filters
        config['kernel_size'] = self.kernel_size
        config['args'] = self.args
        return config
    
    @classmethod
    def from_config(cls, config):
        filters = config.pop('filters')
        kernel_size = config.pop('kernel_size')
        if 'args' in config:
            args = config.pop('args')
        else:
            args = ()
        return cls(filters, kernel_size, *args, **config)

def inject_saconv2d(model, layer_list):
# Method for building new models from old ones by replacing an optional number of
# Conv2D layers with spatially aware counterparts. Because of python binding
# semantics it is possible that the old model should not be used after calling
# this function. If you need to you could save it and load it anew after passing
# it to this function. Do try though!

    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Build a dictionary of layer (name) -> list of layers (names) that pushes 
    # data to it (stored in network_dict['input_layers_of'])
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for lnr in range(1, len(model.layers)):
        layer = model.layers[lnr]

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        hit = False
        for item in layer_list:
            if isinstance(item, int):
                if lnr == item:
                    hit = True  # Replace layer if number matches
            else:
                if re.match(item, layer.name):
                    hit = True  # Replace layer if name matches the regular expression

                  
        if hit:
            if not isinstance(layer, tf.keras.layers.Conv2D):
                raise ValueError('Layer {} "{}" is not a Conv2D layer and can not be replaced by a SAConvolution2D layer.'.format(lnr, layer.name))
            
            # Collect some information about the layer to be replaced
            lconfig = layer.get_config()
            lweights = layer.get_weights()
            lshape = lweights[0].shape
            if not 'input_shape' in lconfig: # The layer won't be built until it knows its input shape
                lconfig['input_shape'] = layer_input.shape[1:]
            linit = tf.keras.initializers.__dict__[lconfig['kernel_initializer']['class_name']](**lconfig['kernel_initializer']['config'])

            # Build new weights from old weights and initializer
            new_weights = np.empty(lshape[0:2] + (lshape[2] + 2,) + (lshape[3],)) 
            new_weights[:,:,:-2,:] = lweights[0]
            new_weights[:,:,-2:,:] = linit(lshape[0:2]+(2,)+(lshape[3],))
            lweights[0] = new_weights
            
            # Build replacement layer using the old configuration and the new weights
            new_layer = SAConvolution2D.from_config(lconfig)
            new_layer.layers[1].set_weights(lweights)

            x = new_layer(layer_input)
 
        else:
            x = layer(layer_input)

        # Set new output tensor for this layer, we populate the output dictionary as we go
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer_name in model.output_names:
            model_outputs.append(x)
            
    # return a new model with the SAME INPUTS but (maybe) OTHER OUTPUTS
    return tf.keras.models.Model(inputs=model.inputs, outputs=model_outputs) 
