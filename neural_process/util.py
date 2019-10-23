"""Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

"""

import tensorflow as tf

# utility methods
def batch_mlp(input, output_sizes, variable_scope):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).
    
    Args:
        input: input tensor of shape [B,n,d_in].
        output_sizes: An iterable containing the output sizes of the MLP as defined 
                in `basic.Linear`.
        variable_scope: String giving the name of the variable scope. If this is set
                to be the same as a previously defined MLP, then the weights are reused.
        
    Returns:
        tensor of shape [B,n,d_out] where d_out=output_sizes[-1]
    """
    # Get the shapes of the input and reshape to parallelise across observations
    batch_size, _, filter_size = input.shape.as_list()
    output = tf.reshape(input, (-1, filter_size))
    output.set_shape((None, filter_size))

    # Pass through MLP
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        for i, size in enumerate(output_sizes[:-1]):
            output = tf.nn.relu(
                    tf.layers.dense(output, size, name="layer_{}".format(i)))

        # Last layer without a ReLu
        output = tf.layers.dense(
                output, output_sizes[-1], name="layer_{}".format(i + 1))

    # Bring back into original shape
    output = tf.reshape(output, (batch_size, -1, output_sizes[-1]))
    return output
    