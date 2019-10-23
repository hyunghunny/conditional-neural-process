"""
Copyright 2018 Google LLC

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


class DeterministicEncoder(object):
    """The Encoder."""

    def __init__(self, output_sizes):
        """CNP encoder.

        Args:
            output_sizes: An iterable containing the output sizes of the encoding MLP.
        """
        self._output_sizes = output_sizes

    def __call__(self, context_x, context_y, num_context_points):
        """Encodes the inputs into one representation.

        Args:
            context_x: Tensor of size bs x observations x m_ch. For this 1D regression
                    task this corresponds to the x-values.
            context_y: Tensor of size bs x observations x d_ch. For this 1D regression
                    task this corresponds to the y-values.
            num_context_points: A tensor containing a single scalar that indicates the
                    number of context_points provided in this iteration.

        Returns:
            representation: The encoded representation averaged over all context 
                    points.
        """

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = encoder_input.shape.as_list()
        hidden = tf.reshape(encoder_input, (batch_size * num_context_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                        tf.layers.dense(hidden, size, name="Encoder_layer_{}".format(i)))

            # Last layer without a ReLu
            hidden = tf.layers.dense(
                    hidden, self._output_sizes[-1], name="Encoder_layer_{}".format(i + 1))

        # Bring back into original shape
        hidden = tf.reshape(hidden, (batch_size, num_context_points, size))

        # Aggregator: take the mean over all points
        representation = tf.reduce_mean(hidden, axis=1)

        return representation

