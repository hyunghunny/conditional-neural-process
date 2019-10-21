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
import tensorflow_probability as tfp


class DeterministicDecoder(object):
    """The Decoder."""

    def __init__(self, output_sizes):
        """CNP decoder.

        Args:
            output_sizes: An iterable containing the output sizes of the decoder MLP 
                    as defined in `basic.Linear`.
        """
        self._output_sizes = output_sizes

    def __call__(self, representation, target_x, num_total_points):
        """Decodes the individual targets.

        Args:
            representation: The encoded representation of the context
            target_x: The x locations for the target query
            num_total_points: The number of target points.

        Returns:
            dist: A multivariate Gaussian over the target points.
            mu: The mean of the multivariate Gaussian.
            sigma: The standard deviation of the multivariate Gaussian.
        """

        # Concatenate the representation and the target_x
        representation = tf.tile(
                tf.expand_dims(representation, axis=1), [1, num_total_points, 1])
        input = tf.concat([representation, target_x], axis=-1)

        # Get the shapes of the input and reshape to parallelise across observations
        batch_size, _, filter_size = input.shape.as_list()
        hidden = tf.reshape(input, (batch_size * num_total_points, -1))
        hidden.set_shape((None, filter_size))

        # Pass through MLP
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            for i, size in enumerate(self._output_sizes[:-1]):
                hidden = tf.nn.relu(
                        tf.layers.dense(hidden, size, name="Decoder_layer_{}".format(i)))

            # Last layer without a ReLu
            hidden = tf.layers.dense(
                    hidden, self._output_sizes[-1], name="Decoder_layer_{}".format(i + 1))

        # Bring back into original shape
        hidden = tf.reshape(hidden, (batch_size, num_total_points, -1))

        # Get the mean an the variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        # Get the distribution
        dist = tfp.distributions.MultivariateNormalDiag(
                loc=mu, scale_diag=sigma)

        return dist, mu, sigma


