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
from neural_process.util import batch_mlp

class DeterministicEncoder(object):
    """The Deterministic Encoder."""

    def __init__(self, output_sizes, attention):
        """(A)NP deterministic encoder.

        Args:
            output_sizes: An iterable containing the output sizes of the encoding MLP.
            attention: The attention module.
        """
        self._output_sizes = output_sizes
        self._attention = attention

    def __call__(self, context_x, context_y, target_x):
        """Encodes the inputs into one representation.

        Args:
            context_x: Tensor of shape [B,observations,d_x]. For this 1D regression
                    task this corresponds to the x-values.
            context_y: Tensor of shape [B,observations,d_y]. For this 1D regression
                    task this corresponds to the y-values.
            target_x: Tensor of shape [B,target_observations,d_x]. 
                    For this 1D regression task this corresponds to the x-values.

        Returns:
            The encoded representation. Tensor of shape [B,target_observations,d]
        """

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([context_x, context_y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes, 
                                             "deterministic_encoder")

        # Apply attention
        with tf.variable_scope("deterministic_encoder", reuse=tf.AUTO_REUSE):
                hidden = self._attention(context_x, target_x, hidden)

        return hidden


class LatentEncoder(object):
    """The Latent Encoder."""

    def __init__(self, output_sizes, num_latents):
        """(A)NP latent encoder.

        Args:
            output_sizes: An iterable containing the output sizes of the encoding MLP.
            num_latents: The latent dimensionality.
        """
        self._output_sizes = output_sizes
        self._num_latents = num_latents

    def __call__(self, x, y):
        """Encodes the inputs into one representation.

        Args:
            x: Tensor of shape [B,observations,d_x]. For this 1D regression
                    task this corresponds to the x-values.
            y: Tensor of shape [B,observations,d_y]. For this 1D regression
                    task this corresponds to the y-values.

        Returns:
            A normal distribution over tensors of shape [B, num_latents]
        """

        # Concatenate x and y along the filter axes
        encoder_input = tf.concat([x, y], axis=-1)

        # Pass final axis through MLP
        hidden = batch_mlp(encoder_input, self._output_sizes, "latent_encoder")
            
        # Aggregator: take the mean over all points
        hidden = tf.reduce_mean(hidden, axis=1)
        
        # Have further MLP layers that map to the parameters of the Gaussian latent
        with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
            # First apply intermediate relu layer 
            hidden = tf.nn.relu(
                    tf.layers.dense(hidden, 
                                                    (self._output_sizes[-1] + self._num_latents)/2, 
                                                    name="penultimate_layer"))
            # Then apply further linear layers to output latent mu and log sigma
            mu = tf.layers.dense(hidden, self._num_latents, name="mean_layer")
            log_sigma = tf.layers.dense(hidden, self._num_latents, name="std_layer")
            
        # Compute sigma
        sigma = 0.1 + 0.9 * tf.sigmoid(log_sigma)

        return tf.contrib.distributions.Normal(loc=mu, scale=sigma)
