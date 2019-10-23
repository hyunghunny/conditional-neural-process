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


class Decoder(object):
    """The Decoder."""

    def __init__(self, output_sizes):
        """(A)NP decoder.

        Args:
            output_sizes: An iterable containing the output sizes of the decoder MLP 
                    as defined in `basic.Linear`.
        """
        self._output_sizes = output_sizes

    def __call__(self, representation, target_x):
        """Decodes the individual targets.

        Args:
            representation: The representation of the context for target predictions. 
                    Tensor of shape [B,target_observations,?].
            target_x: The x locations for the target query.
                    Tensor of shape [B,target_observations,d_x].

        Returns:
            dist: A multivariate Gaussian over the target points. A distribution over
                    tensors of shape [B,target_observations,d_y].
            mu: The mean of the multivariate Gaussian.
                    Tensor of shape [B,target_observations,d_x].
            sigma: The standard deviation of the multivariate Gaussian.
                    Tensor of shape [B,target_observations,d_x].
        """
        # concatenate target_x and representation
        hidden = tf.concat([representation, target_x], axis=-1)
        
        # Pass final axis through MLP
        hidden = batch_mlp(hidden, self._output_sizes, "decoder")

        # Get the mean an the variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * tf.nn.softplus(log_sigma)

        # Get the distribution
        dist = tf.contrib.distributions.MultivariateNormalDiag(
                loc=mu, scale_diag=sigma)

        return dist, mu, sigma