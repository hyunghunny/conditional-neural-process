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
from encoder import *
from decoder import *
import collections

# The CNP takes as input a `CNPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tesor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points"))


class DeterministicModel(object):
    """The CNP model."""

    def __init__(self, encoder_output_sizes, decoder_output_sizes):
        """Initialises the model.

        Args:
            encoder_output_sizes: An iterable containing the sizes of hidden layers of
                    the encoder. The last one is the size of the representation r.
            decoder_output_sizes: An iterable containing the sizes of hidden layers of
                    the decoder. The last element should correspond to the dimension of
                    the y * 2 (it encodes both mean and variance concatenated)
        """
        self._encoder = DeterministicEncoder(encoder_output_sizes)
        self._decoder = DeterministicDecoder(decoder_output_sizes)

    def __call__(self, query, num_total_points, num_contexts, target_y=None):
        """Returns the predicted mean and variance at the target points.

        Args:
            query: Array containing ((context_x, context_y), target_x) where:
                    context_x: Array of shape batch_size x num_context x 1 contains the 
                            x values of the context points.
                    context_y: Array of shape batch_size x num_context x 1 contains the 
                            y values of the context points.
                    target_x: Array of shape batch_size x num_target x 1 contains the
                            x values of the target points.
            target_y: The ground truth y values of the target y. An array of 
                    shape batchsize x num_targets x 1.
            num_total_points: Number of target points.

        Returns:
            log_p: The log_probability of the target_y given the predicted
            distribution.
            mu: The mean of the predicted distribution.
            sigma: The variance of the predicted distribution.
        """

        (context_x, context_y), target_x = query

        # Pass query through the encoder and the decoder
        representation = self._encoder(context_x, context_y, num_contexts)
        dist, mu, sigma = self._decoder(representation, target_x, num_total_points)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None
        if target_y is not None:
            log_p = dist.log_prob(target_y)
        else:
            log_p = None

        return log_p, mu, sigma

