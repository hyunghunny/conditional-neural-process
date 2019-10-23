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
from neural_process.np_encoder import *
from neural_process.np_decoder import *


class LatentModel(object):
    """The (A)NP model."""

    def __init__(self, latent_encoder_output_sizes, num_latents,
                decoder_output_sizes, use_deterministic_path=True, 
                deterministic_encoder_output_sizes=None, attention=None):
        """Initialises the model.

        Args:
            latent_encoder_output_sizes: An iterable containing the sizes of hidden 
                    layers of the latent encoder.
            num_latents: The latent dimensionality.
            decoder_output_sizes: An iterable containing the sizes of hidden layers of
                    the decoder. The last element should correspond to d_y * 2
                    (it encodes both mean and variance concatenated)
            use_deterministic_path: a boolean that indicates whether the deterministic
                    encoder is used or not.
            deterministic_encoder_output_sizes: An iterable containing the sizes of 
                    hidden layers of the deterministic encoder. The last one is the size 
                    of the deterministic representation r.
            attention: The attention module used in the deterministic encoder.
                    Only relevant when use_deterministic_path=True.
        """
        self._latent_encoder = LatentEncoder(latent_encoder_output_sizes, 
                                                                                 num_latents)
        self._decoder = Decoder(decoder_output_sizes)
        self._use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self._deterministic_encoder = DeterministicEncoder(
                    deterministic_encoder_output_sizes, attention)
        

    def __call__(self, query, num_targets, num_contexts, target_y=None):
        """Returns the predicted mean and variance at the target points.

        Args:
            query: Array containing ((context_x, context_y), target_x) where:
                    context_x: Tensor of shape [B,num_contexts,d_x]. 
                            Contains the x values of the context points.
                    context_y: Tensor of shape [B,num_contexts,d_y]. 
                            Contains the y values of the context points.
                    target_x: Tensor of shape [B,num_targets,d_x]. 
                            Contains the x values of the target points.
            num_targets: Number of target points.
            num_contexts: Number of context points. [XXX:unused value]
            target_y: The ground truth y values of the target y. 
                    Tensor of shape [B,num_targets,d_y].

        Returns:
            log_p: The log_probability of the target_y given the predicted
                    distribution. Tensor of shape [B,num_targets].
            mu: The mean of the predicted distribution. 
                    Tensor of shape [B,num_targets,d_y].
            sigma: The variance of the predicted distribution.
                    Tensor of shape [B,num_targets,d_y].
        """

        (context_x, context_y), target_x = query

        # Pass query through the encoder and the decoder
        prior = self._latent_encoder(context_x, context_y)
        
        # For training, when target_y is available, use targets for latent encoder.
        # Note that targets contain contexts by design.
        if target_y is None:
            latent_rep = prior.sample()
        # For testing, when target_y unavailable, use contexts for latent encoder.
        else:
            posterior = self._latent_encoder(target_x, target_y)
            latent_rep = posterior.sample()
        latent_rep = tf.tile(tf.expand_dims(latent_rep, axis=1),
                                                 [1, num_targets, 1])
        if self._use_deterministic_path:
            deterministic_rep = self._deterministic_encoder(context_x, context_y,
                                                                                                            target_x)
            representation = tf.concat([deterministic_rep, latent_rep], axis=-1)
        else:
            representation = latent_rep
            
        dist, mu, sigma = self._decoder(representation, target_x)
        
        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            posterior = self._latent_encoder(target_x, target_y)
            kl = tf.reduce_sum(
                    tf.contrib.distributions.kl_divergence(posterior, prior), 
                    axis=-1, keepdims=True)
            kl = tf.tile(kl, [1, num_targets])
            loss = - tf.reduce_mean(log_p - kl / tf.cast(num_targets, tf.float32))
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss
