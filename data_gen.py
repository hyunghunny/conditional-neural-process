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
import numpy as np
from cnp import CNPRegressionDescription


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """

    def __init__(self,
                 batch_size,
                 num_context,
                 x_size=1,
                 y_size=1,
                 l1_scale=0.4,
                 sigma_scale=1.0,
                 testing=False):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
          batch_size: An integer.
          num_context: The number of observations in the context.
          x_size: Integer >= 1 for length of "x values" vector.
          y_size: Integer >= 1 for length of "y values" vector.
          l1_scale: Float; typical scale for kernel distance function.
          sigma_scale: Float; typical scale for variance.
          testing: Boolean that indicates whether we are testing. If so there are
              more targets for visualization.
        """
        self._batch_size = batch_size
        self._num_context = num_context
        self._x_size = x_size
        self._y_size = y_size
        self._l1_scale = l1_scale
        self._sigma_scale = sigma_scale
        self._testing = testing

    def _gaussian_kernel(self, xdata, l1, sigma_f, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data.

        Args:
          xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
              the values of the x-axis data.
          l1: Tensor with shape `[batch_size, y_size, x_size]`, the scale
              parameter of the Gaussian kernel.
          sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
              of the std.
          sigma_noise: Float, std of the noise that we add for stability.

        Returns:
          The kernel, a float tensor with shape
          `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        num_total_points = tf.shape(xdata)[1]

        # Expand and take the difference
        # [B, 1, num_total_points, x_size]
        xdata1 = tf.expand_dims(xdata, axis=1)
        # [B, num_total_points, 1, x_size]
        xdata2 = tf.expand_dims(xdata, axis=2)
        # [B, num_total_points, num_total_points, x_size]
        diff = xdata1 - xdata2

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = tf.square(diff[:, None, :, :, :] / l1[:, :, None, None, :])

        norm = tf.reduce_sum(
            norm, -1)  # [B, data_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = tf.square(sigma_f)[:, :, None, None] * tf.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * tf.eye(num_total_points)

        return kernel

    def generate_curves(self, num_target=10, min_x=-2., max_x=2.):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between min_x and max_x.

        Returns:
          A `CNPRegressionDescription` namedtuple.
        """
        #num_context = tf.random_uniform(
        #    shape=[], minval=3, maxval=self._num_context, dtype=tf.int32)
        num_context = self._num_context
        
        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_total_points = num_target

        # During training the number of target points and their x-positions are
        # selected at random
        else:
            #num_target = tf.random_uniform(
            #    shape=(), minval=2, maxval=self._max_num_context, dtype=tf.int32)
            #num_target = self._num_context
            num_total_points = num_context + num_target
        x_values = tf.random.uniform(
            [self._batch_size, num_total_points, self._x_size], min_x, max_x)

        # Set kernel parameters
        l1 = (tf.ones(shape=[self._batch_size,
                             self._y_size, self._x_size]) * self._l1_scale)
        sigma_f = tf.ones(
            shape=[self._batch_size, self._y_size]) * self._sigma_scale

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        kernel = self._gaussian_kernel(x_values, l1, sigma_f)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = tf.cast(tf.cholesky(
            tf.cast(kernel, tf.float64)), tf.float32)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = tf.matmul(
            cholesky,
            tf.random_normal([self._batch_size, self._y_size, num_total_points, 1]))

        # [batch_size, num_total_points, y_size]
        y_values = tf.transpose(tf.squeeze(y_values, 3), [0, 2, 1])

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = tf.random_shuffle(tf.range(num_target))
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)

        else:
            # Select the targets which will consist of the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]

            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return CNPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=tf.shape(target_x)[1],
            num_context_points=num_context)
