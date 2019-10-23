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


def build_graph(model, data_train, data_test, 
                lr=1e-4):

    # Define the loss
    _, _, log_prob, _, loss = model(data_train.query, 
                                    data_train.num_total_points,
                                    data_train.num_context_points, 
                                    data_train.target_y)

    # Get the predicted mean and variance at the target points for the testing set
    mu, sigma, _, _, _ = model(data_test.query, 
                               data_test.num_total_points,
                               data_test.num_context_points)

    # Set up the optimizer and train step
    optimizer = tf.train.AdamOptimizer(lr)
    train_step = optimizer.minimize(loss)

    train_summary = tf.summary.scalar('train_loss', loss)
    test_summary = tf.summary.scalar('test_loss', loss)

    # The final output layer of the decoder outputs two values, one for the mean and
    # one for the variance of the prediction at the target location
    return {
        "train_step": train_step, 
        "loss": loss, 
        "mu": mu, 
        "sigma" : sigma,
        "summary" : {
            "train": train_summary,
            "test" : test_summary
        }
    }

