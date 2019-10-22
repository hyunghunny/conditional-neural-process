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
from cnp import DeterministicModel


def build_graph(data_train, data_test, 
                encoder_output_sizes=[128, 128, 128, 128], 
                decoder_output_sizes=[128, 128, 2],
                lr=1e-4):
    # Sizes of the layers of the MLPs for the encoder and decoder

    # Define the model
    model = DeterministicModel(encoder_output_sizes, decoder_output_sizes)

    # Define the loss
    log_prob, _, _ = model(data_train.query, data_train.num_total_points,
                            data_train.num_context_points, data_train.target_y)
    loss = -tf.reduce_mean(log_prob)

    # Get the predicted mean and variance at the target points for the testing set
    _, mu, sigma = model(data_test.query, data_test.num_total_points,
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


def train(graph, data_test, max_iters, validate_after, log_dir):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./logs/{}'.format(log_dir), sess.graph)
        sess.run(init)        
        for it in range(max_iters):
            train_loss, train_summary = sess.run([graph['train_step'], graph['summary']['train']])
            writer.add_summary(train_summary, it)
            if it % validate_after == 0:
                loss_value, pred_y, pred_var, target_y, whole_query, test_summary = sess.run(
                    [ graph['loss'], graph['mu'], graph['sigma'], 
                     data_test.target_y, data_test.query,
                     graph['summary']['test'] ]
                )
                writer.add_summary(test_summary, it)
                (context_x, context_y), target_x = whole_query
                print('Iteration: {}, test loss: {:.5f}'.format(it, loss_value))

        return (target_y, pred_y, pred_var)