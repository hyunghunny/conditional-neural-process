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
import time
import os
import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr as smr

from data.gp_gen import GPCurvesReader
from build import *
from neural_process.cnp import CNPModel
from neural_process.np import LatentModel
from neural_process.attention import *


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


def compute_lcb(func_m, func_v, epsilon=0.0001):
    
    func_s = np.sqrt(func_v) + epsilon
    ucb = func_m - func_s

    return ucb 


def rank_corr(group1, group2):
    corr = smr(group1, group2).correlation
    return corr


def evaluate(target_y, pred_y, pred_var):
    target = target_y[0,:,0]
    mean = pred_y[0,:,0]
    var = pred_var[0,:,0]
    
    pred = compute_lcb(mean, var) # use of UCB

    rmse = np.sqrt(mean_squared_error(target, pred))
    mae = mean_absolute_error(target, pred)
    r2 = r2_score(target, pred)
    print("RMSE: {:.5f}, MAP:{:.5f}, R^2: {:.5f}".format(rmse, mae, r2))
    print("Spearman rank correlation: {:.5f}".format(rank_corr(target, pred)))
    target_max_index = np.argmax(target)
    pred_max_index = np.argmax(pred)
    print("Real: {}, Estimate: {}".format(target_max_index, pred_max_index))
    print("Real optima: {:.5f}, estimated: {:.5f}({:.5f})".format(
        target[target_max_index], mean[target_max_index], var[target_max_index]))
    print("Estimated optima: {:.5f} ({:.5f}), real: {:.5f}".format(
        mean[pred_max_index], var[pred_max_index], target[pred_max_index]))


def main(x_dim=1,
        min_x=-2.,
        max_x=2.,

        lr=1e-4,

        batch_size=64,
        test_targets=500,
        context_points=10,
        model_type='CNP',
        attention_type='multihead',
        hidden_size=128,
        train_iters=int(1.5e5),
        eval_after=int(1e4)):

    
    tf.reset_default_graph()
    random_kernel_parameters = True

    # Train dataset
    dataset_train = GPCurvesReader(x_size=x_dim,
                                    batch_size=batch_size,
                                    random_kernel_parameters=random_kernel_parameters, 
                                    num_context=context_points)
    data_train = dataset_train.generate_curves(min_x=min_x, max_x=max_x)

    # Test dataset
    dataset_test = GPCurvesReader(x_size=x_dim,
                                batch_size=1,
                                random_kernel_parameters=random_kernel_parameters, 
                                num_context=context_points, 
                                testing=True)
    data_test = dataset_test.generate_curves(num_target=test_targets, min_x=min_x, max_x=max_x)
    
    use_deterministic_path = True
    latent_encoder_output_sizes = [hidden_size] * 4
    deterministic_encoder_output_sizes = [hidden_size] * 4
    decoder_output_sizes = [hidden_size]*2 + [2]
    
    model_type = model_type.upper()
    if model_type == 'CNP':
        # Sizes of the layers of the MLPs for the encoder and decoder
        m = CNPModel(deterministic_encoder_output_sizes,
                     decoder_output_sizes)

    elif model_type == 'ANP':
        attention = Attention(rep='mlp', 
                            output_sizes=[hidden_size]*2, 
                            att_type=attention_type)
        # Define the model
        m = LatentModel(latent_encoder_output_sizes, hidden_size,
                        decoder_output_sizes, use_deterministic_path, 
                        deterministic_encoder_output_sizes, attention)
        model_type = "{}-{}".format(model_type, attention_type)                                                     
    
    # NP - equivalent to uniform attention
    elif model_type == 'NP':
        attention = Attention(rep='identity', output_sizes=None, att_type='uniform')
        # Define the model
        m = LatentModel(latent_encoder_output_sizes, hidden_size,
                        decoder_output_sizes, use_deterministic_path, 
                        deterministic_encoder_output_sizes, attention)
    else:
        raise NameError("Invalid model type: {}". format(model_type))

    g = build_graph(m, data_train, data_test,
                    lr=lr)
    log_dir = "{}{}_d{}r{:.0f}_b{}_t{}_c{}_i{}_lr{}".format(
        model_type, hidden_size,
        x_dim, (max_x - min_x), batch_size,
        test_targets, context_points, train_iters, lr)
    target_y, pred_y, pred_var = train(g, data_test, train_iters, eval_after, log_dir)
    evaluate(target_y, pred_y, pred_var)


if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', type=int, default=0, help='GPU ID to be fully assigned')
    
    parser.add_argument('-x_dim', type=int, default=10, help='GP input dimension')
    parser.add_argument('-min_x', type=float, default=-1., help='GP x min value')
    parser.add_argument('-max_x', type=float, default=1., help='GP x max value')

    parser.add_argument('-model_type', type=str, default='CNP', help='Neural Process Type. CNP, NP, ANP available') 
    parser.add_argument('-attention_type', type=str, default='multihead', help='Attention type. [uniform, laplace, dot_product, multihead] supported') 
    parser.add_argument('-hidden_size', type=int, default=128, help='size of neurons in a hidden layer') 

    parser.add_argument('-lr', type=float, default=1e-3, help='learning rate') 
    parser.add_argument('-batch_size', type=int, default=64, help='meta-train batch size') 
    
    parser.add_argument('-test_targets', type=int, default=10000, help='number of test targets') 
    parser.add_argument('-context_points', type=int, default=200, help='number of support set') 
    
    parser.add_argument('-train_iters', type=int, default=int(3e5), help='number of iterations in meta-training phase') 
    parser.add_argument('-eval_after', type=int, default=int(1e4), help='the validation steps during meta-training') 
    
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])
    del args['gpu_id']    
    print("condition: {}".format(args))
    main(**args)
    print("Elapsed time of meta-learning: {:.0f}".format(time.time() - start))