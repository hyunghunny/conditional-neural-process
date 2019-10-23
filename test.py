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

from data_gen import GPCurvesReader
from model import *


def meta_gp_test(x_dim=1,
                min_x=-2.,
                max_x=2.,
                lr=1e-4,

                batch_size=64,
                test_targets=500,
                context_points=10,

                train_iters=int(1.5e5),
                eval_after=int(1e4)):
    
    tf.reset_default_graph()
    # Train dataset
    dataset_train = GPCurvesReader(x_size=x_dim,
                                    batch_size=batch_size, 
                                    num_context=context_points)
    data_train = dataset_train.generate_curves(min_x=min_x, max_x=max_x)

    # Test dataset
    dataset_test = GPCurvesReader(x_size=x_dim,
                                batch_size=1, 
                                num_context=context_points, 
                                testing=True)
    data_test = dataset_test.generate_curves(num_target=test_targets, min_x=min_x, max_x=max_x)
    
    g = build_graph(data_train, 
                    data_test,
                    [128, 128, 128, 128], 
                    [128, 128, 2],
                    lr=lr
        )
    log_dir = "d{}r{:.0f}_b{}_t{}_c{}_i{}_lr{}".format(x_dim, (max_x - min_x), batch_size, test_targets, context_points, train_iters, lr)
    target_y, pred_y, pred_var = train(g, data_test, train_iters, eval_after, log_dir)
    target = target_y[0,:,0]
    pred = pred_y[0,:,0]
    var = pred_var[0,:,0]
    rmse = np.sqrt(mean_squared_error(target, pred))
    mae = mean_absolute_error(target, pred)
    r2 = r2_score(target, pred)
    print("RMSE: {:.5f}, MAP:{:.5f}, R^2: {:.5f}".format(rmse, mae, r2))
    target_max_index = np.argmax(target)
    pred_max_index = np.argmax(pred)
    print("Real: {}, Estimate: {}".format(target_max_index, pred_max_index))
    print("Real optima: {:.5f}, estimated: {:.5f}({:.5f})".format(
        target[target_max_index], pred[target_max_index], var[target_max_index]))
    print("Estimated optima: {:.5f} ({:.5f}), real: {:.5f}".format(
        pred[pred_max_index], var[pred_max_index], target[pred_max_index]))

if __name__ == "__main__":
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id', type=int, default=0, help='GPU ID to be fully assigned')
    parser.add_argument('-x_dim', type=int, default=1, help='GP input dimension')
    parser.add_argument('-min_x', type=float, default=-1., help='GP x min value')
    parser.add_argument('-max_x', type=float, default=1., help='GP x max value')
    parser.add_argument('-lr', type=float, default=5e-4, help='learning rate') 
    parser.add_argument('-batch_size', type=int, default=64, help='meta-train batch size') 
    parser.add_argument('-test_targets', type=int, default=1000, help='number of test targets') 
    parser.add_argument('-context_points', type=int, default=10, help='number of support set') 
    parser.add_argument('-train_iters', type=int, default=int(1e5), help='number of iterations in meta-training phase') 
    parser.add_argument('-eval_after', type=int, default=int(1e4), help='the validation steps during meta-training') 
    
    args = vars(parser.parse_args())
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])
    del args['gpu_id']    
    print("condition: {}".format(args))
    meta_gp_test(**args)
    print("Elapsed time of meta-learning: {:.0f}".format(time.time() - start))