#!/usr/bin/env python
# coding: utf-8
import os
num_threads = "1"
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads 
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads
import shutil
import time
import numpy as np
from decimal import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
from utils import *
from gradient import *
from data import *
from surrogate import *
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numpy import linalg as LA
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import pickle
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp
import json
import gc
import csv

def main(problem_selection):
    '''
    The main program to run the 24 BBOB functions simultansoulsy under each dimension.
    '''
    dimension = 2
    
    # Set the training dataset and test dataset for the origianl GPR (1,000 * d for training, 1,000 * d for test)
    number_of_data_GPR_training = 1000 * dimension
    number_of_data_GPR_test = 1000 * dimension

    # Set the transfer data samples and the dataset for evaluaitng the transferred GPR
    # Here, we give an example on using 100 transfer data, and use 1,000 * d for evaluating.
    number_of_data_TL_training = 100
    max_number_of_data_TL_training = 50 * dimension
    number_of_data_TL_test = 1000 * dimension
    
    # Specify the path of the Result folder
    result_path = 'Result1'
    os.makedirs(result_path, exist_ok=True)

    # We save the experimental results in the log files for further analysis.
    save_original_model = result_path + '/' + str(dimension) + '/' + f"{problem_selection}" + '/original_model'
    save_path = result_path + '/' + str(dimension) + '/' + f"{problem_selection}" + '/' + str(number_of_data_TL_training)
    save_AutoML_RSGD = save_path + '/transfer_model'
    save_without_transfer = save_path + '/without_transfer'
    
    base_path = save_AutoML_RSGD + '/'
    save_loss = save_path + '/loss_value/'
    
    isExist = os.path.exists(save_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_path, exist_ok=True)
        print("The new directory is created!")

    # Set the initial parameter of the transfer learning apraoch before hyper-parameter tuning.
    epochs = 60
    alpha = 0.001
    decay_rate = 0.1
    scheduler_type = "exponential"
    stopping_epoch_threshold = 10
    batch_size = int(number_of_data_TL_training / 5)
    
    # All the experiments are run with 10 repetitions.
    number_of_repetition = 10

    # We save the results in the log file for further analysis.
    complete_name = os.path.join(save_path, f"{problem_selection}_output"'.log')
    logger = Logger(complete_name)
    
    print("This is the BBOB function: ", problem_selection)
    print("Number of dimension: ", dimension)
    print("Number of data for training origianl GPR: ", number_of_data_GPR_training)
    print("Number of data for testing origianl GPR: ", number_of_data_GPR_test)
    print("Number of data for Transferred GPR: ", number_of_data_TL_training)
    print("Number of data for testing Transferred GPR: ", number_of_data_TL_test)
    print("The setting of stopping epoch threshold: ", stopping_epoch_threshold)
    print("The setting of training epochs: ", epochs)
    print("The setting of learning rate: ", alpha)
    print("The setting of batch size: ", batch_size)
    print("The setting of number of repetition for experiments: ", number_of_repetition)

    store_mean_absolute_percentage_error_GPR = []
    store_SMAPE_GPR = []
    store_R_square_score_GPR = []
    store_log_ratio_GPR = []
    store_square_sum_log_ratio_GPR = []

    store_mean_absolute_percentage_error_GPR_TL_test = []
    store_SMAPE_GPR_TL_test = []
    store_R_square_score_GPR_TL_test = []
    store_log_ratio_GPR_TL_test = []
    store_square_sum_log_ratio_GPR_TL_test = []
    
    store_mean_absolute_percentage_error_GPR_without_transfer = []
    store_SMAPE_GPR_without_transfer = []
    store_R_square_score_GPR_without_transfer = []
    store_log_ratio_GPR_without_transfer = []
    store_square_sum_log_ratio_GPR_without_transfer = []

    store_mean_absolute_percentage_error_TL = []
    store_SMAPE_TL = []
    store_R_square_score_TL = []
    store_log_ratio_TL = []
    store_square_sum_log_ratio_TL = []
    
    frobenius_value_list = []
    inner_product_list = []
    new_metric_value_list = []
    
    for k in range(1, number_of_repetition+1):
                
        np.random.seed(k)
        print('-------------------------------------------------------------------------------------------------')
        print("Number of repetition: ", k)
        print('-------------------------------------------------------------------------------------------------')

        # We generate the datasets accordingly.
        X_GPR_training, Y_GPR_training, X_GPR_test, Y_GPR_test, X_TL_training, Y_TL_training, \
            X_TL_test, Y_TL_test, TL_training_data, R, beta, problem1 = generate_data(problem_selection,
                                                                                            dimension,
                                                                                            number_of_data_GPR_training,
                                                                                            number_of_data_GPR_test,
                                                                                            max_number_of_data_TL_training,
                                                                                            number_of_data_TL_test, k, save_path)

        train_data_generation = TL_training_data[np.random.choice(TL_training_data.shape[0], number_of_data_TL_training, replace=False),:]
        X_TL_training = train_data_generation[:, :dimension]
        Y_TL_training = train_data_generation[:, [-1]]
        
        start_build_GPR_model = time.time()
        isExist = os.path.exists(save_original_model)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_original_model, exist_ok=True)
            print("The new directory is created!")
        
        complete_name = os.path.join(save_original_model, f"{problem_selection}_{k}"'.npy')
        # If the model exists, load it
        if os.path.isfile(complete_name):
            print("Loading existing model...")
            m = load_GP_model(X_GPR_training, Y_GPR_training, complete_name)
            print("Gaussian_noise variance of the original model: ", m.Gaussian_noise.variance[0])
        else:
            print("Building new model...")
            m = GP_regression(dimension, X_GPR_training, Y_GPR_training, k, save_original_model)
            np.save(complete_name, m.param_array)

        finish_build_GPR_model = time.time()
        print("The time for training the origianl GPR: ", finish_build_GPR_model - start_build_GPR_model, "seconds.")
        
        print("Test the original GPR model on original GPR test data: ")
        mean_GP, variance_GP = m.predict(X_GPR_test)
        MAPE_GPR, SMAPE_GPR, R_square_GPR, log_ratio_GPR, square_sum_log_ratio_GPR = prediction(mean_GP, Y_GPR_test)

        store_mean_absolute_percentage_error_GPR.append(MAPE_GPR)
        store_SMAPE_GPR.append(SMAPE_GPR)
        store_R_square_score_GPR.append(R_square_GPR)
        store_log_ratio_GPR.append(log_ratio_GPR)
        store_square_sum_log_ratio_GPR.append(square_sum_log_ratio_GPR)
        
        print("Test the origianl GPR on the test dataset of target function: ")
        mean_GP_test, variance_GP_test = m.predict(X_TL_test)
        MAPE_GPR_test, SMAPE_GPR_test, R_square_GPR_test, log_ratio_GPR_test, square_sum_log_ratio_GPR_test = prediction(mean_GP_test, Y_TL_test)

        store_mean_absolute_percentage_error_GPR_TL_test.append(MAPE_GPR_test)
        store_SMAPE_GPR_TL_test.append(SMAPE_GPR_test)
        store_R_square_score_GPR_TL_test.append(R_square_GPR_test)
        store_log_ratio_GPR_TL_test.append(log_ratio_GPR_test)
        store_square_sum_log_ratio_GPR_TL_test.append(square_sum_log_ratio_GPR_test)
        
        isExist = os.path.exists(save_without_transfer)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_without_transfer, exist_ok=True)
            print("The new directory is created!")
        
        complete_name = os.path.join(save_without_transfer, f"{problem_selection}_{k}"'.npy')
        # If the model exists, load it
        if os.path.isfile(complete_name):
            print("Loading existing model...")
            m_without_transfer = load_GP_model(X_TL_training, Y_TL_training, complete_name)
        else:
            print("Building new model...")
            m_without_transfer = GP_regression(dimension, X_TL_training, Y_TL_training, k, save_without_transfer)
            np.save(complete_name, m_without_transfer.param_array)
            
        print("Build and evaluate the GPR trained from scratch with all the transfer data: ")
        mean_GP_without_transfer, variance_GP  = m_without_transfer.predict(X_TL_test)
        MAPE_GPR_without_transfer, SMAPE_GPR_without_transfer, R_square_GPR_without_transfer, log_ratio_GPR_without_transfer, square_sum_log_ratio_GPR_without_transfer = prediction(mean_GP_without_transfer, Y_TL_test)

        store_mean_absolute_percentage_error_GPR_without_transfer.append(MAPE_GPR_without_transfer)
        store_SMAPE_GPR_without_transfer.append(SMAPE_GPR_without_transfer)
        store_R_square_score_GPR_without_transfer.append(R_square_GPR_without_transfer)
        store_log_ratio_GPR_without_transfer.append(log_ratio_GPR_without_transfer)
        store_square_sum_log_ratio_GPR_without_transfer.append(square_sum_log_ratio_GPR_without_transfer)
        
        '''
        Save the graident descent loss values
        '''
        start_training_each_epoch = time.time()
        isExist = os.path.exists(save_AutoML_RSGD)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_AutoML_RSGD, exist_ok=True)
            print("The new directory is created!")
        
        isExist = os.path.exists(save_loss)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_loss, exist_ok=True)
            print("The new directory is created!")
        
        final_rotation_matrix_complete_name = os.path.join(save_AutoML_RSGD, f"final_rotation_{problem_selection}_{k}"'.npy')
        final_translation_matrix_complete_name = os.path.join(save_AutoML_RSGD, f"final_translation_{problem_selection}_{k}"'.npy')
        
        # If the model exists, load it
        if os.path.isfile(final_rotation_matrix_complete_name):
            print("Loading existing model...")
            final_rotation_matrix = np.load(final_rotation_matrix_complete_name)
            final_translation_matrix = np.load(final_translation_matrix_complete_name)
        else:
            print("Building new model with the Gradient Descent ...")
            
            final_cost, final_rotation_matrix, final_translation_matrix, adam_history, rotation_history, translation_history, gradient_rotation_history, alpha_history, projection_gradient_rotation_history, restart_points, total_cpu_time = AutoML_Riemannian_gradient_descent(R, beta, dimension, 
                    X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, m, stopping_epoch_threshold, scheduler_type, epochs, batch_size, alpha, decay_rate, k, save_AutoML_RSGD)
            
            final_loss_path = os.path.join(save_loss, f"adam_history_{problem_selection}_{k}.json")
            with open(final_loss_path, "w") as file:
                json.dump(adam_history, file)
        
            np.save(final_rotation_matrix_complete_name, final_rotation_matrix)
            np.save(final_translation_matrix_complete_name, final_translation_matrix)

        end_training_each_epoch = time.time()
        print("The training time for the whole iterations with Riemannian Gradient : ", end_training_each_epoch - start_training_each_epoch, "seconds.")
        
        print(final_rotation_matrix)
        print(final_translation_matrix)

        # Frobenius norm value between the generated rotation matrix and the origianl matrix
        new_metric_value = rotation_metric(final_rotation_matrix, R)
        print("New metric: ", new_metric_value)
        
        frobenius_value = frobenius_norm(final_rotation_matrix, R)
        print("Frobenius norm: ", frobenius_value)
        
        normalized_frobenius_value = frobenius_norm(final_rotation_matrix, R) / dimension
        print("Frobenius norm (Normalized): ", normalized_frobenius_value)

        # Inner product between the generated rotation matrix and the origianl matrix (another metric)
        inner_product = np.trace(final_rotation_matrix.T @ R)
        print("Inner product: ", inner_product)
        
        normalized_inner_product = np.trace(final_rotation_matrix.T @ R) / dimension
        print("Inner product (Normalized): ", normalized_inner_product)
        
        new_metric_value_list.append(new_metric_value)
        frobenius_value_list.append(normalized_frobenius_value)
        inner_product_list.append(normalized_inner_product)
            
        print("Test the transferred GPR on transfer learning test data: ")
        x_affine_test = (final_rotation_matrix @ X_TL_test.T).T + final_translation_matrix
        mean_TL, variance_TL = m.predict(x_affine_test)
        mean_absolute_percentage_error_TL, SMAPE_TL, R_square_TL, log_ratio_TL, square_sum_log_ratio_TL = prediction(mean_TL, Y_TL_test)
        
        store_mean_absolute_percentage_error_TL.append(mean_absolute_percentage_error_TL)
        store_SMAPE_TL.append(SMAPE_TL)
        store_R_square_score_TL.append(R_square_TL)
        store_log_ratio_TL.append(log_ratio_TL)
        store_square_sum_log_ratio_TL.append(square_sum_log_ratio_TL)

        # Delete the model and free up memory
        del m, m_without_transfer, final_rotation_matrix, final_translation_matrix

        # At the end of each repetition, delete the datasets
        del X_GPR_training, Y_GPR_training, X_GPR_test, Y_GPR_test
        del X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, TL_training_data

        gc.collect()

    print("The Original Gaussian Process Regression Model: ")
    evaluation(store_mean_absolute_percentage_error_GPR, store_SMAPE_GPR, store_R_square_score_GPR, store_log_ratio_GPR, store_square_sum_log_ratio_GPR)

    print("The Original Gaussian Process Regression Model on transfer learning test data: ")
    evaluation(store_mean_absolute_percentage_error_GPR_TL_test, store_SMAPE_GPR_TL_test, store_R_square_score_GPR_TL_test, store_log_ratio_GPR_TL_test, store_square_sum_log_ratio_GPR_TL_test)
    
    print("The model built with transfer learning data: ")
    evaluation(store_mean_absolute_percentage_error_GPR_without_transfer, store_SMAPE_GPR_without_transfer, store_R_square_score_GPR_without_transfer, store_log_ratio_GPR_without_transfer, store_square_sum_log_ratio_GPR_without_transfer)

    print("The transferred Gaussian Process Regression Model on transfer learning test data: ")
    evaluation(store_mean_absolute_percentage_error_TL, store_SMAPE_TL, store_R_square_score_TL, store_log_ratio_TL, store_square_sum_log_ratio_TL)

    print("1. The p-value of U-test on MAPE")
    p_value_MAPE = mannwhitneyu_test(store_mean_absolute_percentage_error_GPR_TL_test, store_mean_absolute_percentage_error_TL)
    
    print("2. The p-value of U-test on SMAPE")
    p_value_SMAPE = mannwhitneyu_test(store_SMAPE_GPR_TL_test, store_SMAPE_TL)
    
    print("3. The p-value of U-test on R squared score")
    p_value_R2 = mannwhitneyu_test(store_R_square_score_GPR_TL_test, store_R_square_score_TL)
    
    print("4. The p-value of U-test on log ratio score")
    p_value_log_ratio = mannwhitneyu_test(store_log_ratio_GPR_TL_test, store_log_ratio_TL)
    
    print("5. The p-value of U-test on square sum log ratio")
    p_value_square_sum_log_ratio = mannwhitneyu_test(store_square_sum_log_ratio_GPR_TL_test, store_square_sum_log_ratio_TL)
    
    print("6. The average of frobenius value is: ", np.mean(frobenius_value_list))
    print("7. The average of inner product value is: ", np.mean(inner_product_list))
    print("8. The average of new metric value is: ", np.mean(new_metric_value_list))
    
    frobenius_box_plot(frobenius_value_list, inner_product_list, problem_selection, dimension, save_path)
    
    # Kruskal-Wallis Test
    group1 = store_SMAPE_GPR_TL_test
    group2 = store_SMAPE_GPR_without_transfer
    group3 = store_SMAPE_TL
    h_stat, p_val = kruskal(group1, group2, group3)

    # If p_val is significant, proceed with post-hoc analysis
    if p_val < 0.05:
        data = pd.DataFrame({
            'values': group1 + group2 + group3,
            'groups': ['group1']*len(group1) + ['group2']*len(group2) + ['group3']*len(group3)
        })
        
        # Dunn's test with Bonferroni correction
        post_hoc_res = sp.posthoc_dunn(data, val_col='values', group_col='groups', p_adjust='bonferroni')
        print(post_hoc_res)
        
    logger.reset()

if __name__ == '__main__':
    BBOB_function_list = ['Sphere', 'Ellipsoid', 'Rastrigin', 'BuecheRastrigin', 'LinearSlope', 'AttractiveSector', 'StepEllipsoid', 
                          'Rosenbrock', 'RosenbrockRotated', 'EllipsoidRotated', 'Discus', 'BentCigar', 'SharpRidge', 'DifferentPowers', 
                          'RastriginRotated', 'Weierstrass', 'Schaffers10', 'Schaffers1000', 'GriewankRosenBrock', 'Schwefel', 'Gallagher101', 
                          'Gallagher21', 'Katsuura', 'LunacekBiRastrigin']
    pool = Pool()
    result = pool.map(main, BBOB_function_list)
    pool.close()
    pool.join()