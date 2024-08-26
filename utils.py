#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from decimal import *
from scipy.stats import mannwhitneyu
import seaborn as sns
from scipy.linalg import logm
import math


class Logger(object):
    '''
    Save all the printout information for further analyze
    '''
    def __init__(self, fileN='default.log'):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(fileN, 'w+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass

        
def frobenius_norm(obtained_matrix, actual_affine_norm):
    '''
    Frobenius norm, we use it to calculate the distance between two matrix
    '''
    return LA.norm(obtained_matrix - actual_affine_norm, 'fro')

def rotation_metric(R1, R2):
    # Compute the product of R1 and the transpose of R2
    product = np.dot(R1, R2.T)

    # Compute the matrix logarithm of the product
    log_product = logm(product)

    # Compute the Frobenius (Euclidean) norm of the matrix logarithm
    norm = np.linalg.norm(log_product, 2)
    return norm

def average_value(stored_value):
    average_score = np.mean(stored_value)
    try:
        if isinstance(average_score, (int, float)) and not (math.isnan(average_score) or math.isinf(average_score)):
            average_score = Decimal(average_score).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
        else:
            print(f"Invalid average_score: {average_score}")
    except InvalidOperation:
        print(f"Error converting average_score to Decimal: {average_score}")
    return average_score

def median_value(stored_value):
    median_score = statistics.median(stored_value)
    median_score = Decimal(median_score).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
    return median_score

def numpy_average_value(stored_value):
    final_array = sum(stored_value) / len(stored_value)
    return final_array

def IQR(stored_value):
    q3, q1 = np.percentile(stored_value, [75, 25])
    iqr = q3 - q1
    iqr = Decimal(iqr).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
    return iqr

def standard_deviation(x):
    std = np.std(x)
    std = Decimal(std).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
    return std
    
def round_number(x):
    round_x = Decimal(x).quantize(Decimal('0.0000'), rounding=ROUND_HALF_UP)
    return round_x

def get_identifier_for_number(base_path, number):
    unique_ids = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for uid in unique_ids:
        if os.path.exists(os.path.join(base_path, uid, str(number))):
            return uid
    return None
    
def loss_plot(adam_history, problem_selection, save_path, k, bsf_value):
    '''
    Getting the loss plot of each repetition.
    '''
    figure1 = plt.figure(figsize=(8, 6), dpi=180)
    host = host_subplot(111)
    host.set_xlabel("Epochs")
    host.set_ylabel("Loss value")
    
    transformed_adam_history = np.log10(adam_history) + bsf_value
    
    p1, = host.plot(transformed_adam_history, label="Loss")
    host.legend(labelcolor="linecolor")
    host.yaxis.get_label().set_color(p1.get_color())
    plt.title(f"Problem: {problem_selection}, Best_so_far: {bsf_value}", fontsize=10)
    complete_name = os.path.join(save_path, f"{problem_selection}_Loss_{k}"'.png')
    plt.savefig(complete_name)
    plt.close()
    
def frobenius_plot(label, value, problem_selection, dimension, save_path, name):
    '''
    Getting the frobenius norm value plot of each repetition.
    '''
    figure2 = plt.figure(figsize=(8, 6), dpi=180)
    host = host_subplot(111)
    host.set_xlabel("Number of data")
    host.set_ylabel(label)
    p1, = host.plot(value, label=label)
    host.legend(labelcolor="linecolor")
    host.yaxis.get_label().set_color(p1.get_color())
    plt.title(f"Problem: {problem_selection}, Dimension: {dimension}, Sample Size Experiment from 50 to 10", fontsize=10)
    plt.gca().invert_xaxis()
    plt.xticks(range(8, -1, -1), range(10, 51, 5))
    complete_name = os.path.join(save_path, f"{problem_selection}_{name}"'.png')
    plt.savefig(complete_name)
    plt.close()
    
def frobenius_box_plot(frobenius_value_list, inner_product_list, problem_selection, dimension, save_path):
    '''
    Visualize the frobneius value with all the repetitions.
    '''
    data1 = frobenius_value_list
    data2 = inner_product_list
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=[data1, data2], palette='Set3')
    sns.stripplot(data=[data1, data2], color=".25", jitter=0.2, size=8)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='dotted')
    plt.title(f"Problem: {problem_selection}, Dimension: {dimension}, Frobenius Norm and Inner product value")
    plt.xticks([0, 1], ['Frobenius Norm', 'Inner Product'])  # rename the x-axis labels
    complete_name = os.path.join(save_path, f"{problem_selection}_Frobenius_norm_boxplot"'.png')
    plt.savefig(complete_name)
    plt.close()
    
def standardlize_data(Y):
    scaler = StandardScaler()
    scaler.fit(Y)
    Y = scaler.transform(Y)
    return Y

def MinMaxScaler_data(Y):
    scaler = MinMaxScaler()
    scaler.fit(Y)
    Y = scaler.transform(Y)
    return Y

def normalizer_data(Y):
    transformer = Normalizer().fit(Y)
    return transformer.transform(Y)

def mannwhitneyu_test(Y1, Y2):
    # perform mann whitney test
    stat, p_value = mannwhitneyu(Y1, Y2, alternative="less", method="exact")
    alpha = 0.05
    print('Statistics=%.5f, p=%.5f' % (stat, p_value))
    if p_value < alpha:
        print('Reject Null Hypothesis (Significant difference between two samples)')
    else:
        print('Do not Reject Null Hypothesis (No significant difference between two samples)')
    print("The p-value for U-test: ", p_value)
    return p_value

def prediction(Y_predict, Y_true):
    '''
    Evaluate the models based on different evaluation metrics.
    '''
    mean_absolute_percentage_error = metrics.mean_absolute_percentage_error(Y_true, Y_predict)
    symmetric_mean_absolute_percentage_error = smape(Y_true, Y_predict)
    R_square_score = metrics.r2_score(Y_true, Y_predict)
    log_ratio_score = log_ratio(Y_true, Y_predict)
    square_sum_log_ratio_score = square_sum_log_ratio(Y_true, Y_predict)
    print('-------------------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------------------')
    print("The mean absolute percentage error (MAPE) is: ", mean_absolute_percentage_error)
    print("The symmetric mean absolute percentage error (SMAPE) is: ", symmetric_mean_absolute_percentage_error)
    print("The R Square score is: ", R_square_score)
    print("The log ratio score is: ", log_ratio_score)
    print("The square sum log ratio score is: ", square_sum_log_ratio_score)
    print('-------------------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------------------')
    
    return mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, R_square_score, log_ratio_score, square_sum_log_ratio_score

def evaluation(store_MAPE, store_SMAPE, store_R_square, store_log_ratio_score, store_square_sum_log_ratio_score):
    '''
    Get the averaged values for the models based on different evaluation metrics.
    '''
    print(store_MAPE)
    MAPE_final = average_value(store_MAPE)
    MAPE_final_std = standard_deviation(store_MAPE)
    
    print(store_SMAPE)
    SMAPE_final = average_value(store_SMAPE)
    SMAPE_final_std = standard_deviation(store_SMAPE)
    
    print(store_R_square)
    R_square_score_final = average_value(store_R_square)
    R_square_score_final_std = standard_deviation(store_R_square)
    
    print(store_log_ratio_score)
    store_log_ratio_score_final = average_value(store_log_ratio_score)
    store_log_ratio_score_final_std = standard_deviation(store_log_ratio_score)
    
    print(store_square_sum_log_ratio_score)
    store_square_sum_log_ratio_score_final = average_value(store_square_sum_log_ratio_score)
    store_square_sum_log_ratio_score_final_std = standard_deviation(store_square_sum_log_ratio_score)
    
    print('-------------------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------------------')
    print("1. The mean absolute percentage error (MAPE) is: ", MAPE_final)
    print("The standard deviation of MAPE is: ", MAPE_final_std)
    print("2. The symmetric mean absolute percentage error (SMAPE) is: ", SMAPE_final)
    print("The standard deviation of SMAPE is: ", SMAPE_final_std)
    print("3. The R Squared score is: ", R_square_score_final)
    print("The standard deviation of R Squared score is: ", R_square_score_final_std)
    print("4. The log ratio is: ", store_log_ratio_score_final)
    print("The standard deviation of log ratio is: ", store_log_ratio_score_final_std)
    print("5. The square sum log ratio is: ", store_square_sum_log_ratio_score_final)
    print("The standard deviation of square sum log ratio is: ", store_square_sum_log_ratio_score_final_std)
    print('-------------------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------------------')

def log_ratio(Y_true, Y_predict, epsilon=1e-10):
    ratio = Y_predict / Y_true
    ratio = np.clip(ratio, epsilon, np.inf)  # Ensure ratio is not zero or negative
    log_ratio = np.log(ratio)
    return np.abs(np.mean(log_ratio))

def square_sum_log_ratio(Y_true, Y_predict, epsilon=1e-10):
    ratio = np.clip(Y_predict / Y_true, epsilon, np.inf)
    squared_log_ratio = np.log(ratio) ** 2
    return np.sum(squared_log_ratio)
    
def smape(actual, predict):
    '''
    The SMAPE evaluation metric.
    '''
    errors = np.abs(actual - predict)
    scale = np.abs(actual) + np.abs(predict)
    return np.mean(2 * errors / scale)

# Define a custom colorbar formatter
def log_formatter(x, pos):
    return "{:.0f}".format(np.log(x+0.1))

def rotation_angle_from_matrix(R):
    # Ensure the matrix is square and of appropriate dimensions
    if R.shape[0] == 2 and R.shape[1] == 2:
        # Calculate the angle using arctan2
        theta = np.arctan2(R[1, 0], R[0, 0])
        return np.degrees(theta)  # Convert the angle to degrees
    elif R.shape[0] == 3 and R.shape[1] == 3:
        # Calculate the angle for a 3D rotation matrix around z-axis
        theta = np.arctan2(R[1, 0], R[0, 0])
        return np.degrees(theta)  # Convert the angle to degrees
    else:
        raise ValueError("The matrix is not a 2D or 3D rotation matrix.")