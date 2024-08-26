#!/usr/bin/env python
# coding: utf-8
'''
Script Name: gradient.py
Description: This script is the program of implementing the cost function, riemannian gradient descent method.
'''
import math
import numpy as np
from scipy.linalg import expm, logm
from utils import *
from smac import HyperparameterOptimizationFacade, Scenario
from ConfigSpace.conditions import InCondition
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from numpy_ml.neural_nets.schedulers import ConstantScheduler, ExponentialScheduler, KingScheduler
from scipy.stats import special_ortho_group
import time


def cost_function(x, y, weight, translation, m):
    '''
    Implement the defined cost function.
    '''
    n_T = x.shape[0]
    x_affine = (weight @ x.T).T + translation
    mean, variance = m.predict(x_affine)
    diff = mean - y
    differences_squared = diff ** 2
    mean_diff = np.sum(differences_squared) / (2 * n_T)
    return mean_diff

def manifold_gradient_cost_function_with_penalties(x, y, weight, translation, mean, derivative_mean, dimension):
    '''
    The gradient functions for Riemannian transferredGP.
    '''
    n_T = x.shape[0]
    weight_derivative = np.zeros((dimension, dimension))
    translation_derivative = np.zeros((1, dimension))
    derivative_mean = derivative_mean.reshape((n_T, dimension))
    diff = mean - y
    
    start_time = time.process_time()  # Start timing for CPU time
    for k in range(n_T):
        weight_derivative += diff[k] * np.outer(derivative_mean[k], x[k])
    weight_derivative /= n_T

    translation_derivative = np.dot(diff.T, derivative_mean)
    translation_derivative /= n_T

    gradient_calculation_time = time.process_time() - start_time  # CPU time for the total Gradient Calculation
                
    projection_weight_derivative = weight @ (0.5 * (weight.T @ weight_derivative - weight_derivative.T @ weight))

    # Check if the weight_derivative is orthogonal to its projection
    difference = weight_derivative - projection_weight_derivative
    orthogonality_check = np.trace(difference.T @ projection_weight_derivative)

    # Debug output for orthogonality
    # print(f"Orthogonality Check Value: {orthogonality_check}")

    # Extract theta from the rotation matrix for derivative calculation
    theta = np.arctan2(weight[1, 0], weight[0, 0])
    dW_dtheta = np.array([
        [-np.sin(theta), -np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]
    ])

    # Derivative w.r.t theta
    derivative_theta = np.trace(dW_dtheta.T @ weight_derivative)
    
    # Riemannian gradient Derivative w.r.t theta
    projection_derivative_theta = np.trace(dW_dtheta.T @ projection_weight_derivative)

    returnValue = np.array([projection_weight_derivative, translation_derivative, derivative_theta, projection_derivative_theta, gradient_calculation_time], dtype=object)

    return returnValue

def iterate_minibatches(inputs, targets, batchsize, j, shuffle=False):
    '''
    The mini batch function.
    '''
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        rng = np.random.default_rng(seed = j)
        rng.shuffle(indices)
    for start_idx in range(0, inputs.shape[0], batchsize):
        end_idx = min(start_idx + batchsize, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]
 
def Riemannian_gradient_descent(R, beta, dimension, X_TL_training, Y_TL_training, m, stopping_epoch_threshold, k, 
                                scheduler_type, epochs, batch_size, alpha, decay_rate):
    '''
    The implementations of Riemannian gradient descent method.
    '''
    def initialize(last_rotation=None, current_translation=None, epoch=None):
        if last_rotation is None:
            if epoch is not None:
                rotation_seed = k + epoch  # Varies with each epoch
                last_rotation = special_ortho_group.rvs(dim=dimension, random_state=rotation_seed)
            else:
                last_rotation = np.identity(dimension)  # Use identity matrix when epoch is None
        if current_translation is None:
            current_translation = np.random.uniform(-0.5, 0.5, (1, dimension))
        initial_theta = np.array([last_rotation, current_translation], dtype=object)
        
        scheduler = schedulers[scheduler_type.lower()]()
        
        return initial_theta, scheduler
    
    # Define your schedulers
    schedulers = {
        "exponential": lambda: ExponentialScheduler(initial_lr=alpha, decay_rate=decay_rate),
    }
    
    # Check if scheduler_type is valid
    if scheduler_type.lower() not in schedulers:
        raise ValueError(f"Invalid scheduler_type: {scheduler_type}")
    
    theta, scheduler = initialize()

    previous_cost = None
    adam_history = []
    rotation_matrix_history = []
    translation_matrix_history = []
    gradient_rotation_history = []  
    alpha_history = []  
    projection_gradient_rotation_history = []
    total_cpu_time = 0

    restart_points = []  # List to store the epochs where restarts occur
    
    for j in range(epochs):
        alpha = scheduler(j)
        alpha_history.append(alpha)
        
        np.random.seed(k + j)
        for batch in iterate_minibatches(X_TL_training, Y_TL_training, batch_size, j, shuffle=True):
            X_mini, y_mini = batch
            X_affine = (theta[0] @ X_mini.T).T + theta[1]
            mean_training, variance_training = m.predict(X_affine)
            derivative_mean_training, derivative_variance_training = m.predictive_gradients(X_affine)

            # Calculating the gradients
            gradients = manifold_gradient_cost_function_with_penalties(X_mini, y_mini, theta[0], theta[1], mean_training, derivative_mean_training, dimension)
            gradient_rotation_history.append(gradients[2])
            projection_gradient_rotation_history.append(gradients[3])
            total_cpu_time += gradients[4]
            
            # Update theta using gradients
            theta[0] = theta[0] @ expm(-alpha * theta[0].T @ gradients[0])
            theta[1] = theta[1] - (alpha * gradients[1])

        # Calculate cost and update history
        current_cost = cost_function(X_TL_training, Y_TL_training, theta[0], theta[1], m)
        adam_history.append(current_cost)
        rotation_matrix_history.append(theta[0])
        translation_matrix_history.append(theta[1])

        # Check the stopping condition for small cost change
        if previous_cost and np.isclose(previous_cost, current_cost):
            restart_points.append(j)
            print(f"Restarting with a new rotation matrix but the same translation matrix due to small cost change: {previous_cost} to {current_cost}")
            theta, scheduler = initialize(epoch=j)  # Pass current epoch to vary random state
            previous_cost = None  # Reset the previous cost
            continue

        previous_cost = current_cost

    print("The number of iterations for training: ", len(adam_history))
    index_minimum_cost = np.argmin(adam_history)
    final_cost = adam_history[index_minimum_cost]
    print("The minimum loss value: ", adam_history[index_minimum_cost])
    final_rotation_matrix = rotation_matrix_history[index_minimum_cost]
    final_translation_matrix = translation_matrix_history[index_minimum_cost]
    print("The determinant of the generated rotation matrix: ", np.linalg.det(final_rotation_matrix))
    print('\nFinal rotation matrix = {}'.format(final_rotation_matrix),
          '\nFinal translation matrix = {}'.format(final_translation_matrix))
    
    return final_cost, final_rotation_matrix, final_translation_matrix, adam_history, rotation_matrix_history, translation_matrix_history, gradient_rotation_history, alpha_history, projection_gradient_rotation_history, restart_points, total_cpu_time


class AutoML():
    '''
    The implementations of the hyper parameter tuning of the riemannian graident descent method.
    '''
    def __init__(self, R, beta, dimension, X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, m, stopping_epoch_threshold, k, scheduler_type, epochs, batch_size, 
                 alpha, decay_rate):
        self.R = R
        self.beta = beta
        self.dimension = dimension
        self.alpha = alpha
        self.epochs = epochs
        self.X_TL_training = X_TL_training
        self.Y_TL_training = Y_TL_training
        self.batch_size = batch_size
        self.m = m
        self.stopping_epoch_threshold = stopping_epoch_threshold
        self.decay_rate = decay_rate
        self.scheduler_type = scheduler_type
        self.k = k
        self.X_TL_test = X_TL_test
        self.Y_TL_test = Y_TL_test
        

    def configspace(self, seed) -> ConfigurationSpace:
        # Build the configuration space with all parameters.
        cs = ConfigurationSpace(seed = seed)

        # Create the hyperparameters with their range
        lr = Float("lr", (1e-7, 1e-1), default=0.001, log=True)
        batch_size = Integer("batch_size", (math.ceil(self.batch_size/2), self.batch_size), default=self.batch_size)
        decay_rate = Float("decay_rate", (0, 1), default=0.1)

        # Add hyperparameters to the configspace
        cs.add_hyperparameters([lr, batch_size, decay_rate])

        return cs

    def train(self, config: Configuration, seed: int) -> float:
        config_dict = config.get_dictionary()

        final_cost, final_rotation_matrix, final_translation_matrix, adam_history, rotation_history, translation_history, gradient_rotation_history, alpha_history, projection_gradient_rotation_history, restart_points, total_cpu_time = Riemannian_gradient_descent(self.R, self.beta,
            self.dimension, self.X_TL_training, self.Y_TL_training, self.m, self.stopping_epoch_threshold, self.k, self.scheduler_type, self.epochs, 
            batch_size=config_dict['batch_size'], alpha=config_dict['lr'], decay_rate=config_dict['decay_rate'])
        
        x_affine_test = (final_rotation_matrix @ self.X_TL_test.T).T + final_translation_matrix
        mean_TL, variance_TL = self.m.predict(x_affine_test)
        symmetric_mean_absolute_percentage_error = smape(self.Y_TL_test, mean_TL)

        return symmetric_mean_absolute_percentage_error

def AutoML_Riemannian_gradient_descent(R, beta, dimension, X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, m, stopping_epoch_threshold, scheduler_type, epochs, batch_size, alpha, decay_rate, k, save_path):
    Riemannian_SGD = AutoML(R, beta, dimension, X_TL_training, Y_TL_training, X_TL_test, Y_TL_test, m, stopping_epoch_threshold, k, scheduler_type, epochs, batch_size, alpha, decay_rate)
    
    # Create a configurationSpace instance
    cs = Riemannian_SGD.configspace(k)
    
    # Specify the optimization environment with a defined n_trials
    scenario = Scenario(cs, deterministic=True, n_trials=250, output_directory=save_path, seed=k)

    # Use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(scenario, Riemannian_SGD.train)
    incumbent = smac.optimize()
    
    print("Incumbent lr :", incumbent["lr"])
    print("Incumbent decay_rate :", incumbent["decay_rate"])
    print("Incumbent batch_size :", incumbent["batch_size"])
    
    final_cost, final_rotation_matrix, final_translation_matrix, adam_history, rotation_history, translation_history, gradient_rotation_history, alpha_history, projection_gradient_rotation_history, restart_points, total_cpu_time = Riemannian_gradient_descent(R, beta, 
            dimension, X_TL_training, Y_TL_training, m, stopping_epoch_threshold, k, scheduler_type, epochs, batch_size=incumbent["batch_size"],
            alpha=incumbent["lr"], decay_rate=incumbent["decay_rate"])
    
    return final_cost, final_rotation_matrix, final_translation_matrix, adam_history, rotation_history, translation_history, gradient_rotation_history, alpha_history, projection_gradient_rotation_history, restart_points, total_cpu_time