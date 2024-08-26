#!/usr/bin/env python
# coding: utf-8
'''
Script Name: surrogate.py
Description: This script is the program of implementing the GPR surrogate model.
'''
import GPy
import numpy as np
from sklearn.model_selection import cross_val_score
from utils import *
from IPython.display import display
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from smac import HyperparameterOptimizationFacade, Scenario


def GP_regression(dimension, X, Y, k, save_path):
    '''
    Implement the GPR with GPy library.
    '''
    # Define the RBF kernel
    rbf = GPy.kern.RBF(input_dim=dimension, variance=1., lengthscale=1.)
    kernel = rbf

    m = GPy.models.GPRegression(X, Y, kernel)

    print("Default the noise variance: ", m.Gaussian_noise.variance)
    
    m.optimize(messages=False)
    display(m)
    
    print("After Optimization: ", m.Gaussian_noise.variance)
    
    return m

def load_GP_model(X, Y, parameter):
    '''
    Load the GPR with GPy library if the model is already existed.
    '''
    # Model creation, without initialization:
    m_load = GPy.models.GPRegression(X, Y, initialize=False)
    m_load.update_model(False) 
    m_load.initialize_parameter() 
    m_load[:] = np.load(parameter)
    m_load.update_model(True)
    return m_load
