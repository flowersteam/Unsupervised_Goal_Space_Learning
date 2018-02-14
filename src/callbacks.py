#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
This module contains callback to use with Base Network training method.

Author: Alexandre Péré

"""

import numpy
import IPython.display
import matplotlib.pyplot as plt
import time

def ipython_clear_callback():
    """
    A callback that clears ipython outputs.
    """
    IPython.display.clear_output()
    
    
def plot_callback(network, save_path=None):
    """
    A callback that plots train and test accuracy over time.
    """
    history = network.get_history()
    plt.plot(history[:,1], label="Train")
    plt.plot(history[:,2], label="Test")
    plt.legend(loc=4)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
def print_callback(network):
    """
    A callback that print informations about the training.
    """    
    history = network.get_history()
    print("Number of iterations: %i"%(history.shape[0]*100))
    print("Elapsed Time: %s"%(time.strftime("%H:%M:%S", time.gmtime(history[-1,0]))))
    print("Last iteration duration: %.2f seconds"%(history[-1,0]-history[-2,0]))
    print("Last train accuracy: %f"%(history[-1,1]))
    print("Last test accuracy: %f"%(history[-1,2]))
    
def save_callback(network, path, over_iter=100):
    """
    A callback that saves if the iterations number is a multiple of some integer.
    """
    history = network.get_history()
    if history.shape[0]%over_iter == 0:
        network.save(path)
        print("Model Saved")
    
    
    
    
    
    
    
    
    