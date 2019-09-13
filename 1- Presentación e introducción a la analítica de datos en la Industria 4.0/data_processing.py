# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 11:03:02 2014

Module that defines some basic data processing functions.

@author: adiez
"""

import matplotlib.pyplot as plt
from scipy.spatial import distance
import random
from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd
#import pywt
from datetime import datetime


def totimestamp(dt, epoch=datetime(1970,1,1)):
    '''
    Transform date to timestamp.
        Inputs: 
            - dt: a date
            - epoch: from when we start to count
        Output: resulting timestamp, in milliseconds
    '''
    td = dt - epoch
    # return td.total_seconds()
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 1e6 


def csv_writer(data, path):    
    '''
    Write data to a CSV file path.
        Inputs: 
            - data: a data set (matrix [m,n])
            - path: path in which the data will be generated ('...\...csv')
        Output: the data file is created
    '''
    np.savetxt(path,data,delimiter=',')


def scale(matrix, axis=0):
    '''
    A simple scale function to normalize a matrix based on the mean and the std of its values.
        Inputs: 
            - matrix: a data set (matrix [m,n])
            - axis: apply the scaling by rows (0) or by columns (1)
        Output: scaled matrix (matrix [m,n])
    '''
    from numpy import mean, std
    return (matrix - mean(matrix, axis=axis)) / std(matrix, axis=axis)


def wavelet_transform(noisySignal):
    '''
    Method that applies wavelt tarnsform (Daubechies, decomposition level=12, filter size=12) to a given signal.
        Inputs: 
            - noisySignal: noisy signal to be transformed (array [m])            
        Output: (signal) wavelet transform of noisySignal (array [m])
    '''
    ########################################
    from statsmodels.robust import stand_mad
    ########################################
    levels = 12    
    denoised = pywt.wavedec(noisySignal, 'db12', level=levels, mode='per')
    sigma = stand_mad(denoised[-1])
    threshold = sigma*np.sqrt(2*np.log(len(noisySignal)))
    denoised = map(lambda x: pywt.thresholding.soft(x, value=threshold), denoised)
    signal = pywt.waverec(denoised, 'db12', mode='per')
    return signal


def scale_linear_bycolumn(rawpoints, high=1.0, low=0.0):
    '''
    Method that linearly scales a data set by columns.
        Inputs: 
            - rawpoints: a data set (matrix [m,n])
            - high: the maximum value
            - low: the minimum value 
        Output: scaled data set
    '''
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)


def get_random_color(pastel_factor=0.5):
    '''
    Method that randomly generates a new color.
        Inputs: 
            - pastel_factor: the pastel factor for the new color
        Output: the generated color
    '''    
    return [(x + pastel_factor)/(1.0 + pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

 
 
def color_distance(c1,c2):
    '''
    Method that calculates the distance between two colors.
        Inputs: 
            - c1: a color
            - c1: a color
        Output: the distance between c1 and c2
    '''
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])
 
def generate_new_color(existing_colors,pastel_factor = 0.5):
    '''
    Method that generates a new color given some constraints.
        Inputs: 
            - existing_colors: currently used colors, none of them will be generated
            - pastel_factor: the pastel factor for the new color
        Output: (best_color) the new color, different to any color in existing_colors
    '''
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color
    

def get_signal_features(data):
    '''
    Method that calculates some basic signal features.
        Inputs: 
            - data: training data set containing the events to be processed (matrix [m,n])
        Outputs:
            - data: calculated features, per event (matrix [m,k], where k is the number of features)
    '''
    data_tmp = []    
    for i in range(len(data)):
        sample = []
        mean = np.mean(data[i])
        std = np.std(data[i])
        mx = np.max(data[i])        
        # entropy = spectral_entropy(data[i], [0.5,4,7,12,30], 400)
        #power = bin_power(data[i], [0.5,4,7,12,30], 400)
        #pfd_value = pfd(data[i])
        #hjorth_value = hjorth(data[i])                
        sample.append(mean)
        sample.append(std)
        sample.append(mx+std)
        # sample.append(entropy)
        # sum of energies
        sample.append(np.sum(np.abs(data[i])**2))
        # energy peaks
        sample.append(np.max(np.abs(data[i])**2))
        sample.append(np.min(np.abs(data[i])**2))
        # energy peaks
        #sample.append(pfd_value)
        #sample.append(hjorth_value[0])
        #sample.append(hjorth_value[1])
        #for j in range(len(power[0])):
        #    sample.append(power[0][j])        
        data_tmp.append(sample)    
    data = np.array(data_tmp)
    # data = scale(data)
    
    return data    


def draw_joints_dist(labels, samples_distribution):
    '''
    This method creates a plot showing the joints distribution
        Inputs:
            - labels: np.array that contains the list of labels
            - samples_distribution: the labels distribution, one row per set of labels that will have the same color in the graph
        Outputs: a chart showing the label distribution
    '''
    # draw joints distribution
    joint_colors = []
    joints = []
    # set colors based on groups of joints
    # samples_distribution = pd.value_counts(pd.Series(index[:,0]))        
    for i in range(len(samples_distribution)):        
        #if (i != 6):
        color = generate_new_color(joint_colors, pastel_factor = 0.1)
        if isinstance(samples_distribution[i], list):
            for j in range(len(samples_distribution[i])):
                if (list(labels).count(samples_distribution[i][j]) > 0):
                    joint_colors.append(color)
                    joints.append(samples_distribution[i][j])
        else:
            if (list(labels).count(samples_distribution[i]) > 0):
                    joint_colors.append(color)
                    joints.append(samples_distribution[i])
                    
    joint_colors = np.array(joint_colors)    
    joints_distribution = pd.value_counts(pd.Series(labels.ravel()))    
    
    tmp = []    
    for i in joints:                         
        count = joints_distribution.values[joints_distribution.index == i]        
        if count > 0:         
            tmp.append(count)
            
    # the x locations for the groups
    ind = np.arange(len(joints))  
    # the width of the bars            
    width = 1                      
            
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.bar(ind, 
           tmp,           
           width,
           color = joint_colors)      
    ax.set_ylabel('number of events')
    ax.set_title('joints distribution from North to South')
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0, np.max(joints_distribution.values)+100)                   
    xTickMarks = ['Joint ' + str(int(joints[i])) for i in range(len(joints))]
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)