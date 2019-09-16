# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:07:19 2015

Module that defines some basic methods to calculate and draw maps of pairwise distances.

@author: adiez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib



class MyMath(object):

    def is_number(self, x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    def get_distance(self, x, y, distance=None, p=None, w=None, cov=None):
        #sort out dimensions
        if x.ndim > 1:
            #at least one of them is a table, add over columns
            axis_tmp = 1
            n_features = x.shape[1]
        elif y.ndim > 1:
            axis_tmp = 1
            n_features = y.shape[1]
        else:
            axis_tmp = 0
            n_features = x.shape[0]
        if w is None:
            w = np.ones(n_features)
        if distance is None or distance == 'SqEuclidean':
            return (((x - y)**2)*w).sum(axis=axis_tmp)
        if distance == 'Euclidean':
            return (((x - y)**2)*w).sum(axis=axis_tmp)**0.5
        elif distance == 'Manhattan':
            return (abs(x - y)*w).sum(axis=axis_tmp)
        elif distance == 'Correlation':
            return (np.corrcoef(x,y)[0][1])
        elif distance == 'Minkowski' and self.is_number(p):
            if p == 0:
                raise Exception('Exponent p cannot be zero (Minkowski distance)')
            else:
                return ((abs(x - y)**p)*w).sum(axis=axis_tmp)**(1/p)
        elif distance == 'Minkowski_pthPower' and self.is_number(p):
            return ((abs(x - y)**p)*w).sum(axis=axis_tmp)
        elif distance == 'Chebyshev':
            return (abs(x - y)*w).max(axis=axis_tmp)
        elif distance == 'Mahalanobis': 
            if cov == None:
                raise Exception('Inverse of covariance matrix cannot be null (Mahalanobis distance)')            
            delta = x - y            
            if axis_tmp == 1:
#                return [mahalanobis(x[i],y,cov) for i in range(len(x))]
#            else:
#                return mahalanobis(x,y,cov)
                m = [np.dot(np.dot(delta[i], cov), delta[i]) for i in range(len(delta))]                
            else:
                m = np.dot(np.dot(delta, cov), delta)
            return np.sqrt(m)
        else:
            raise Exception('Unrecognised distance')

    def get_center(self, x, distance=None, p=None, gradient=np.array([0.001])):
        if distance is None or distance == 'SqEuclidean' or distance == 'Euclidean' or distance == 'Mahalanobis':
            return x.mean(axis=0)
        elif distance == 'Manhattan':
            return np.median(x, axis=0)
        elif distance == 'Chebyshev':
            return x.mean(axis=0) #(x.max(axis=0) - x.min(axis=0))/2
        elif (distance == 'Minkowski' or distance == 'Minkowski_pthPower') and self.is_number(p):
            gradient = gradient.repeat(x.shape[1])
            #starts the search from the mean
            x_center = x.mean(axis=0)
            distance_to_x_center = (abs(x - x_center)**p).sum(axis=0)
            nx_center = x_center + gradient
            distance_to_nx_center = (abs(x - nx_center)**p).sum(axis=0)
            gradient[distance_to_x_center < distance_to_nx_center] *= -1
            while True:
                nx_center = x_center + gradient
                distance_to_nx_center = (abs(x - nx_center)**p).sum(axis=0)
                gradient[distance_to_nx_center >= distance_to_x_center] *= 0.9
                x_center[distance_to_nx_center < distance_to_x_center] = nx_center [distance_to_nx_center < distance_to_x_center]
                distance_to_x_center[distance_to_nx_center < distance_to_x_center] = distance_to_nx_center[distance_to_nx_center < distance_to_x_center]
                if np.all(np.abs(gradient) < 0.0001):
                    return x_center
        else:
            raise Exception('Unrecognised distance')

    def compare_categorical_vectors(self, x, y):
        #simple comparison between vectors x and y using a confusion matrix
        #returns: accuracy in [0,1], the confusion matrix, base-zero indexes of categorical pairs
        max_n_categories = int(np.max([x.max(), y.max()]) + 1)
        #Create confusion matrix
        confusion_matrix = np.zeros([max_n_categories, max_n_categories])
        for i in range(max_n_categories):
            for ii in range(max_n_categories):
                confusion_matrix[i, ii] = sum(y[x == i] == ii)
        final_confusion_matrix = confusion_matrix[:]
        # Remove pairs from the table, from high to low
        pairs = np.zeros([max_n_categories, 2])
        r = 0
        for i in range(max_n_categories):
            [pairs[i, 0], pairs[i, 1]] = np.unravel_index(confusion_matrix.argmax(), confusion_matrix.shape)
            r += confusion_matrix[pairs[i, 0], pairs[i, 1]]
            confusion_matrix[pairs[i, 0], :] = -1
            confusion_matrix[:, pairs[i, 1]] = -1
        return r/np.max([x.size, y.size]), final_confusion_matrix, pairs

    def standardize(self, data, categorical_features=np.array([]), stand_type='range'):
        #Standarize a data set that is NxFeatures
        #If stand_type = std, it uses the corrected standard deviation
        if stand_type == 'range':
            data_range = data.max(axis=0) - data.min(axis=0)
            if any(data_range == 0):
                raise Exception('Division by zero (range)')
            else:
                return (data - data.mean(axis=0)) / data_range
        elif stand_type == 'std':
            data_std = data.std(axis=0, ddof=1)
            if any(data_std == 0):
                raise Exception('Division by zero (std)')
            else:
                return (data - data.mean(axis=0)) / data_std
        else:
            raise Exception('Unrecognised standardisation type')

    def get_distance_entity_to_centroid(self, u, data, centroids, distance=None, p=None):
        #return the sum of distances between entities and their respective centroids
        r = 0
        for k_i in range(u.max()+1):
            r += sum(self.get_distance(data[u == k_i, :], centroids[k_i, :], distance, p))
        return r

    def get_entity_with_min_distance(self, data, distance=None, p=None, w=None):
        #Returns the entity with the smallest sum of distances to all others, and the sum of distances
        if data.ndim == 1:
            return 0,0
        entities_n = data.shape[0]
        (min_distance, entity_index) = self.get_distance(data, data[0, :], distance, p, w).sum(),  0
        for i in range(1, entities_n):
            tmp_dist = self.get_distance(data, data[i, :], distance, p, w).sum()
            if tmp_dist < min_distance:
                (min_distance, entity_index) = tmp_dist,  i
        return entity_index, min_distance



def draw_matrix_of_distances(centroids, labels, njoints=6, weights=None, distance='Euclidean', p=0.1, cov_matrix=None):   
    '''
    Method that creates a map of pairwise distances.
        Inputs:
            - centroids: joint representatives (matrix [m,n])            
            - labels: centroids' labels (array [m])
            - weights: weighting of centroids (array [m]) 
            - distance: distance metric to be used
            - p: pthPower used by Minkowski distance
        Output: (global_matrix) the resulting map of pairwise distances (matrix [m,m])
    '''    
    # calculate matrix of distances    
    mymath = MyMath()
    global_matrix = []        
    for i in range(len(centroids)):    
        matrix_tmp = []      
        # calculate distance to closest centroid of other joints
        for k in range(len(centroids)):                
            matrix_tmp.append(mymath.get_distance(centroids[i], centroids[k], distance, w=weights, p=p, cov=cov_matrix))        
            #matrix_tmp.append(mahalanobis(centroids[i], centroids[k], cov_matrix))
        matrix_tmp = np.array(matrix_tmp)     
        global_matrix = np.concatenate((global_matrix, matrix_tmp), axis=0)
        
    global_matrix = np.array(global_matrix)
    global_matrix = global_matrix.reshape(len(centroids),len(centroids))   
    cmap = matplotlib.cm.coolwarm    
    #cmap = matplotlib.cm.RdBu    
    # draw matrix of distances    
    plt.figure(figsize=(30,24))
    plt.imshow(np.array(global_matrix), cmap = cmap, interpolation='nearest')       
    tickMarks = ['Joint ' + str(int(labels[i])) for i in range(len(labels))]
    plt.xticks(range(len(tickMarks)), tickMarks, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(tickMarks)), tickMarks, rotation=45, ha='right', fontsize=10)
    #plt.title('Map of pair-wise distances between joints centroids. ' + distance)
    # axis x info: labels that must be customised depending on the number of joints and areas under study    
    #    plt.text(15, 77, 'span6',horizontalalignment='center',verticalalignment='top',color='red')
#    plt.text(44, 77, 'span7',horizontalalignment='center',verticalalignment='top',color='red')    
#    plt.text(60, 77, 'span8',horizontalalignment='center',verticalalignment='top',color='red')    
#    plt.text(69, 77, 'N_pylon',horizontalalignment='center',verticalalignment='top',color='red')    
#    plt.text(74, 79, 'N_mainSpan',horizontalalignment='center',verticalalignment='top',color='red')    
#    # axis y info    
#    plt.text(-5, 16, 'span6',horizontalalignment='center',verticalalignment='top',color='red',rotation='vertical')
#    plt.text(-5, 42, 'span7',horizontalalignment='center',verticalalignment='top',color='red',rotation='vertical')    
#    plt.text(-5, 58, 'span8',horizontalalignment='center',verticalalignment='top',color='red',rotation='vertical')    
#    plt.text(-5, 67, 'N_pylon',horizontalalignment='center',verticalalignment='top',color='red',rotation='vertical')    
#    plt.text(-7, 69, 'N_mainSpan',horizontalalignment='center',verticalalignment='top',color='red',rotation='vertical')    
    # color bar for the heat map    
    plt.colorbar()
    plt.grid()
    # show lines that separates different bridge areas or groups of joints: it must be customised depending on the number of joints and areas under study  
    #draw_cmap_lines()
    # show only areas of interest by specifying graph bounds: it must be customised depending on the number of joints and areas under study  
    #plt.xlim(-0.5, 29.5)
    #plt.ylim(29.5, -0.5)    
    if njoints==6:
        plt.xlim(29.5, 50.5)
        plt.ylim(50.5, 29.5)
    #plt.xlim(50.5, 62.5)
    #plt.ylim(62.5, 50.5)    
    #plt.xlim(62.5, 69.5)
    #plt.ylim(69.5, 62.5)
    return global_matrix


def calculate_joints_means(data, labels, samples_distribution):
    '''
    Method that calculates the means of the data related to each label (joints representatives).
        Inputs:
            - data: training data set (matrix [m,n])            
            - labels: array that contains the list of data labels, or data joints (array [m])
            - samples_distribution: the labels distribution, one row per set of labels that will have the same color in the graph
        Outputs: (means, joints) the calculated joints data means (matrix [k,n], k=number of different labels or joints) and joints the list of different joints (array [k])
    '''    
    # calculate means of joints' events
    tmp = []
    for i in range(len(samples_distribution)):
        if isinstance(samples_distribution[i], list):
            for j in range(len(samples_distribution[i])):
                tmp.append(samples_distribution[i][j])
        else:
            tmp.append(samples_distribution[i])
    joints = []
    for i in range(len(tmp)):
        joint_data = data[labels == tmp[i]] 
        if (len(joint_data) > 0):
            if (i == 0):
                means = np.mean(joint_data, axis = 0)
            else:
                means = np.vstack((means, np.mean(joint_data, axis = 0)))
            joints.append(tmp[i])
    
    return means, joints


def draw_cmap_lines():
    '''
    Method that draws lines that separate different areas in the map of pairwise distances (it must be customised to the case under study).
    '''
    # span 6
    plt.axvline(x=5.5, color='k', linewidth=2)
    plt.axhline(y=5.5, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6, color='k', linewidth=2)
    plt.axhline(y=5.5+6, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3, color='k', linewidth=2)

    plt.axvline(x=5.5+6+6+5+3+4, color='r', linewidth=4)
    plt.axhline(y=5.5+6+6+5+3+4, color='r', linewidth=4)
    
    # span 7
    plt.axvline(x=5.5+6+6+5+3+4+6, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4, color='r', linewidth=4)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4, color='r', linewidth=4)
    
    # span 8
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4+2, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4+2, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4+2+4, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4+2+4, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4, color='r', linewidth=4)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4, color='r', linewidth=4)
    
    # north pylon
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4+1, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4+1, color='k', linewidth=2)
    
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4+1+6, color='r', linewidth=4)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4+1+6, color='r', linewidth=4)
    
    # north main span  
    plt.axvline(x=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4+1+6+1, color='k', linewidth=2)
    plt.axhline(y=5.5+6+6+5+3+4+6+5+4+2+4+2+4+2+4+1+6+1, color='k', linewidth=2)
