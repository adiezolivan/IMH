# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:40:45 2015

Module containing outliers removal algorithm.

@author: adiez
"""

from sklearn.neighbors import KDTree
import numpy as np
import pandas as pd


def outliers_removal(data, index, k=500, normalityLoops=20, anomaly_threshold=2):
    '''
    Outliers removal method for data cleaning and resampling, which for each joint eliminates those events that are far from the mean of the energy of the joint's events.
        Inputs:
            - data: training data set containing the events to be processed (matrix [m,n])
            - index: the list of events' labels (joints), (array [m])
            - k: number of neighbours to be considered by the KDTree and maximum number of resulting events per label
            - normalityLoops: stopping criteria
            - anomaly_threshold: std from the mean to be used
        Outputs: (reduced_data, reduced_index, samples_distribution) objects containing the cleaned data, the corresponding labels and the final label distribution           
    '''    
    # when adding joints locations to training data
    #    samples_distribution = []
    #    for i in range(len(joints)):
    #        for j in range(len(joints[i])):
    #            samples_distribution.append(joints[i][j])
    #samples_distribution = pd.value_counts(pd.Series(index[:,0])).index 
    samples_distribution = pd.value_counts(pd.Series(index))
    counter = 0
    # outliers removal process, per joint
    #for i in range(len(samples_distribution)):        
    for i in sorted(samples_distribution.index):                
        #joint_data = data[index[:,0] == samples_distribution[i]] 
        joint_data = data[index == i] 
        
        if len(joint_data) == 0:
            continue
        
        counter += 1
        
        energy_data = joint_data     
        # energy based cleaning values
        energy_data=[np.sum(np.abs(energy_data[m])**2) for m in range(len(energy_data))]
        #        for j in samples_distribution[i]:
        #            if (j != samples_distribution[i][0]):
        #                joint_dist = np.vstack((joint_dist, data[index[:,0] == j]))
        #                idx_filter.extend(repeat(j, len(data[index[:,0] == j])))                
        
        print('\nJoint %i' % i)
        mean_current = np.Inf        
        # normality loops                    
        for normalityLoop in range(1, normalityLoops+1):                          
            print('Iteration %i' % normalityLoop)                        
            energy_data_tmp = []      
            reduced_data_tmp = []
            mean_old = mean_current            
            idx_filter = []

            if len(energy_data) > 0: # and len(joint_dist) >= portion:                                    
                # calculate mean of data
                mean = np.mean(energy_data, axis=0)                
                #joint_dist = np.vstack((mean, joint_dist))
                energy_data = np.append(mean, energy_data)           
                
                # use KDTree to extract k nearest neighbours from the mean        
                energy_data = np.array(energy_data).reshape(energy_data.size, 1)
                kdt = KDTree(energy_data, leaf_size=30, metric='euclidean')

                n = k                                                
                if len(energy_data) < k:                
                    n = len(energy_data)          
                    
                print(n)
                # get the n closest points     
                dx, idx = kdt.query(energy_data[0].reshape(-1, 1), k = n)   
                mean_current = np.mean(dx)
                max_distance = np.max(dx)     
                heuristic = max_distance / mean_current                                                        
                
                # remove those points far from the mean
                c = 0
                normal_samples = 0
                for j in idx[0]:        
                    if j > 0:
                        # check if current point is close to the mean
                        if (dx[0][c] <= np.mean(dx) + anomaly_threshold * np.std(dx)):
                            normal_samples += 1
                            energy_data_tmp.append(energy_data[j])                                                        
                            reduced_data_tmp.append(joint_data[j-1]) # mean value has not been added to original data    
                            idx_filter.append(j-1) 
                                
                    c += 1                                                                                        
                                
                print('Average distance: %f' % mean_current)
                print('Max distance: %f' % max_distance)                
                
                if (normalityLoop == 1):                
                    tmp = dx[0:int(0.9*len(dx))]
                    max_threshold = np.mean(tmp) + 0.5 * np.std(tmp)
                    print('Setting threshold to  %f' % max_threshold)
                
                # stopping criteria
#                if len(joint_dist) < k:
#                    print('Group of joints ' + str(i) + ' under k. Exiting')
#                    break
#                if (dx[0][len(dx)] <= np.mean(dx) + 2 * np.std(dx)):
#                    print('Convergence condition met for group ' + str(i) + '. Exiting')
#                    break
                if (normal_samples > 0):
                    print('Number of samples removed: %i' % (len(energy_data)-normal_samples))
                else:
                    print('No normal samples added. Exiting')
                    break
                if max_distance < max_threshold:
                    print('Maximum distance is small enough. Exiting')
                    break                
                if heuristic < 1.1:
                    print('Deviation threshold reached. Exiting')
                    break
                elif mean_current > mean_old:
                    print('Distance to mean increasing. Exiting')
                    break
                elif normalityLoop == normalityLoops:
                    print('Number of iterations reached. Exiting')
                    break
                
                # original data
                #joint_data = reduced_data_tmp
                # calculated energy
                energy_data = energy_data_tmp    
        
            # no data for current joint
            else:
                print('Empty joint!. Exiting')
                break        
        
        joint_data = reduced_data_tmp
        if len(joint_data) > 0:
            if counter == 1:
                reduced_data = joint_data#joint_dist
                reduced_index = [i] * len(joint_data)             
                removed_data = np.delete(data[index==i], np.array(idx_filter), axis=0)
                removed_index = [samples_distribution[i]] * len(removed_data)                 
            else:                
                reduced_data = np.vstack((reduced_data, joint_data))
                reduced_index = np.hstack((reduced_index, [i] * len(joint_data)))                  
                tmp = np.delete(data[index==i], np.array(idx_filter), axis=0)
                removed_data = np.vstack((removed_data, tmp))
                removed_index = np.hstack((removed_index, [samples_distribution[i]] * len(tmp)))                      
                
    reduced_data = np.array(reduced_data)
    reduced_index = np.array(reduced_index)                
    
    return reduced_data, reduced_index, samples_distribution, removed_data, removed_index