# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 12:26:30 2014

Module that provides methods to use scipy kmeans implementation.

@author: adiez
"""

# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial import distance
import matplotlib.cm as cmx
from pandas import DataFrame
from scipy.spatial.distance import cdist,pdist
from matplotlib import cm
from scipy.cluster.vq import kmeans
import random
from scipy import stats
from sklearn.decomposition import PCA
from itertools import repeat


def get_magicNumber(X, K_MAX=101):
    '''
    Method that estimates the optimal number of clusters, k, given a training data set.
        Inputs:
            - X: training data set (matrix [m,n])
            - K_MAX: maximum number of clusters considered            
    '''
    ##### cluster data into K=1..K_MAX clusters #####    
    KK = range(1,K_MAX+1,1)    
    KM = [kmeans(X,k) for k in KK]
    centroids = [cent for (cent,var) in KM]
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]

    tot_withinss = [sum(d**2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(X)**2)/X.shape[0]       # The total sum of squares
    betweenss = totss - tot_withinss          # The between-cluster sum of squares

    ##### plots #####
    #kIdx = 9        # K=10
    clr = cm.spectral( np.linspace(0,1,10) ).tolist()
    mrk = 'os^p<dvh8>+x.'

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(KK, betweenss/totss*100, 'b*-')
    #ax.plot(KK[kIdx], betweenss[kIdx]/totss*100, marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    ax.set_ylim((0,100))
    ax.set_xlim((0,K_MAX))
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')
    plt.title('Elbow for KMeans clustering')
    
    plt.show()


def getDistance(a, b):
    '''
    It calculated the euclidean distance based on the scipy implementation.
        Inputs:
            - a: data array [m]
            - b: data array [m]
        Outputs: resulting euclidean distance between a and b
    '''
    return distance.euclidean(a,b)


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    '''        
        Kernel Density Estimation with Scikit-learn
    '''
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


def calculate_compactness(data, labels, centroids):
    """Calculate compactness of data grouped by labels"""
    distinct_labels = list()
    map(lambda x: not x in distinct_labels and distinct_labels.append(x), labels)
    # total data variance
    # data_var = np.var(data)
    compactness = []
    for i in range(len(centroids)):
        tmp_data = data[labels == distinct_labels[i]]         
        group_compactness = np.sum([getDistance(tmp_data[j], centroids[i]) for j in range(len(tmp_data))])        
        group_compactness = np.sqrt(group_compactness / len(tmp_data))
        compactness.append(group_compactness)            
    
    return compactness


def k_means(data, n_clusters=5):       
    '''
    sklearn based KMeans method.
        Inputs:
            - data: training data set containing the events to be processed (matrix [m,n])            
            - n_clusters: number of clusters to be used
        Outputs: (Z, centroids, kmeans) array containing to which formed cluster every event in the training data set belongs to (array [m]), the centroids or mean values (matrix [m, n_clusters]) and the resulting kmeans object
    '''
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=20)
    # testing kmedians and fuzzy kmeans...
    #kmeans = KMedians(k=n_clusters)
    # kmeans = FuzzyKMeans(k=7, m=2)
    kmeans.fit(data)
    
    # Obtain labels for each point in mesh. Use last trained model.
    #Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])    
    #Z = kmeans.predict(data) 
    Z = kmeans.labels_ 
    # obtain centroids    
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_      
    print('Sum of distances of events to their closest cluster center: ' + str(inertia))
    
    return Z, centroids, kmeans


def draw_clustering_results(data, index, samples_distribution, centroids, Z, njoints=6):     
    '''
    Method to show clustering results. 
        Inputs:
            - data: training data set containing the events to be processed (matrix [m,n])            
            - index: the list of events' labels (joints), (array [m])
            - samples_distribution: the labels distribution, one row per set of labels that will have the same color in the graph
            - centroids: the centroids (matrix [m, n_clusters])
            - Z: array containing to which formed cluster every event in the training data set belongs to (array [m])
        Outputs: each cluster is represented by two graphs, above the centroid and the std of the events grouped in the cluster and below the events distribution
    '''
    joints = []
    joint_colors = []
    # set colors based on groups of joints
    # samples_distribution = pd.value_counts(pd.Series(index[:,0]))        
    for i in range(len(samples_distribution)):        
    #for i in samples_distribution:        
        #if (i != 6):
        color = generate_new_color(joint_colors, pastel_factor = 0.1)
        for j in range(len(samples_distribution[i])):                        
            if (list(index).count(samples_distribution[i][j]) > 0):
            #if (list(index).count(i) > 0):
                joint_colors.append(color)
                joints.append(samples_distribution[i][j])
                #joints.append(i)
    
    color_list = np.array(joint_colors)
    # create figures pannel
    #fig1 = plt.figure(1)
    #fig2 = plt.figure(2)
    #nr = n_clusters/2 + n_clusters%2bx = fig1.add_subplot(nr,nc,i+1)        
    #nc = n_clusters/2 
    #fig1.subplots_adjust(left=0.02,top=0.98,bottom=0.02,right=0.98,wspace=0.1,hspace=0.1)        
    for i in range(len(centroids)):                   
        # create chart
        fig1 = plt.figure(figsize=(24,12))
        #bx = fig1.add_subplot(nr,nc,i+1)        
        bx = fig1.add_subplot(211)
                          
        # samples_in_cluster = index[Z == i]
        samples_distribution_in_cluster = pd.value_counts(pd.Series(index[Z == i]))
        data_tmp = data[Z == i]               
        
        # index_tmp = index[Z == Z[i]]
        # data_tmp = np.column_stack((data_tmp, index_tmp))
        # df = DataFrame(data = data_tmp)
        # draw data
        # parallel_coordinates(df, len(data_tmp[0,:])-1)                        
        #p_x = np.arange(0, len(data[0, :]), 1)
        if njoints == 6:
            p_x = np.arange(0, 375/2, .625) # 6
        elif njoints == 71:
            p_x = np.arange(0, (250/2)+.5, .5) # 71
        else:
            print ('Wrong number of joints. Please check.')
            return
        # plot signals
        #bx = fig1.add_subplot(211)            
        #for j in range(len(data_tmp)):              
        #    plt.plot(p_x, data_tmp[j, :], c = color_list[samples_distribution == samples_in_cluster[j]][0])                            
        bx.errorbar(p_x, np.mean(data_tmp,axis=0), np.std(data_tmp,axis=0), label='std of ' + str(len(data_tmp)) + ' events')
        if njoints == 6:
            bx.set_xlim(0,375/2-.625) # 6        
        elif njoints == 71:
            bx.set_xlim(0,125) # 71
        else:
            print ('Wrong number of joints. Please check.')
            return
        # check dominant class                
        # elements = np.column_stack((pd.value_counts(pd.Series(index[Z == Z[i]][:,0])), color_list))        
        bx.plot(p_x, centroids[i, :], label='centroid of cluster ' + str(i), linewidth=2, color='black')                   
        bx.grid()  
        bx.legend(fontsize=24)
        bx.set_xlabel('frequency (Hz)', fontsize=24)
        bx.set_ylabel('amplitude', fontsize=24)# ($V=|A_{i}|-|A_{r}|$)')
        bx.tick_params(labelsize=20)

        # plot histogram
        cx = fig1.add_subplot(212)               
        #cx = fig2.add_subplot(nr,nc,i+1)
        cx.set_ylabel('percentage of events per joint', fontsize=24)
        #cx.set_title('Cluster ' + str(i) + ': joints distribution from North to South')
        # bar_colors = [(color_list[samples_distribution == samples_distribution_in_cluster.index[i]][0]) for i in range(len(samples_distribution_in_cluster))]
        
        # order data by joint position        
        tmp = []
        for i in range(len(joints)):                    
            count = samples_distribution_in_cluster.values[samples_distribution_in_cluster.index == joints[i]]
            if count > 0: 
                tmp.append(count * 100 / list(index).count(joints[i])) # percentage of events within joint
            else:
                tmp.append(0.0)
                
        # normalise values
        joints_distribution = np.array(tmp).ravel()#[tmp[i] / np.sum(tmp) for i in range(len(tmp))] 
        ind = np.arange(len(joints))                # the x locations for the groups
        width = 1                      # the width of the bars
        cx.bar(ind, 
               joints_distribution,
               #scale(joints_distribution, axis=0, with_mean=True, with_std=True, copy=True), 
               width = width,
               color = color_list)
        # axes and labels
        cx.set_xlim(-width,len(ind)+width) 
        cx.set_ylim(0, 110)#np.max(joints_distribution)+.5)
        xTickMarks = ['Joint ' + str(int(joints[i])) for i in range(len(joints_distribution))]
        cx.set_xticks(range(len(xTickMarks)))
        cx.set_xticklabels(xTickMarks, minor=False, rotation=45, ha='right')
        cx.grid()        
       
        # add kde
#        kde = []
#        c = 0
#        for i in tmp:          
#            kde.extend(repeat(c, int(i)))
#            c += 1
#            
#        density = stats.kde.gaussian_kde(kde)            
#        cx.plot(p_x, density(p_x), 'r--', label='gaussian kde', linewidth=3, alpha=0.5)        
#        cx.legend()

    # draw centroids in the same plot
    fig3 = plt.figure(figsize=(24,12))
    dx = fig3.add_subplot(211)  
    dx.set_title('Cluters centroids')
    color_list = []
    color_points = []
    for i in range(len(centroids)):
        color = generate_new_color(joint_colors, pastel_factor = 0.1)
        color_list.append(color)                        
        joint_colors.append(color)
        color_points.extend(repeat(color, np.count_nonzero(Z == i)))
    
    for i in range(len(centroids)):
        dx.plot(p_x, centroids[i, :], label='centroid ' + str(i), linewidth=2, color=color_list[i])                   

    dx.grid()  
    dx.legend(fontsize=24)
    dx.tick_params(labelsize=20)
    ###############################################################################        
    # draw signals as 2-d by pca reduction
    fig3.add_subplot(212)
    #dx.set_title('2-d PCA based clusters distribution')
    reduced_data = PCA(n_components=2).fit_transform(data)        
    #dx.scatter(pca[:,0], pca[:,1], c=color_points)    
    #dx.grid()  
    #dx.legend('cluster ' + str(i)) for i in range(len(centroids))
    kmeans_pca = KMeans(init='k-means++', n_clusters=len(centroids), n_init=10)
    kmeans_pca.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    # h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))

    # Obtain labels for each point in mesh. Use last trained model.
    z = kmeans_pca.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Put the result into a color plot
    z = z.reshape(xx.shape)     
    plt.clf
    #cmap = plt.get_cmap()
    plt.imshow(z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', color_list),#plt.cm.Paired,
           aspect='auto', origin='lower')

    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=color_points)
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    reduced_centroids = kmeans_pca.cluster_centers_
    plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.title('K-means clustering on 2D PCA-reduced data', fontsize=20)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    
    plt.show()         



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
