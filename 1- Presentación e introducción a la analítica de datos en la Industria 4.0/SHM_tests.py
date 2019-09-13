# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 09:37:11 2014

Module to start playing with the SHM data.

@author: adiez
"""

from pandas import DataFrame, read_csv
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import time
from datetime import datetime
from scipy.fftpack import fft
import scipy.signal.signaltools as sigtool
import sklearn.preprocessing as pre
from sklearn.cross_validation import train_test_split
import data_processing
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


def load_118nodes_data(csv='118_nodes_01_10_2014_07_10_2014.csv', sensor_id="2"):
    '''
    Method that loads sensor data related to 118 nodes test (it must be customised depending on the file format, this is just a template).
        Inputs:
            - csv: the path and csv file name that contains the data            
            - sensor_id: the sensor 
        Outputs: loaded data is exported to different csv files
    '''   
    # variables to be used
    data_x = []
    data_y = []
    data_z = []
    data_v1 = []
    data_v2 = []
    index = []
    joint = 0
    joint_tmp = joint                
    date = 0
    date_tmp = date                
    tmp_x = []        
    tmp_y = []
    tmp_z = []
    tmp_v1 = []
    tmp_v2 = []
    counter = 0
    # take the joint id + sensor id
    #with open('20_nodes_01_08_2014_07_08_2014.csv', 'rb') as f:104_nodes_01_08_2014_07_08_2014
    with open(csv, 'r') as f:
        for line in f:                        
            s = line.split(',')                        
            if (len(s) > 1):    
                sensor = s[2]           
                # check sensor id
                if str(sensor[-1:]) == sensor_id:                    
                    joint_tmp = s[3]#sensor[:-1]
                    date_tmp = s[0]                    
                    date_tmp = totimestamp(datetime.strptime(date_tmp[1:-1], "%Y-%m-%d %H:%M:%S.%f"))
                    # check if another event is coming
                    if ((joint != joint_tmp and joint != 0) or (date != date_tmp and date != 0)):
                        # index will be date + joint id
                        #index.append(str(date) + '_' + joint)                        
                        if (counter == 0):                            
                            # axis x
                            #data_x = tmp_x
                            #tmp_x = []        
                            # axis y
                            #data_y = tmp_y
                            #tmp_y = []
                            # axis z
                            #data_z = tmp_z
                            #tmp_z = []
                            # v1
                            data_v1 = tmp_v1
                            tmp_v1 = []
                            index.append(int(joint))
                            # v2
                            #data_v2 = tmp_v2
                            #tmp_v2 = []                                                        
                        else:                                   
                            # axis x
                            #data_x = np.vstack((data_x, tmp_x))
                            #tmp_x = []        
                            # axis y
                            #data_y = np.vstack((data_y, tmp_y))
                            #tmp_y = []
                            # axis z
                            #data_z = np.vstack((data_z, tmp_z))
                            #tmp_z = []
                            # v1
                            if counter > 1: 
                                if len(tmp_v1) == 500:
                                    data_v1 = np.vstack((data_v1, tmp_v1))
                                    index.append(int(joint))
                            
                            tmp_v1 = []                            
                            # v2
                            #data_v2 = np.vstack((data_v2, tmp_v2))
                            #tmp_v2 = []                                                                                                          
                        print(counter)
                        counter += 1
                    # axis x
                    #tmp_x.append(float(s[4]))
                    # axis y
                    #tmp_y.append(float(s[5]))
                    # axis z
                    #tmp_z.append(float(s[6]))
                    # axis v1
                    tmp_v1.append(float(s[7]))
                    # axis v2
                    #tmp_v2.append(float(s[8]))
                    # update joint    
                    joint = joint_tmp                
                    date = date_tmp                        
    
    print('done!')
    index = np.array(index)    
    # set indexes
    #dfx = DataFrame(data=data_x,index=index)
    #dfy = DataFrame(data=data_y,index=index)
    #dfz = DataFrame(data=data_z,index=index)
    #dfv1 = DataFrame(data=data_v1,index=index)
    #dfv2 = DataFrame(data=data_v2,index=index)
    # export data to csv
    #data_processing.csv_writer(index, 'joints_118nodes_oct.csv')
    #csv_writer(data_x, 'data_x_118nodes_oct.csv')
    #csv_writer(data_y, 'data_y_118nodes_oct.csv')
    #csv_writer(data_z, 'data_z_118nodes_oct.csv')
    #data_processing.csv_writer(data_v1, 'data_v1_118nodes_oct.csv')
    #csv_writer(data_v2, 'data_v2_118nodes_oct.csv')
    joints_dist = [[17,24,25,26,21,22],[13,20,16,14,15,11],[18,12,19,27,28,29],[30,31,33,34,35],[37,39,41],[42,43,45,47],[143,142,141,140,139,138],[136,135,134,133,131],[129,128,126,125],[123,121],[111,110,109,107],[177,175],[174,173,172,169],[165,164],[162,161,160,159],[82],[78,77,75,74,73,72],[99]]
    #data = dfx.drop(dfx.index[pd.isnull(dfx).any(1).nonzero()[0]])
    #data = data[~np.isnan(data).any(axis=1)]
    #data=scale(data)
    #data=np.fft(data.values)
    return data_v1, index, joints_dist


def load_6joints(csv_file="SHM_6_joints_18_sensors_Aug12_Oct12.csv"):
    '''
    Method that loads sensor data related to 6 joints test (it must be customised depending on the file format, this is just a template).
        Inputs:
            - csv: the path and csv file name that contains the data            
        Outputs: (datav1, joints) loaded v1 data (matrix [m,n]) and the list of joints (array [m])
    '''       
    c = -1            
    datav1 = []
    joints = []    
    for line in open(csv_file):        
        c += 1
        #print(c)    
        if (6329 < c <= 12653):        
            joints.append(1)
            csv_row = line.split(',')[0:599]
            tmp = [float(i) for i in csv_row]
            datav1.append(np.array(tmp))
        elif (26215 < c <= 33452):        
            joints.append(2)
            csv_row = line.split(',')[0:599]
            tmp = [float(i) for i in csv_row]
            datav1.append(np.array(tmp))
        elif (45675 < c <= 50659):        
            joints.append(3)
            csv_row = line.split(',')[0:599]
            tmp = [float(i) for i in csv_row]
            datav1.append(np.array(tmp))
        elif (62529 < c <= 69415):        
            joints.append(4)
            csv_row = line.split(',')[0:599]
            tmp = [float(i) for i in csv_row]
            datav1.append(np.array(tmp))
        elif (83019 < c <= 89734):        
            joints.append(5)
            csv_row = line.split(',')[0:599]
            tmp = [float(i) for i in csv_row]
            datav1.append(np.array(tmp))
        elif (101251 < c <= 106052):        
            joints.append(6)
            csv_row = line.split(',')[0:599]
            tmp = [float(i) for i in csv_row]
            datav1.append(np.array(tmp))
        else:
            continue    
    
    return datav1, joints


def tests():
    '''
    Different operations that can be directly typed in console.
    '''
    # code for loading and processing data directly
    datav1 = read_csv("data_v1_118nodes_oct.csv")
    joints = read_csv("joints_118nodes_oct.csv")
    joints = joints[~np.isnan(datav1).any(axis=1)]
    datav1 = datav1[~np.isnan(datav1).any(axis=1)]
    datav1 = scale(datav1.values)
    
    # fft
    #datav1_fft = np.fft.rfft(datav1)
    #datav1_fft = np.abs(datav1_fft)
    
    joints_dist = read_csv("joints_dist.csv")   
    # load joints ids
    joints_id = []
    tmp = []
    for joint in joints_dist['Joint'].values.ravel():
        # group joints by proximity
        if (np.isnan(joint)):
            joints_id.append(np.array(tmp))
            tmp = []
        else:
            tmp.append(joint)
    # add last group of joints
    joints_id.append(np.array(tmp))
    
    joints=np.array(joints)
    joints=joints.reshape(joints.shape[0])
            
    reduced_data, reduced_index, samples_distribution = outliers_removal(datav1, joints, joints_id)

    # fft transform    
    reduced_data_fft = np.fft.rfft(reduced_data)
    reduced_data_fft = np.abs(reduced_data_fft)
    
    # 1d wavelet transform of reduced data    
    reduced_data_dwt = []
    for i in np.arange(0, len(reduced_data)):
        if i == 0:
            reduced_data_dwt = np.array(data_processing.wavelet_transform(reduced_data[i]))
        else:
            reduced_data_dwt = np.vstack((reduced_data_dwt, np.array(data_processing.wavelet_transform(reduced_data[i]))))
    
    # load joints positions
    positions = []
    for pos in joints_dist['Pos'].values.ravel():
        # group joints by proximity
        if (not np.isnan(pos)):
            positions.append(pos)

    reduced_data_fft_pos = []
    # add positions to data, fft case
    for i in np.arange(0, len(reduced_data_fft)):
        if reduced_index[i] < 10:
            continue    
        tmp = list(reduced_data_fft[i])
        tmp.append(joints_dist['Pos'][joints_dist['Joint']==reduced_index[i]])
        if i==0:
            reduced_data_fft_pos = np.array(tmp)
        else:
            reduced_data_fft_pos = np.vstack((reduced_data_fft_pos, np.array(tmp)))
    
    reduced_data_dwt_pos = []
    # add positions to data, dwt case
    for i in np.arange(0, len(reduced_data_dwt)):
        if reduced_index[i] < 10:
            continue    
        tmp = list(reduced_data_dwt[i])
        tmp.append(joints_dist['Pos'][joints_dist['Joint']==reduced_index[i]])
        if i==0:
            reduced_data_dwt_pos = np.array(tmp)
        else:
            reduced_data_dwt_pos = np.vstack((reduced_data_dwt_pos, np.array(tmp)))
        
    # weights specification
    weights=[1]*251
    weights.append(1)
    for i in np.arange(5):
        if i == 0:
            w=weights
        else:        
            w=np.vstack((w,weights))
        
    # log normalisation
    tmp = reduced_data_fft_pos[:,:251]
    k = log10(reduced_data_fft_pos[:,251])   
    # normalise data by columns in range 0..1
    tmp=scale_linear_bycolumn(tmp)
    #removed_data_fft_pos=scale_linear_bycolumn(removed_data_fft_pos)
    tmp = np.column_stack((tmp,np.array(k).T))
    
    mymath=MyMath()
    k=WKMeans(mymath)
    cov_matrix = np.linalg.inv(np.cov(reduced_data_fft_pos, rowvar=0))
    final_u, final_centroids, weights, final_ite, final_dist=k.wk_means(data=reduced_data_fft_pos,k=5,beta=1,init_weights=w,distance='Euclidean',max_ite=100,p=0.1,cov=cov_matrix)
    draw_clustering_results(reduced_data_fft_pos,reduced_index,joints_id,final_centroids,final_u)
    
    means, joints = calculate_joints_means(reduced_data_dwt, reduced_index, joints_id)
    draw_matrix_of_distances(means, joints, distance='Mahalanobis', cov_matrix=cov_matrix)
    
    # test data
    test=train_test_split(reduced_data_fft_pos,test_size=.1,random_state=42)

    
def time_test():    
    import warnings
    path = '/media/adiez/Datos/TIFER/work_tracking/SHM/python_code/'
    warnings.filterwarnings('ignore')
    t0 = time.time()
    # 71 joints
    datav1 = read_csv(path+"data_v1_118nodes_oct.csv")
    joints = read_csv(path+"joints_118nodes_oct.csv")
    joints = joints[~np.isnan(datav1).any(axis=1)]
    datav1 = datav1[~np.isnan(datav1).any(axis=1)]        
#        
#    # 6 joints
    datav1, joints = load_6joints()    
#    
#    # 71 joints
#    #datav1, joints, joints_dist = load_118nodes_data()
#    
    datav1=DataFrame(datav1)
            
    # simulate feature extraction phase
    test = datav1.values[0]
    for i in np.arange(datav1.shape[0]):
        ai=np.sqrt((test*test)+(test*test)+(test*test))
        ai=np.array(ai)
        ai=abs(ai)
        rest=np.array([test[:100],test[:100],test[:100]])
        ar=rest.mean()
        v=ai-abs(ar)    
    
    datav1 = scale(datav1.values)
    joints=np.array(joints)
    joints=joints.reshape(joints.shape[0])    
    t1 = time.time()                

    # 6
    reduced_data, reduced_index, samples_distribution, removed_data, removed_index = outliers_removal(datav1, joints, k=5000, normalityLoops=1)
    # 71
    reduced_data, reduced_index, samples_distribution, removed_data, removed_index = outliers_removal(datav1, joints, k=500, normalityLoops=1)
    t2 = time.time()
    
    # fft transform    
    reduced_data_fft = np.fft.rfft(reduced_data)
    reduced_data_fft = np.abs(reduced_data_fft)
    t3 = time.time()
        
    # weights specification
    #weights=[1]*251
#    weights=[1]*301
#    weights.append(1)
#    for i in np.arange(2):
#        if i == 0:
#            w=weights
#        else:        
#            w=np.vstack((w,weights))
#        
#    mymath=MyMath()
#    k=WKMeans(mymath)    
    joints_dist = [[17,24,25,26,21,22],[13,20,16,14,15,11],[18,12,19,27,28,29],[30,31,33,34,35],[37,39,41],[42,43,45,47],[143,142,141,140,139,138],[136,135,134,133,131],[129,128,126,125],[123,121],[111,110,109,107],[177,175],[174,173,172,169],[165,164],[162,161,160,159],[82],[78,77,75,74,73,72],[99]]
    joints_dist = [[1],[2],[3],[4],[5],[6]]
    # k-means clustering
    #final_u, final_centroids, weights, final_ite, final_dist=k.wk_means(data=reduced_data_fft,k=5,beta=1,init_weights=w,distance='Correlation',max_ite=100,p=0.1)
    # 6
    final_u, final_centroids, final_dist=k_means(data=reduced_data_fft,n_clusters=2)        
    # 71
    kk = []
    peo = []
    joints_dist = [[136],[135],[134],[133],[131]]
    for i in np.arange(len(reduced_index)):
        if reduced_index[i] in [136,135,134,133,131]:
            kk.append(i)
            peo.append(int(reduced_index[i]))
    culo = DataFrame(reduced_data_fft).ix[kk]
    peo = np.array(peo)
    final_u, final_centroids, final_dist=k_means(data=culo,n_clusters=2)        
    draw_clustering_results(culo,peo,joints_dist,final_centroids,final_u)
    
    final_u, final_centroids, final_dist=k_means(data=reduced_data_fft,n_clusters=5)
    draw_clustering_results(reduced_data_fft,reduced_index,joints_dist,final_centroids,final_u)
    # pairwise map
    #means, joints = calculate_joints_means(reduced_data_fft, reduced_index, joints_dist)
    #draw_matrix_of_distances(means, joints, distance='Correlation')
    
    t4 = time.time()
    
    print('[INFO] data preprocessing step ' + str(np.around(t1-t0,2)) + ' secs.')     
    print('[INFO] knn outliers removal step ' + str(np.around(t2-t1,2)) + ' secs.') 
    print('[INFO] fft step ' + str(np.around(t3-t2,2)) + ' secs.') 
    print('[INFO] kmeans clustering step ' + str(np.around(t4-t3,2)) + ' secs.') 
    print('[INFO] total time ' + str(np.around(t4-t0,2)) + ' secs.') 


def draw_signals():
    fig1 = plt.figure()        
    ax = fig1.add_subplot(211)
    #p_x = np.arange(.0025, 1.5, .0025) # 6
    p_x = np.arange(0, 2, .004) # 71
    ax.errorbar(p_x, np.mean(data.loc[reduced_index].values,axis=0), np.std(data.loc[reduced_index].values,axis=0), label='std of training events')
    ax.grid()
    ax.legend(loc=4)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('magnitude ($V=|A_{i}|-|A_{r}|$)')
    ax.set_xlim(.0025,1.5)
    ax.set_ylim(-25,20)
    bx = fig1.add_subplot(212)
    bx.errorbar(p_x, np.mean(data.loc[removed_index].values,axis=0), np.std(data.loc[removed_index].values,axis=0), label='std of filtered events')
    bx.grid()
    bx.legend(loc=4)
    bx.set_xlabel('time (s)')
    bx.set_ylabel('magnitude ($V=|A_{i}|-|A_{r}|$)')
    bx.set_xlim(.0025,1.5)
    bx.set_ylim(-25,20)
    
    fig1 = plt.figure()        
    ax = fig1.add_subplot(211)
    p_x = np.arange(.0025, 1.5, .0025) # 6
    ax.set_xlim(.0025,1.5)
    #p_x = np.arange(0, 2, .004) # 71
    #ax.set_xlim(0,2)
    ax.plot(p_x, data.values[0])
    ax.grid()
    ax.legend()
    ax.set_xlabel('time (s)')
    ax.set_ylabel('magnitude ($V=|A_{i}|-|A_{r}|$)')
    bx = fig1.add_subplot(212)
    p_x = np.arange(0, 375/2, .625) # 6
    bx.set_xlim(0,375/2-.625)
    #p_x = np.arange(0, (250/2)+.5, .5) # 71
    #bx.set_xlim(0,125)
    bx.plot(p_x, np.abs(np.fft.rfft(data.values[0])))
    bx.grid()  
    bx.legend()
    bx.set_xlabel('frequency (Hz)')
    bx.set_ylabel('amplitude ($V=|A_{i}|-|A_{r}|$)')


def time_online(test):
    # instant warning
    t0 = time.time()
    ai=np.sqrt((test*test)+(test*test)+(test*test))
    ai=np.array(ai)
    ai=abs(ai)
    rest=np.array([test[:100],test[:100],test[:100]])
    ar=rest.mean()
    v=ai-abs(ar)    
    test = scale(test) 
    test_data_fft = np.fft.rfft(test)
    test_data_fft = np.abs(test_data_fft)
    for i in np.arange(len(final_centroids)):
        print(mymath.get_distance(final_centroids[i],test_data_fft,distance='Euclidean'))
#    updated_mean=np.array([means[0],test_data_fft])   
#    updated_mean.mean(axis=0)
    
    t1 = time.time()
    print('[INFO] test time ' + str(np.around(t1-t0,5)) + ' secs.') 