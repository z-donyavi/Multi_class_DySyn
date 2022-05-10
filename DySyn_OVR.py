# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:43:28 2022

@author: Zahra
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn import preprocessing
# from quantifiers.DySyn import DySyn
#import qnt_utils as qntu
from sklearn.ensemble import RandomForestClassifier
import qnt_utils as qntu
import pdb
import sys
import os


# Test_class_relative = [0.05, 0.05, 0.25, 0.65]
# test_distributions = dict()
# test_distributions[3] = np.array(
#     [[0.1, 0.7, 0.2], [0.55, 0.1, 0.35], [0.35, 0.55, 0.1], [0.4, 0.25, 0.35], [0., 0.05, 0.95]])
# test_distributions[4] = np.array(
#     [[0.65, 0.25, 0.05, 0.05], [0.2, 0.25, 0.3, 0.25], [0.45, 0.15, 0.2, 0.2], [0.2, 0, 0, 0.8],
#      [0.3, 0.25, 0.35, 0.1]])
# test_distributions[5] = np.array(
#     [[0.15, 0.1, 0.65, 0.1, 0], [0.45, 0.1, 0.3, 0.05, 0.1], [0.2, 0.25, 0.25, 0.1, 0.2], [0.35, 0.05, 0.05, 0.05, 0.5],
#      [0.05, 0.25, 0.15, 0.15, 0.4]])


# exp_name = 'wine'
# folder = "C:/Users/Zahra/OneDrive - UNSW/Desktop/Implimentations/quantifiers4python-main"
# dts = pd.read_csv(folder + '/dataset/%s'% exp_name +'/%s' % exp_name + '.csv', index_col=False, engine='python')

# X = dts.drop(['class'], axis=1)
# y = dts['class']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# X_train = X_train.reset_index(drop=True)
# y_train = y_train.reset_index(drop=True)

# C = np.unique(y_train)
# a = len(C)+1
# y = np.where(y_train == 0, a, y_train) # replace class 0 with another number in dataset to prevent confilict with negative class

# #Binarization for training data
# lb = preprocessing.LabelBinarizer()
# y = pd.DataFrame(lb.fit_transform(y))
# binary_train = {}
# for i in range(len(C)):
#     D = X_train.copy()
#     D.loc[:,'class'] = y[i]
#     binary_train[i] = D

# #X_test = MaxAbsScaler().fit_transform(X_test)

# # y = y_test.values
# # N = len(y)

# # split test data based on their class and keep their indexes in y_idx for sampling
# y_cts = np.unique(y_test, return_counts=True)
# Y = y_cts[0]
# n_classes = len(Y)
# y_cts = y_cts[1]
# y_idx = [np.where(y_test == l)[0] for l in Y]

# #create a sample with pre-defined class prevalences

# test_dist = test_distributions[len(C)]
# z = test_dist[0][0]
# #for te_dist in range(np.shape(test_dist)[0]): #create test samples with different test distributions

# sample_test1_1 = np.random.choice(y_idx[0], 17, replace = False)
# sample_test1_2 = np.random.choice(y_idx[1], 17, replace = False)
# sample_test1_3 = np.random.choice(y_idx[2], 87, replace = False)
# sample_test1_4 = np.random.choice(y_idx[3], 226, replace = False)

# sample_xtest1_1 = X_test.iloc[sample_test1_1]
# sample_xtest1_2 = X_test.iloc[sample_test1_2]
# sample_xtest1_3 = X_test.iloc[sample_test1_3]
# sample_xtest1_4 = X_test.iloc[sample_test1_4]

# sample_ytest1_1 = y_test.iloc[sample_test1_1]
# sample_ytest1_2 = y_test.iloc[sample_test1_2]
# sample_ytest1_3 = y_test.iloc[sample_test1_3]
# sample_ytest1_4 = y_test.iloc[sample_test1_4]


# sample_xtest1_1['class'] = sample_ytest1_1
# sample_xtest1_2['class'] = sample_ytest1_2
# sample_xtest1_3['class'] = sample_ytest1_3
# sample_xtest1_4['class'] = sample_ytest1_4

# sample1 = pd.DataFrame()
# sample1 = sample1.append(sample_xtest1_1)
# sample1 = sample1.append(sample_xtest1_2)
# sample1 = sample1.append(sample_xtest1_3)
# sample1 = sample1.append(sample_xtest1_4)



# #reset indexes
# z = sample1.reset_index(drop=True)
# X = z.drop(['class'], axis=1)
# y = z['class']

# C = np.unique(y)
# a = len(C)+1
# y = np.where(y == 0, a, y) # replace class 0 with another number in dataset to prevent confilict with negative class

# #Binarization
# lb = preprocessing.LabelBinarizer()
# y = pd.DataFrame(lb.fit_transform(y))
# binary_ts = {}
# for i in range(len(C)):
#     D = z.copy()
#     D['class'] = y[i]
#     binary_ts[i] = D


def DySyn(test_scores):
     
    MF = [0.1,0.3,0.5,0.7,0.9] #These are the mergim factor used to search the best match
    result  = []
    for mfi in MF:        
        scores = MoSS(1000, 0.5, mfi)        
        prop = DyS(scores[scores['label']=='1']['score'], scores[scores['label']=='2']['score'], test_scores, measure = 'topsoe')
        result.append(prop)                                           
                        
    pos_prop = round(np.median(result),2)
    print(pos_prop)
    return pos_prop


def MoSS(n, alpha, m):
    p_score = np.random.uniform(0,1,int(n*alpha))**m
    n_score = 1-np.random.uniform(0,1,int(n*(1- alpha)))**m    
    scores  = pd.concat([pd.DataFrame(np.append(p_score, n_score)), pd.DataFrame(np.append(['1']*len(p_score), ['2']*len(n_score)))], axis=1)
    scores.columns = ['score', 'label']
    return scores


def DyS(pos_scores, neg_scores, test_scores, measure='topose'):
    
    bin_size = np.linspace(2,20,10)  #[10,20] range(10,111,10) #creating bins from 2 to 10 with step size 2
    bin_size = np.append(bin_size, 30)
    
    result  = []
    for bins in bin_size:
        #....Creating Histograms bins score\counts for validation and test set...............
        
        p_bin_count = qntu.getHist(pos_scores, bins)
        n_bin_count = qntu.getHist(neg_scores, bins)
        te_bin_count = qntu.getHist(test_scores, bins)
        
        def f(x):            
            return(qntu.DyS_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = measure))
    
        result.append(qntu.TernarySearch(0, 1, f))                                           
                        
    pos_prop = round(np.median(result),2)
    return pos_prop
    

#j = 0
def run_setup_DySyn(X, y, train_idx, test_idx):
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    C = np.unique(y_train)
    a = len(C)+1
    # replace class 0 with another number in dataset to prevent confilict with negative class
    y_train = np.where(y_train == 0, a, y_train) 
    y_test = np.where(y_test == 0, a, y_test)

    #Binarization for training data
    lb = preprocessing.LabelBinarizer()
    y_tr = pd.DataFrame(lb.fit_transform(y_train))
    # binary_train = []
    # for i in range(len(C)):
    #     # tr_b = X_train.copy()
    #     # tr_b[:,'class'] = y_tr[i]
    #     binary_train[i][0] = X_train.copy()
    #     binary_train[i][1] = y_tr[i]
   
    

    #Binarization for test data
    # lb = preprocessing.LabelBinarizer()
    # y_ts = pd.DataFrame(lb.fit_transform(y_test))
    # binary_ts = {}
    # for i in range(len(C)):
    #     te_b = X_test.copy()
    #     te_b['class'] = y_ts[i]
    #     binary_ts[i] = te_b
    
    pred_pos_prop = pd.DataFrame()
    # calcultd_pos_prop = pd.DataFrame()
    
    
    for j in range(len(C)):
        # X_tr = binary_train[j].drop(['class'], axis=1)
        # y_tr = binary_train[j]['class']
        rf_clf = RandomForestClassifier(n_estimators=200)
        rf_clf.fit(X_train, y_tr[j])  
   
        # sample_test = binary_ts[j]
        # test_label = sample_test["class"]
        
        # test_sample = sample_test.drop(["class"], axis=1)  #dropping class label columns
        te_scores = rf_clf.predict_proba(X_test)[:, 1]  #estimating test sample scores
      
    
        # n_pos_sample_test = list(test_label).count(1) #Counting num of actual positives in test sample
        # calcultd_pos_prop.loc[j, 0] = round(n_pos_sample_test/len(sample_test), 2) #actual pos class prevalence in generated sample
        
        pos = DySyn(te_scores)
       # pos_prop.loc[:, 'pred_pos'] = pos
        pred_pos_prop.loc[j, 0] = pos
    
    p_sum = pred_pos_prop.sum()
    pred_pos = np.array(pred_pos_prop/p_sum)
    pred_pos = pred_pos.flatten()
     
    return pred_pos    
  
  # abs_error = pd.DataFrame()
       
  #   for k in range(len(C)):
  #       abs_error.loc[k, 0] = round(abs(calcultd_pos_prop.loc[k, 0] - pred_pos_prop.loc[k, 0]),2) #absolute error
    #error = round(calcultd_pos_prop - pred_pos_prop , 2)     # simple error Biasness
    # table = table.append(pd.DataFrame([(iter + 1),sample_size,alpha,calcultd_pos_prop,pred_pos_prop,abs_error,error,quantifier]).T)
    






   
# z = X_test.iloc[sample_test1_1]
# w = y_test.iloc[sample_test1_1]