'''
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2023-11-21 11:45:45
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2023-12-16 22:20:48
FilePath: /AMLS_assignment23_24-SN23043574/B/models/KMeans.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- encoding: utf-8 -*-
'''
@File    :   KMeans.py
@Time    :   2023/12/16 22:19:58
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@Author  :   Wenrui Li
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :    This file is used for the initialization, training and testing proces of K-Means clustering method.
'''

# here put the import lib

import numpy as np
from sklearn.cluster import KMeans as kmeans

class KMeans:
    
    '''
    description: model initialization
    param {*} self
    '''  
    def __init__(self):
        self.model = kmeans(n_clusters=9) # 9 clusters

    '''
    description: training for K-Means clustering model
    param {*} self
    param {*} Xtrain: train images
    param {*} ytrain: train ground truth labels 
    '''  
    def train(self, Xtrain, ytrain):

        print(f"Start training for KMeans......")
        self.model.fit(Xtrain, ytrain)  
        print(f"Finish training for KMeans.")
    
    '''
    description: testing for K-Means clustering model
    param {*} self
    param {*} Xtrain: train images
    param {*} Xval: validation images
    param {*} Xtest: test images
    return {*}: predicted clusters for train/validation/test
    '''    
    def test(self, Xtrain, Xval, Xtest):
        pred_train = self.model.predict(Xtrain)

        print(f"Start evaluating for KMeans......")
        pred_val = self.model.predict(Xval)
        print(f"Finish evaluating for KMeans.")

        print(f"Start testing for KMeans......")
        pred_test = self.model.predict(Xtest)
        print(f"Finish testing for KMeans.")
        
        return pred_train, pred_val, pred_test



