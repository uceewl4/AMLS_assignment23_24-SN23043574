import numpy as np
import os
import cv2
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class Baselines:

    def __init__(self, method=None):

        self.method = method
        if method == "LR":
            self.model = LogisticRegression(C=0.0001,solver="liblinear") 
        elif method == "KNN":  
            self.model = KNeighborsClassifier()
        elif method == "SVM":
            self.model = svm.SVC(kernel='poly') 
        elif method == "DT":  
            self.model = DecisionTreeClassifier(criterion='entropy')
        elif method == "NB": 
            self.model = MultinomialNB()
        elif method == "RF":  
            self.model = RandomForestClassifier(criterion='entropy',verbose=1)
        elif method == "ABC":  
            self.model = AdaBoostClassifier()
        
        
    def train(self, Xtrain, ytrain, Xval, yval, gridSearch=False):

        print(f"Start training for {self.method}......")
        self.model.fit(Xtrain, ytrain)  
        print(f"Finish training for {self.method}.")

        if gridSearch:  
            print(f"Start tuning(cross-validation) for {self.method}......")
            if self.method == "KNN":
                params = [{"n_neighbors": [i for i in range(1,30,2)]}]
            if self.method == "DT":
                params = [{"max_leaf_nodes": [i for i in range(20,100,5)]}]
            if self.method == "RF":  # very slow need to notify TA
                params = [{"n_estimators": [120, 140, 160, 180, 200], "max_depth": [8, 10, 12, 14, 16]}]
            if self.method == "ABC":
                params = [{"n_estimators": [50, 75, 100, 125, 150, 175], "learning_rate": [0.001, 0.01, 0.1, 1]}]
            grid = GridSearchCV(self.model, params, cv=10, scoring="accuracy")

            grid.fit(np.concatenate((Xtrain,Xval),axis=0), ytrain+yval)
            print(grid.best_params_)
            self.model = grid.best_estimator_

            print(f"Finish tuning(cross-validation) for {self.method}.")
            return grid.cv_results_


    def test(self, Xtrain, ytrain, Xval, yval, Xtest):

        print(f"Start testing for {self.method}......")
        self.model.fit(np.concatenate((Xtrain,Xval),axis=0),ytrain+yval)
        pred_test = self.model.predict(Xtest)
        pred_train = self.model.predict(Xtrain)
        pred_val = self.model.predict(Xval)
        
        print(f"Finish testing for {self.method}.")
        
        return pred_train, pred_val, pred_test



