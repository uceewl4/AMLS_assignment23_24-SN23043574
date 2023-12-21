# -*- encoding: utf-8 -*-
"""
@File    :   baselines.py
@Time    :   2023/12/16 21:50:32
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for ML baselines including model initialization/selection,
        cross-validation, training and testing process.
"""

# here put the import lib
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier


class Baselines:

    """
    description: This function is used for initialization of Baselines class with instance variables configuration.
    param {*} self
    param {*} method: used for specifying the baseline model of experiment
    """

    def __init__(self, method=None):
        self.method = method
        if method == "LR":
            self.model = LogisticRegression(
                multi_class="ovr", C=0.001, solver="liblinear"
            )
        elif method == "KNN":
            self.model = KNeighborsClassifier()
        elif method == "SVM":
            self.model = OneVsRestClassifier(svm.SVC(kernel="rbf"))
        elif method == "DT":
            self.model = DecisionTreeClassifier(criterion="entropy")
        elif method == "NB":
            self.model = GaussianNB()
        elif method == "RF":
            self.model = RandomForestClassifier(criterion="entropy", verbose=2)
        elif method == "ABC":
            self.model = AdaBoostClassifier()

    """
    description: This function includes entire training process and
        the cross-validation procedure for baselines of KNN, DT, RF and ABC.
        Notice that because of large size of task B dataset, high dimensional features and 
        principle of some models, the entire implementation process may be extremely slow.
        It can even take several hours for a model to run. 
        Some quick models are recommended on README.md and Github link.
    param {*} self
    param {*} Xtrain: train images
    param {*} ytrain: train ground truth labels
    param {*} Xval: validation images
    param {*} yval: validation ground truth labels
    param {*} gridSearch: whether grid search cross-validation (only for KNN, DT, RF and ABC)
    return {*}: if grid search is performed, the cv results are returned.
    """

    def train(self, Xtrain, ytrain, Xval, yval, gridSearch=False):
        print(f"Start training for {self.method}......")
        self.model.fit(Xtrain, ytrain)
        print(f"Finish training for {self.method}.")

        # cross-validation
        if gridSearch:
            print(f"Start tuning(cross-validation) for {self.method}......")
            if self.method == "KNN":
                params = [{"n_neighbors": [i for i in range(1, 12, 1)]}]
            if self.method == "DT":
                params = [{"max_leaf_nodes": [i for i in range(25, 60, 5)]}]
            if self.method == "RF":
                params = [
                    {"n_estimators": [100, 120, 140, 160], "max_depth": [4, 6, 8, 10]}
                ]
            if self.method == "ABC":
                params = [
                    {
                        "n_estimators": [50, 75, 100, 125],
                        "learning_rate": [0.001, 0.1, 1],
                    }
                ]
            grid = GridSearchCV(self.model, params, cv=10, scoring="accuracy")

            grid.fit(np.concatenate((Xtrain, Xval), axis=0), ytrain + yval)
            print(grid.best_params_)
            self.model = grid.best_estimator_

            print(f"Finish tuning(cross-validation) for {self.method}.")

            return grid.cv_results_

    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} Xtrain: train images
    param {*} ytrain: train ground truth labels
    param {*} Xval: validation images
    param {*} yval: validation ground truth labels
    param {*} Xtest: test images
    return {*}: predicted labels for train, validation and test respectively
    """

    def test(self, Xtrain, ytrain, Xval, yval, Xtest):
        print(f"Start testing for {self.method}......")
        self.model.fit(np.concatenate((Xtrain, Xval), axis=0), ytrain + yval)
        pred_test = self.model.predict(Xtest)
        pred_train = self.model.predict(Xtrain)
        pred_val = self.model.predict(Xval)
        print(f"Finish testing for {self.method}.")

        return pred_train, pred_val, pred_test
