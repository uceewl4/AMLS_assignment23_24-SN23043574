# -*- encoding: utf-8 -*-
"""
@File    :   InceptionV3.py
@Time    :   2023/12/16 22:17:47
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for pretrained model Inception-V3 as feature extractor,
  followed by 7 classifiers of ML baselines.
"""

# here put the import lib
import numpy as np
from sklearn import svm
import tensorflow as tf
from tensorflow.keras import Model, models
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB


class InceptionV3(Model):

    """
    description: This function is used for initialization of Inception-V3 + classifiers.
    param {*} self
    param {*} method: baseline model selected
    """

    def __init__(self, method=None):
        super(InceptionV3, self).__init__()

        # need resizing to satisfy the minimum image size need of Inception-V3
        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.Resizing(75, 75, interpolation="bilinear"),
                tf.keras.layers.Rescaling(1.0 / 255, input_shape=(75, 75, 3)),
            ]
        )

        # pretrained model
        self.base_model = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_shape=(75, 75, 3)
        )
        self.base_model.trainable = False

        # feature extractor
        self.model = models.Sequential(
            [self.data_augmentation, self.base_model, tf.keras.layers.Flatten()]
        )

        # classifiers
        self.method = method
        if method == "InceptionV3_SVM":
            self.clf = OneVsRestClassifier(svm.SVC(kernel="rbf"))
        elif method == "InceptionV3_LR":
            self.clf = LogisticRegression(
                penalty="l1", solver="liblinear", multi_class="ovr"
            )
        elif method == "InceptionV3_KNN":
            self.clf = KNeighborsClassifier()
        elif method == "InceptionV3_DT":
            self.clf = DecisionTreeClassifier(criterion="entropy")
        elif method == "InceptionV3_NB":
            self.clf = GaussianNB()
        elif method == "InceptionV3_RF":
            self.clf = RandomForestClassifier(criterion="entropy", verbose=2)
        elif method == "InceptionV3_ABC":
            self.clf = AdaBoostClassifier()

    """
    description: This function is used for extracting features with pre-trained model.
    param {*} self
    param {*} Xtrain: train images
    param {*} Xval: validation images
    param {*} Xtest: test images
  """

    def get_features(self, Xtrain, Xval, Xtest):
        print(f"Start getting features through InceptionV3......")
        self.train_features = self.model.predict(Xtrain)
        self.val_features = self.model.predict(Xval)
        self.test_features = self.model.predict(Xtest)
        self.tune_features = self.model.predict(np.concatenate((Xtrain, Xval), axis=0))
        print("Finish getting features.")

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
    param {*} Xtest: test images
    param {*} gridSearch: whether grid search cross-validation (only for KNN, DT, RF and ABC)
    return {*}: if grid search is performed, the cv results are returned.
  """

    def train(self, model, Xtrain, y_train, Xval, y_val, Xtest, gridSearch=False):
        # get features from pretrained network
        model.get_features(Xtrain, Xval, Xtest)

        # concate with classifier
        print(f"Start training for {self.method}......")
        model.clf.fit(self.train_features, y_train)
        print(f"Finish training for {self.method}.")

        # cross-validation
        if gridSearch:
            print(f"Start tuning(cross-validation) for {self.method}......")
            if "KNN" in self.method:
                params = [{"n_neighbors": [i for i in range(1, 30, 2)]}]
            if "DT" in self.method:
                params = [{"max_leaf_nodes": [i for i in range(20, 65, 5)]}]
            if "RF" in self.method:
                params = [
                    {"n_estimators": [120, 140, 160, 180], "max_depth": [8, 10, 12, 14]}
                ]
            if "ABC" in self.method:
                params = [
                    {
                        "n_estimators": [50, 75, 100, 125],
                        "learning_rate": [0.001, 0.1, 1],
                    }
                ]
            grid = GridSearchCV(model.clf, params, cv=10, scoring="accuracy")

            grid.fit(self.tune_features, y_train + y_val)
            print(grid.best_params_)
            model.clf = grid.best_estimator_

            print(f"Finish tuning(cross-validation) for {self.method}.")
            return grid.cv_results_

    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} ytrain: train ground truth labels
    param {*} yval: validation ground truth labels
    return {*}: predicted labels for train, validation and test respectively
  """

    def test(self, model, ytrain, yval):
        print("Start testing......")
        model.clf.fit(self.tune_features, ytrain + yval)

        pred_train = model.clf.predict(self.train_features)
        pred_val = model.clf.predict(self.val_features)
        pred_test = model.clf.predict(self.test_features)
        print("Finish training.")

        return pred_train, pred_val, pred_test
