from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model,models
from sklearn import svm
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class VGG16(Model):
  def __init__(self, method=None):
    super(VGG16, self).__init__()
    # self.flatten = Flatten(input_shape=(28, 28, 3))
    # self.d1 = Dense(128, activation='relu')
    # # self.d2 = Dense(1, activation='sigmoid')
    # self.d2 = Dense(9)

    self.data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(32,32,interpolation='bilinear'),
        tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 3)),
    ])
    self.base_model =tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                  input_shape=(32, 32, 3)) 
    self.base_model.trainable = False

    self.model =models.Sequential([
      self.data_augmentation,
      self.base_model,
      tf.keras.layers.Flatten()]
    )

    self.method = method

    if method == "VGG16_SVM":
      self.clf = svm.SVC(gamma=0.1,kernel='linear') 
    elif method == "VGG16_LR":
      self.clf = LogisticRegression(solver="liblinear") 
    elif method == "VGG16_KNN":  
      self.clf = KNeighborsClassifier()
    elif method == "VGG16_DT":  
      self.clf = DecisionTreeClassifier(criterion='entropy')
    elif method == "VGG16_NB": 
      self.clf = MultinomialNB() 
    elif method == "VGG16_RF":  
      self.clf = RandomForestClassifier(criterion='entropy')
    elif method == "VGG16_ABC":  
      self.clf = AdaBoostClassifier()
       

  def get_features(self, Xtrain, Xval, Xtest):
    print(f"Start getting features through VGG16......")
    self.train_features = self.model.predict(Xtrain)     
    self.val_features = self.model.predict(Xval)   
    self.test_features = self.model.predict(Xtest)
    self.tune_features = self.model.predict(np.concatenate((Xtrain,Xval),axis=0))
    print("Finish getting features.")

  def train(self, model, Xtrain, y_train, Xval, y_val, Xtest, gridSearch=False):
      model.get_features(Xtrain, Xval, Xtest)

      print(f"Start training for {self.method}......")
      model.clf.fit(self.train_features, y_train) 
      print(f"Finish training for {self.method}.")

      if gridSearch:  
          print(f"Start tuning(cross-validation) for {self.method}......")
          if "KNN" in self.method:
              params = [{"n_neighbors": [i for i in range(1,30,2)]}]
          if "DT" in self.method:
              params = [{"max_leaf_nodes": [i for i in range(45,95,5)]}]
          if "RF" in self.method:  # very slow need to notify TA
              params = [{"n_estimators": [120, 140, 160, 180, 200], "max_depth": [8, 10, 12, 14, 16]}]
          if "ABC" in self.method:
              params = [{"n_estimators": [50, 75, 100, 125, 150, 175], "learning_rate": [0.001, 0.01, 0.1, 1]}]
          grid = GridSearchCV(model.clf, params, cv=10, scoring="accuracy")

          grid.fit(self.tune_features, y_train+y_val)
          print(grid.best_params_)
          model.clf = grid.best_estimator_

          print(f"Finish tuning(cross-validation) for {self.method}.")
          return grid.cv_results_

  
  def test(self, model, ytrain, yval):
      print("Start testing......")
      model.clf.fit(self.tune_features,ytrain+yval)

      pred_train = model.clf.predict(self.train_features)
      pred_val = model.clf.predict(self.val_features)
      pred_test = model.clf.predict(self.test_features)
      print("Finish training.")

      return pred_train, pred_val, pred_test



