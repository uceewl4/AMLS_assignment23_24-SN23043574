from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model,models
from sklearn import svm
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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, CategoricalNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class InceptionV3(Model):
  def __init__(self,method=None):
    super(InceptionV3, self).__init__()
    # self.flatten = Flatten(input_shape=(28, 28, 3))
    # self.d1 = Dense(128, activation='relu')
    # # self.d2 = Dense(1, activation='sigmoid')
    # self.d2 = Dense(9)

    self.data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(75,75,interpolation='bilinear'),
        tf.keras.layers.Rescaling(1./255, input_shape=(75, 75, 3)),
    ])
    self.base_model =tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
                  input_shape=(75, 75, 3)) 
    self.base_model.trainable = False
    self.model =models.Sequential([
      self.data_augmentation,
      self.base_model,
      tf.keras.layers.Flatten()]
    )
    self.method = method

    if method == "InceptionV3_SVM":
      self.clf = svm.SVC(kernel='rbf') 
    elif method == "InceptionV3_LR":
      self.clf = LogisticRegression(C=0.01,solver="liblinear") 
    elif method == "InceptionV3_KNN":  
      self.clf = KNeighborsClassifier()
    elif method == "InceptionV3_DT":  
      self.clf = DecisionTreeClassifier(criterion='entropy')
    elif method == "InceptionV3_NB": 
      self.clf = BernoulliNB() 
    elif method == "InceptionV3_RF":  
      self.clf = RandomForestClassifier(criterion='entropy',verbose=2)
    elif method == "InceptionV3_ABC":  
      self.clf = AdaBoostClassifier()


  def get_features(self, Xtrain, Xval, Xtest):
    print(f"Start getting features through InceptionV3......")
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
              params = [{"n_estimators": [120, 140, 160, 180], "max_depth": [8, 10, 12, 14]}]
          if "ABC" in self.method:
              params = [{"n_estimators": [50, 75, 100, 125], "learning_rate": [0.001, 0.1, 1]}]
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


