import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class MobileNetV2(Model):
  def __init__(self, method=None):
    super(MobileNetV2, self).__init__()
    # self.flatten = Flatten(input_shape=(28, 28, 3))
    # self.d1 = Dense(128, activation='relu')
    # # self.d2 = Dense(1, activation='sigmoid')
    # self.d2 = Dense(9)

    self.data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(32,32,interpolation='bilinear'),
        tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 3)),
    ])
    self.base_model = tf.keras.applications.MobileNetV2(input_shape=(32,32,3),  # resize into 32x32 to satisfy the requirement of MobileNetV2
                                               include_top=False,
                                               weights='imagenet')
    self.base_model.trainable = False
    # self.base_model.summary()
    self.model = models.Sequential([
        self.data_augmentation,
        self.base_model,
        tf.keras.layers.Flatten()]
      )

    if method == "MobileNetV2_SVM":
      self.clf = svm.SVC(kernel='linear')
    elif method == "MobileNetV2_LR":
      self.clf = LogisticRegression(penalty="l1",solver="liblinear")
    elif method == "MobileNetV2_KNN":  
      self.clf = KNeighborsClassifier()
    elif method == "MobileNetV2_DT":  
      self.clf = DecisionTreeClassifier(criterion='entropy')
    elif method == "MobileNetV2_NB": 
      self.clf = GaussianNB()
    elif method == "MobileNetV2_RF":  
      self.clf = RandomForestClassifier(criterion='entropy')
    elif method == "MobileNetV2_ABC":  
      self.clf = AdaBoostClassifier()

  def get_features(self, x):
      features = self.model.predict(x)     

      return features

  def train(self, model, Xtrain, y_train):
      features = model.get_features(Xtrain)
      model.clf.fit(features, y_train)

      # 这里是否grid_search再考虑
      
  def test(self, model, Xtest, ytest):
      test_features = model.get_features(Xtest)
      y_pred = model.clf.predict(test_features)
      print(accuracy_score(ytest,y_pred))