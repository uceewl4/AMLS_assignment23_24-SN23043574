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
    # self.base_model.summary()
    self.model =models.Sequential([
      self.data_augmentation,
      self.base_model,
      tf.keras.layers.Flatten()]
    )
    # self.model = Model(inputs=self.base_model.input, outputs=self.model.get_layer('flatten').output) #想获取哪层的特征，就把引号里的内容改成那个层的名字就行
    if method == "VGG16_SVM":
      self.clf = svm.SVC(kernel='linear')
    elif method == "VGG16_LR":
      self.clf = LogisticRegression(penalty="l1",solver="liblinear")
    elif method == "VGG16_KNN":  
      self.clf = KNeighborsClassifier()
    elif method == "VGG16_DT":  
      self.clf = DecisionTreeClassifier(criterion='entropy')
    elif method == "VGG16_NB": 
      self.clf = GaussianNB()
    elif method == "VGG16_RF":  
      self.clf = RandomForestClassifier(criterion='entropy')
    elif method == "VGG16_ABC":  
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

  

