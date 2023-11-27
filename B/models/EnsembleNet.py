import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Model
import numpy as np
from B.models.MLP import MLP
from B.models.CNN import CNN

class EnsembleNet(Model):
  def __init__(self):
    super(EnsembleNet, self).__init__()
    self.w1_model = MLP(task="B",method="MLP")
    self.w2_model = CNN(task="B",method="CNN")

    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.val_loss = tf.keras.metrics.Mean(name='teval_loss')
    self.test_loss = tf.keras.metrics.Mean(name='test_loss')

  def train(self, train_ds,val_ds, EPOCHS):
    self.w1_model.train(self.w1_model, train_ds, val_ds, EPOCHS)
    self.w2_model.train(self.w2_model, train_ds, val_ds, EPOCHS)
    
          
  def weight_selection(self, train_ds, val_ds):
    train_pred,val_pred = [],[]
    ytrain,yval = [],[]
    ratios = [1/i for i in range(2,10)]
    loss_list = []
    for ratio in ratios:
      for val_images,val_labels in val_ds:
          w1_predictions = self.w1_model(val_images, training=False)
          w2_predictions = self.w2_model(val_images, training=False)
          total_predictions = ratio*w1_predictions + (1-ratio)*w2_predictions
          val_prob = tf.nn.softmax(total_predictions) 
          val_pred += np.argmax(val_prob,axis=1).tolist() 
          yval += np.array(val_labels).tolist()

          t_loss = self.loss_object(val_labels, total_predictions)
          self.val_loss(t_loss)

      loss_list.append(np.array(self.val_loss.result()).tolist())
    
    self.weight = ratios[loss_list.index(min(loss_list))]

    for train_images,train_labels in train_ds:
        w1_predictions = self.w1_model(train_images, training=False)
        w2_predictions = self.w2_model(train_images, training=False)
        total_predictions = ratio*w1_predictions + (1-ratio)*w2_predictions
        train_prob = tf.nn.softmax(total_predictions) 
        train_pred += np.argmax(train_prob,axis=1).tolist()
        ytrain += np.array(train_labels).tolist()
        t_loss = self.loss_object(train_labels, total_predictions)
        self.train_loss(t_loss)

    train_res = {
        "train_loss": np.array(self.train_loss.result()).tolist(),
      }
    
    val_res = {
            "val_loss": np.array(self.val_loss.result()).tolist(),
    }

    train_pred = np.array(train_pred)
    val_pred = np.array(val_pred)
    
    return train_res, val_res, train_pred, val_pred, ytrain, yval

  def test(self, test_ds):
    print("Start testing......")
    test_pred = []
    ytest = []
    
    for test_images,test_labels in test_ds:
        w1_predictions = self.w1_model(test_images, training=False)
        w2_predictions = self.w2_model(test_images, training=False)
        total_predictions = self.weight*w1_predictions + (1-self.weight)*w2_predictions
        test_prob = tf.nn.softmax(total_predictions)
        test_pred += np.argmax(test_prob,axis=1).tolist()

        ytest += np.array(test_labels).tolist()
        t_loss = self.loss_object(test_labels, total_predictions)
        self.test_loss(t_loss)
      
    test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
    }
    print("Finish testing.")
    test_pred = np.array(test_pred)
    
    return test_res, test_pred, ytest