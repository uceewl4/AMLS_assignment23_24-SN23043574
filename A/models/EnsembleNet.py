import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Model
import numpy as np
from A.models.MLP import MLP
from A.models.CNN import CNN

class EnsembleNet(Model):
  def __init__(self):
    super(EnsembleNet, self).__init__()
    self.w1_model = MLP()
    self.w2_model = CNN()

    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
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
          total_predictions = tf.nn.softmax(ratio*w1_predictions + (1-ratio)*w2_predictions)
          val_prob = tf.nn.sigmoid(total_predictions)
          for i in val_prob:
            val_pred.append(1) if i >= 0.5 else val_pred.append(0)
          yval += np.array(val_labels).tolist()

          t_loss = self.loss_object(val_labels, total_predictions)
          self.val_loss(t_loss)

      loss_list.append(np.array(self.val_loss.result()).tolist())
    
    self.weight = ratios[loss_list.index(max(loss_list))]

    for train_images,train_labels in train_ds:
        w1_predictions = self.w1_model(train_images, training=False)
        w2_predictions = self.w2_model(train_images, training=False)
        total_predictions = tf.nn.softmax(self.weight*w1_predictions + (1-self.weight)*w2_predictions)
        train_prob = tf.nn.sigmoid(total_predictions)

        for i in train_prob:
            train_pred.append(1) if i >= 0.5 else train_pred.append(0)
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
        total_predictions = tf.nn.softmax(self.weight*w1_predictions + (1-self.weight)*w2_predictions)
        test_prob = tf.nn.sigmoid(total_predictions)

        for i in test_prob:
            test_pred.append(1) if i >= 0.5 else test_pred.append(0)
        ytest += np.array(test_labels).tolist()
        t_loss = self.loss_object(test_labels, total_predictions)
        self.test_loss(t_loss)
      
    test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
    }
    print("Finish testing.")
    test_pred = np.array(test_pred)
    
    return test_res, test_pred, ytest