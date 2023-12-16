# -*- encoding: utf-8 -*-
'''
@File    :   EnsembleNet.py
@Time    :   2023/12/16 22:14:19
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@Author  :   Wenrui Li
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for customized network of EnsembleNet, which integrate both MLP and CNN.
'''

# here put the import lib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from B.models.MLP import MLP
from B.models.CNN import CNN

class EnsembleNet(Model):
  '''
  description: This function is used for initialization of EnsembleNet including independent model components and loss function.
  param {*} self
  param {*} lr: learning rate
  '''  
  def __init__(self,lr):
    super(EnsembleNet, self).__init__()
    self.w1_model = MLP(task="B",method="MLP",lr=lr)  # model component MLP
    self.w2_model = CNN(task="B",method="CNN",lr=lr)  # model component CNN
    self.lr = lr

    # objective funtio: binary cross entropy, calculated from logits
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.val_loss = tf.keras.metrics.Mean(name='teval_loss')
    self.test_loss = tf.keras.metrics.Mean(name='test_loss')

  '''
  description: his function is used for the entire process of training with parallel MLP and CNN.
  param {*} self
  param {*} train_ds: loaded train dataset as batches
  param {*} val_ds: loaded validation dataset as batches
  param {*} EPOCHS: number of epochs
  ''' 
  def train(self, train_ds,val_ds, EPOCHS):
    self.w1_model.train(self.w1_model, train_ds, val_ds, EPOCHS)
    self.w2_model.train(self.w2_model, train_ds, val_ds, EPOCHS)
    
  '''
  description: This function is used for weight selection to find optimized ratio
    for decision-level fusion with validation set.
  param {*} self
  param {*} train_ds: loaded train dataset as batches
  param {*} val_ds: loaded validation dataset as batches
  return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
  '''  
  def weight_selection(self, train_ds, val_ds):
    train_pred,val_pred = [],[]   # predicted labels
    ytrain,yval = [],[]  # ground truth
    ratios = [1/i for i in range(2,10)]  # weight candidates
    loss_list = []

     # weight selection
    for ratio in ratios:
      for val_images,val_labels in val_ds:
          w1_predictions = self.w1_model(val_images, training=False)  # MLP logits
          w2_predictions = self.w2_model(val_images, training=False)  # CNN logits
          # decision-level fusion, logits
          total_predictions = ratio*w1_predictions + (1-ratio)*w2_predictions
          val_prob = tf.nn.softmax(total_predictions)  # probability
          val_pred += np.argmax(val_prob,axis=1).tolist()  # predicted labels
          yval += np.array(val_labels).tolist()  # ground truth

          t_loss = self.loss_object(val_labels, total_predictions)
          self.val_loss(t_loss)

      loss_list.append(np.array(self.val_loss.result()).tolist())
    
    self.weight = ratios[loss_list.index(min(loss_list))]  # optimized weight
    
    # try optimized weight on train to get predicted train labels
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

  '''
  description: This function is used for testing with optimized weight.
  param {*} self
  param {*} test_ds: loaded test dataset as batches
  return {*}: accuracy and loss result, predicted labels and ground truth of test dataset
  ''' 
  def test(self, test_ds):
    print("Start testing......")
    test_pred = []  # predicted label
    ytest = []  # ground truth
    
    for test_images,test_labels in test_ds:
        w1_predictions = self.w1_model(test_images, training=False)
        w2_predictions = self.w2_model(test_images, training=False)
        total_predictions = self.weight*w1_predictions + (1-self.weight)*w2_predictions  # logits
        test_prob = tf.nn.softmax(total_predictions)  # probability 
        test_pred += np.argmax(test_prob,axis=1).tolist()  # predicted labels

        ytest += np.array(test_labels).tolist()  # ground truth
        t_loss = self.loss_object(test_labels, total_predictions)
        self.test_loss(t_loss)
      
    test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
    }
    print("Finish testing.")
    print(f"The best ratio for EnsembleNet is {self.weight}.")
    test_pred = np.array(test_pred)
    
    return test_res, test_pred, ytest