# -*- encoding: utf-8 -*-
'''
@File    :   CNN.py
@Time    :   2023/12/16 20:53:52
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@Author  :   Wenrui Li
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for customized network of CNN, including network initialization.
    construction and entire process of training, validation and testing.
'''

# here put the import lib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorboardX import SummaryWriter  # used for nn curves visualization
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout,BatchNormalization

class CNN(Model):

  '''
  description: This function includes all initialization of CNN, like layers used for construction,
    loss function object, optimizer, measurement of accuracy and loss.
  param {*} self
  param {*} task: task A or B
  param {*} method: CNN
  param {*} lr: learning rate
  '''  
  def __init__(self, task, method,lr=0.001):
    super(CNN, self).__init__()
    # network layers definition
    self.c1 = Conv2D(32, 3, padding='same', activation='relu', input_shape=(28,28,3))
    self.b1 = BatchNormalization()
    self.p1 = MaxPooling2D()
    self.c2 = Conv2D(64, 3, padding='same', activation='relu')
    self.b2 = BatchNormalization()
    self.p2 = MaxPooling2D()
    self.dropout = Dropout(0.2)
    self.c3 = Conv2D(128, 3, padding='same', activation='relu')
    self.b3 = BatchNormalization()
    self.p3 = MaxPooling2D()
    self.dropout = Dropout(0.2)
    self.fc = Flatten()
    self.d1 = Dense(64, activation='relu')
    self.d2 = Dense(1, name="outputs")  # binary classification

    # objective function: binary cross entropy
    # notice that here the loss is calculated from logits, no need to set activation function for the output layer
    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
    self.lr = lr

    # adam optimizer
    self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr) 

    # loss and accuracy
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

    self.val_loss = tf.keras.metrics.Mean(name='eval_loss')
    self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    self.method = method
    self.task = task
  
  '''
  description: This function is the actual construction process of customized network.
  param {*} self
  param {*} x: input 
  return {*}: output logits
  '''  
  def call(self, x):
    x = self.c1(x)
    x = self.b1(x)
    x = self.p1(x)
    x = self.c2(x)
    x = self.b2(x)
    x = self.p2(x)
    x = self.c3(x)
    x = self.b3(x)
    x = self.p3(x)
    x = self.dropout(x)
    x = self.fc(x)
    x = self.d1(x)
    return self.d2(x)


  '''
  description: This function is used for the entire process of training. 
    Notice that loss of both train and validation are backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} train_ds: loaded train dataset as batches
  param {*} val_ds: loaded validation dataset as batches
  param {*} EPOCHS: number of epochs
  return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
  '''  
  def train(self, model, train_ds, val_ds, EPOCHS):
    print("Start training......")
    if not os.path.exists("Outputs/images/nn_curves/"):
        os.makedirs("Outputs/images/nn_curves/") 
    writer = SummaryWriter(f"Outputs/images/nn_curves/{self.method}_task{self.task}")

    for epoch in range(EPOCHS):

      # train
      train_pred = []  # label prediction
      ytrain = [] # ground truth
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()

    
      for step,(train_images, train_labels) in enumerate(train_ds):
        with tf.GradientTape() as tape:
          predictions = model(train_images, training=True)  # logits
          train_prob = tf.nn.sigmoid(predictions)  # probabilities
          for i in train_prob:
            train_pred.append(1) if i >= 0.5 else train_pred.append(0)  # predicted labels
          ytrain += np.array(train_labels).tolist()  # ground truth
          loss = self.loss_object(train_labels, predictions)
        
        # backward propagation
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(train_labels, predictions)
      
        # validation
        if step % 50 == 0:  
          val_pred = [] 
          yval = []
          self.val_loss.reset_states()
          self.val_accuracy.reset_states()
    
          for val_images,val_labels in val_ds:
            with tf.GradientTape() as tape:
              predictions = model(val_images, training=True)
              val_prob = tf.nn.sigmoid(predictions)
              for i in val_prob:
                val_pred.append(1) if i >= 0.5 else val_pred.append(0)
              yval += np.array(val_labels).tolist()
              val_loss = self.loss_object(val_labels, predictions)
            
              self.val_loss(val_loss)
              self.val_accuracy(val_labels, predictions)
            
            # backward propagation
            gradients = tape.gradient(val_loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            self.val_loss(val_loss)
            self.val_accuracy(val_labels, predictions)
              
          val_res = {
            "val_loss": np.array(self.val_loss.result()).tolist(),
            "val_acc": round(np.array(self.val_accuracy.result()) * 100,4),
          }
          print(f'Epoch: {epoch + 1}, Step: {step} ', val_res)

      train_res = {
            "train_loss": np.array(self.train_loss.result()).tolist(),
            "train_acc": round(np.array(self.train_accuracy.result()) * 100,4),
          }
      print(f'Epoch: {epoch + 1}', train_res)

      writer.add_scalars('loss',{"train_loss":np.array(self.train_loss.result()).tolist(), \
                                            "val_loss": np.array(self.val_loss.result()).tolist()}, epoch)
      writer.add_scalars('accuracy',{"train_accuracy":np.array(self.train_accuracy.result()).tolist(), \
                                            "val_accuracy": np.array(self.val_accuracy.result()).tolist()}, epoch)

      train_pred = np.array(train_pred)
      val_pred = np.array(val_pred)

    print("Finish training.")
    writer.close()

    return train_res, val_res, train_pred, val_pred, ytrain, yval

  '''
  description: This function is used for the entire process of testing. 
    Notice that loss of testing is not backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} test_ds: loaded test dataset as batches
  return {*}: accuracy and loss result, predicted labels and ground truth of test dataset
  '''  
  def test(self,model, test_ds):
    print("Start testing.")
    test_pred = []  # predicted labels
    ytest = []  # ground truth
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

    for test_images, test_labels in test_ds:
      predictions = model(test_images, training=False)  # logits
      test_prob = tf.nn.sigmoid(predictions)  # probability
      for i in test_prob:
        test_pred.append(1) if i >= 0.5 else test_pred.append(0)  # predicted labels
      ytest += np.array(test_labels).tolist()  # ground truth

      t_loss = self.loss_object(test_labels, predictions)
      self.test_loss(t_loss)
      self.test_accuracy(test_labels, predictions)

    test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
            "test_acc": round(np.array(self.test_accuracy.result()) * 100,4),
          }
    print("Finish testing.")
    test_pred = np.array(test_pred)
    
    return test_res, test_pred, ytest