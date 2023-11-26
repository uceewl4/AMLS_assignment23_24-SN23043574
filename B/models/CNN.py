import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model
import numpy as np
import os
from tensorboardX import SummaryWriter

class CNN(Model):
  def __init__(self, task, method,multilabel=False,lr=0.001):
    super(CNN, self).__init__()
    # different to MLP, no need to make the image input in a flattened way, can just input image with batches and calculate with kernel
    # but for MLP we need to first flatten the image because it's not a kernel calculation
    self.multilabel = multilabel
    self.c1 = Conv2D(64, 5, padding='same', activation='relu', input_shape=(28,28,3))
    self.b1 = BatchNormalization()
    self.p1 = MaxPooling2D()
    self.c2 = Conv2D(128, 2, padding='same', activation='relu')
    self.b2 = BatchNormalization()
    self.p2 = MaxPooling2D()
    self.dropout = Dropout(0.2)
    self.c3 = Conv2D(128, 3, padding='same', activation='relu')
    self.b3 = BatchNormalization()
    self.p3 = MaxPooling2D()
    self.dropout = Dropout(0.3)
    self.fc = Flatten()
    self.d1 = Dense(256, activation='relu')
    self.d2 = Dense(64, activation='relu')
    self.d3 = Dense(9, name="outputs")

    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
    self.lr = lr
    self.optimizer = tf.keras.optimizers.Adam(learning_rate = lr)

    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    self.val_loss = tf.keras.metrics.Mean(name='eval_loss')
    self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    self.method = method
    self.task = task
  
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
    x = self.d2(x)
    return self.d3(x)


  def train(self, model, train_ds, val_ds, EPOCHS):
    print("Start training......")
    if not os.path.exists("Outputs/images/nn_curves/"):
        os.makedirs("Outputs/images/nn_curves/") 
    writer = SummaryWriter(f"Outputs/images/nn_curves/{self.method}_task{self.task}")
    for epoch in range(EPOCHS):
      train_pred = []
      ytrain = []
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()

    
      for step,(train_images, train_labels) in enumerate(train_ds):
        with tf.GradientTape() as tape:
          predictions = model(train_images, training=True)
          if self.multilabel == False:
            train_prob = tf.nn.softmax(predictions) 
            train_pred += np.argmax(train_prob,axis=1).tolist() 
          else:
            train_prob_multilabel = tf.nn.sigmoid(predictions)
            train_prob = tf.nn.softmax(predictions) 

            train_pred_multilabel = np.zeros_like(predictions)
            train_pred_multilabel[train_prob_multilabel >= 0.6] = 1

            tmp = np.argmax(train_prob,axis=1).tolist()
            for index,(pred,label) in enumerate(zip(train_pred_multilabel,train_labels)):
              if pred[int(label)] == 1:
                tmp[index] = label
            train_pred += tmp

          ytrain += np.array(train_labels).tolist()
          loss = self.loss_object(train_labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(train_labels, predictions)
      
        if step % 600 == 0:  # evaluate
          val_pred = []
          yval = []
          self.val_loss.reset_states()
          self.val_accuracy.reset_states()
    
          for val_images,val_labels in val_ds:

            with tf.GradientTape() as tape:
              predictions = model(val_images, training=True)
              if self.multilabel == False:
                val_prob = tf.nn.softmax(predictions) 
                val_pred += np.argmax(val_prob,axis=1).tolist() 
              else:
                val_prob_multilabel = tf.nn.sigmoid(predictions)
                val_prob = tf.nn.softmax(predictions) 

                val_pred_multilabel = np.zeros_like(predictions)
                val_pred_multilabel[val_prob_multilabel >= 0.6] = 1  # threshold cannot be set to 0.5 considering the distribution

                tmp = np.argmax(val_prob,axis=1).tolist()
                for index,(pred,label) in enumerate(zip(val_pred_multilabel,val_labels)):
                  if pred[int(label)] == 1:
                    tmp[index] = label
                val_pred += tmp

              yval += np.array(val_labels).tolist()
              val_loss = self.loss_object(val_labels, predictions)
            
              self.val_loss(val_loss)
              self.val_accuracy(val_labels, predictions)
            
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

    if self.multilabel == False:
      return train_res, val_res, train_pred, val_pred, ytrain, yval
    else:
      return train_res, val_res, train_pred, train_pred_multilabel, val_pred, val_pred_multilabel, ytrain, yval


  def test(self,model, test_ds):
    print("Start testing......")
    test_pred = []
    ytest = []
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

    for test_images, test_labels in test_ds:
      predictions = model(test_images, training=False)

      if self.multilabel == False:
        test_prob = tf.nn.softmax(predictions) 
        test_pred += np.argmax(test_prob,axis=1).tolist() 
      else:
        test_prob_multilabel = tf.nn.sigmoid(predictions)
        test_prob = tf.nn.softmax(predictions) 

        test_pred_multilabel = np.zeros_like(predictions)
        test_pred_multilabel[test_prob_multilabel >= 0.6] = 1

        tmp = np.argmax(test_prob,axis=1).tolist()
        for index,(pred,label) in enumerate(zip(test_pred_multilabel,test_labels)):
          if pred[int(label)] == 1:
            tmp[index] = label
        test_pred += tmp
      
      ytest += np.array(test_labels).tolist()

      t_loss = self.loss_object(test_labels, predictions)
      self.test_loss(t_loss)
      self.test_accuracy(test_labels, predictions)

    test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
            "test_acc": round(np.array(self.test_accuracy.result()) * 100,4),
          }
    print("Finish testing.")
    test_pred = np.array(test_pred)
    
    if self.multilabel == False:
      return test_res, test_pred, ytest
    else:
      return test_res, test_pred, test_pred_multilabel, ytest