import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model
import numpy as np
import os
from tensorboardX import SummaryWriter

class CNN(Model):
  def __init__(self, task, method):
    super(CNN, self).__init__()
    # different to MLP, no need to make the image input in a flattened way, can just input image with batches and calculate with kernel
    # but for MLP we need to first flatten the image because it's not a kernel calculation
    self.c1 = Conv2D(16, 3, padding='same', activation='relu', input_shape=(28,28,3))
    self.p1 = MaxPooling2D()
    self.c2 = Conv2D(32, 3, padding='same', activation='relu')
    self.p2 = MaxPooling2D()
    self.c3 = Conv2D(64, 3, padding='same', activation='relu')
    self.p3 = MaxPooling2D()
    self.dropout = Dropout(0.2)
    self.fc = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(9, name="outputs")

    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
    self.optimizer = tf.keras.optimizers.Adam()

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
    x = self.p1(x)
    x = self.c2(x)
    x = self.p2(x)
    x = self.c3(x)
    x = self.p3(x)
    x = self.dropout(x)
    x = self.fc(x)
    x = self.d1(x)
    return self.d2(x)


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
          train_prob = tf.nn.softmax(predictions)
          train_pred += np.argmax(train_prob,axis=1).tolist()
         
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
              val_prob = tf.nn.softmax(predictions)
              val_pred += np.argmax(val_prob,axis=1).tolist()
              
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
      writer.add_scalars('accuracy',{"train_loss":np.array(self.train_accuracy.result()).tolist(), \
                                            "val_loss": np.array(self.val_accuracy.result()).tolist()}, epoch)
      

      train_pred = np.array(train_pred)
      val_pred = np.array(val_pred)

    print("Finish training.")
    writer.close()

    return train_res, val_res, train_pred, val_pred, ytrain, yval


  def test(self,model, test_ds):
    print("Start testing......")
    test_pred = []
    ytest = []
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

    for test_images, test_labels in test_ds:
      predictions = model(test_images, training=False)

      test_prob = tf.nn.softmax(predictions)
      test_pred += np.argmax(test_prob,axis=1).tolist()
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
    
    return test_res, test_pred, ytest