import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
from tensorflow.keras import Model
import numpy as np

class MLP(Model):
  def __init__(self):
    super(MLP, self).__init__()
    self.flatten = Flatten(input_shape=(28, 28, 3))
    self.d1 = Dense(4096, activation='relu')
    self.d2 = Dense(4096, activation='relu')
    self.do1 = Dropout(0.2)
    self.d3 = Dense(1024, activation='relu')
    self.d4 = Dense(256, activation='relu')
    self.do2 = Dropout(0.2)
    self.d5 = Dense(64, activation='relu')
    self.d6 = Dense(1)

    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
    self.optimizer = tf.keras.optimizers.Adam()

    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
   
    self.val_loss = tf.keras.metrics.Mean(name='eval_loss')
    self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
  
    self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.do1(x)
    x = self.d3(x)
    x = self.d4(x)
    x = self.do2(x)
    x = self.d5(x)
   
    return self.d6(x)


  def train(self, model, train_ds, val_ds, EPOCHS):
    print("Start training......")
    for epoch in range(EPOCHS):
      train_pred = []
      ytrain = []
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()

    
      for step,(train_images, train_labels) in enumerate(train_ds):
        with tf.GradientTape() as tape:
          predictions = model(train_images, training=True)
          train_prob = tf.nn.sigmoid(predictions)
          for i in train_prob:
            train_pred.append(1) if i >= 0.5 else train_pred.append(0)
          ytrain += np.array(train_labels).tolist()
          loss = self.loss_object(train_labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(train_labels, predictions)
      
        if step % 50 == 0:  # evaluate
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

      train_pred = np.array(train_pred)
      val_pred = np.array(val_pred)

    print("Finish training......")

    return train_res, val_res, train_pred, val_pred, ytrain, yval


  def test(self,model, test_ds):
    print("Start testing......")
    test_pred = []
    ytest = []
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

    for test_images, test_labels in test_ds:
      predictions = model(test_images, training=False)

      test_prob = tf.nn.sigmoid(predictions)
      for i in test_prob:
        test_pred.append(1) if i >= 0.5 else test_pred.append(0)
      ytest += np.array(test_labels).tolist()

      t_loss = self.loss_object(test_labels, predictions)
      self.test_loss(t_loss)
      self.test_accuracy(test_labels, predictions)

    test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
            "test_acc": round(np.array(self.test_accuracy.result()) * 100,4),
          }
    print("Finish testing......")
    test_pred = np.array(test_pred)
    
    return test_res, test_pred, ytest