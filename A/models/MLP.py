import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MLP(Model):
  def __init__(self):
    super(MLP, self).__init__()
    self.flatten = Flatten(input_shape=(28, 28))
    self.d1 = Dense(128, activation='relu')
    # self.d2 = Dense(1, activation='sigmoid')  # 这里需要考量一下用sigmoid还是softmax
    self.d2 = Dense(1)

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
    return self.d2(x)


  def train(self, model, train_ds, val_ds, EPOCHS):
    print("Start training......")
    for epoch in range(EPOCHS):
      train_pred = []
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()

    
      for step,(train_images, train_labels) in enumerate(train_ds):
        with tf.GradientTape() as tape:
          predictions = model(train_images, training=True)
          prob = tf.nn.sigmoid(predictions)
          
          train_pred.append()
          loss = self.loss_object(train_labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(train_labels, predictions)
      
        if step % 100 == 0:  # evaluate
          self.val_loss.reset_states()
          self.val_accuracy.reset_states()
    
          for val_images,val_labels in val_ds:
            with tf.GradientTape() as tape:
              predictions = model(val_images, training=True)
              val_loss = self.loss_object(val_labels, predictions)
            
              self.val_loss(val_loss)
              self.val_accuracy(val_labels, predictions)
            
            gradients = tape.gradient(val_loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            self.val_loss(val_loss)
            self.val_accuracy(val_labels, predictions)
              
          val_res = {
            "eval_loss": {self.val_loss.result()},
            "eval_acc": round({self.val_accuracy.result() * 100},4),
          }
          print(f'Epoch: {epoch + 1}, Step: {step+1} ', val_res)

      for images, labels in test_ds:
        predictions = model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
      train_res = {
            "train_loss": {self.train_loss.result()},
            "train_acc": round({self.train_accuracy.result() * 100},4),
          }
      print(f'Epoch: {epoch + 1}, Step: {step+1} ', train_res)

      print("Finish training......")
    
    return train_res.update(val_res)


  def test(self,model, test_ds):
    print("Start testing......")
    self.test_loss.reset_states()
    self.test_accuracy.reset_states()

    for images, labels in test_ds:
      predictions = model(images, training=False)
      t_loss = self.loss_object(labels, predictions)

      self.test_loss(t_loss)
      self.test_accuracy(labels, predictions)

    test_res = {
            "test_loss": {self.test_loss.result()},
            "test_acc": round({self.test_accuracy.result() * 100},4),
            "test_pre": round({self.test_precision.result() * 100},4),
            "test_rec": round({self.test_recall.result() * 100},4),
            "test_f1": round({self.test_f1.result() * 100},4),

          }
    print("Finish testing......")
    
    return test_res