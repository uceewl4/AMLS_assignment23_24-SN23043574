import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MLP(Model):
  def __init__(self):
    super(MLP, self).__init__()
    self.flatten = Flatten(input_shape=(28, 28, 3))
    self.d1 = Dense(128, activation='relu')
    # self.d2 = Dense(1, activation='sigmoid')
    self.d2 = Dense(9)

    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
    self.optimizer = tf.keras.optimizers.Adam()

    self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    self.train_precision= tf.keras.metrics.Precision(name='train_precision')
    self.train_recall = tf.keras.metrics.Recall(name='train_recall')
    self.train_f1 = tf.keras.metrics.F1Score(name='train_f1')

    self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
    self.eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')
    self.eval_precision= tf.keras.metrics.Precision(name='eval_precision')
    self.eval_recall = tf.keras.metrics.Recall(name='eval_recall')
    self.eval_f1 = tf.keras.metrics.F1Score(name='eval_f1')

    self.test_loss = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    self.test_precision= tf.keras.metrics.Precision(name='test_precision')
    self.test_recall = tf.keras.metrics.Recall(name='test_recall')
    self.test_f1 = tf.keras.metrics.F1Score(name='test_f1')


  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


  def train(self, model, train_ds, eval_ds, EPOCHS):
    print("Start training......")
    for epoch in range(EPOCHS):
      self.train_loss.reset_states()
      self.train_accuracy.reset_states()
      self.train_precision.reset_states()
      self.train_recall.reset_states()
      self.train_f1.reset_states()

    
      for step,(train_images, train_labels) in enumerate(train_ds):
        with tf.GradientTape() as tape:
          predictions = model(train_images, training=True)
          loss = self.loss_object(train_labels, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(train_labels, predictions)
        
        if step % 100 == 0:  # evaluate
          self.eval_loss.reset_states()
          self.eval_accuracy.reset_states()
    
          for (eval_images,eval_labels) in enumerate(eval_ds):
            with tf.GradientTape() as tape:
              predictions = model(eval_images, training=True)
              eval_loss = self.loss_object(eval_labels, predictions)
            
              self.eval_loss(loss)
              self.eval_accuracy(eval_labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(train_labels, predictions)
          
          eval_res = {
            "eval_loss": {self.eval_loss.result()},
            "eval_acc": round({self.eval_accuracy.result() * 100},4),
            "eval_pre": round({self.eval_precision.result() * 100},4),
            "eval_rec": round({self.eval_rec.result() * 100},4),
            "eval_f1": round({self.eval_f1.result() * 100},4),

          }
          print(f'Epoch: {epoch + 1}, Step: {step+1} ', eval_res)

      train_res = {
            "train_loss": {self.train__loss.result()},
            "train_acc": round({self.train__accuracy.result() * 100},4),
            "train_pre": round({self.train__precision.result() * 100},4),
            "train_rec": round({self.train__rec.result() * 100},4),
            "train_f1": round({self.train__f1.result() * 100},4),

          }
      print(f'Epoch: {epoch + 1}, Step: {step+1} ', train_res)

      print("Finish training......")
    
    return train_res.update(eval_res)


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
            "test_loss": {self.test__loss.result()},
            "test_acc": round({self.test__accuracy.result() * 100},4),
            "test_pre": round({self.test__precision.result() * 100},4),
            "test_rec": round({self.test__rec.result() * 100},4),
            "test_f1": round({self.test__f1.result() * 100},4),

          }
    print("Finish testing......")
    
    return test_res

    