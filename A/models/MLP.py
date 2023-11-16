import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MLP(Model):
  def __init__(self):
    super(MLP, self).__init__()
    self.flatten = Flatten(input_shape=(28, 28))
    self.d1 = Dense(128, activation='relu')
    # self.d2 = Dense(1, activation='sigmoid')
    self.d2 = Dense(1)

  def call(self, x):
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


def train(model, train_ds, eval_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS):
  print("Start training")
  for epoch in range(EPOCHS):
    print(f"This is epoch {epoch}")
    train_loss.reset_states()
    train_accuracy.reset_states()
  
    for images, labels in train_ds:
      with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        # print(predictions)
        # print(labels)
        loss = loss_object(labels, predictions)

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_loss(loss)
      train_accuracy(labels, predictions)
    
    print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
  )


def test(model, loss_object, test_loss, test_accuracy, test_ds):
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in test_ds:
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

  print(
    f'Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
  )

  