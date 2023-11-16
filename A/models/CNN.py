import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Model

class CNN(Model):
  def __init__(self):
    super(CNN, self).__init__()
    # different to MLP, no need to make the image input in a flattened way, can just input image with batches and calculate with kernel
    # but for MLP we need to first flatten the image because it's not a kernel calculation
    self.c1 = Conv2D(16, 3, padding='same', activation='relu', input_shape=(28,28,1))
    self.p1 = MaxPooling2D()
    self.c2 = Conv2D(32, 3, padding='same', activation='relu')
    self.p2 = MaxPooling2D()
    self.c3 = Conv2D(64, 3, padding='same', activation='relu')
    self.p3 = MaxPooling2D()
    self.dropout = Dropout(0.2)
    self.fc = Flatten()
    self.d1 = Dense(64, activation='relu')
    self.d2 = Dense(1, name="outputs")
  
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

  