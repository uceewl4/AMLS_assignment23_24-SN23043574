import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class MobileNetV2(Model):
  def __init__(self):
    super(MobileNetV2, self).__init__()
    # self.flatten = Flatten(input_shape=(28, 28, 3))
    # self.d1 = Dense(128, activation='relu')
    # # self.d2 = Dense(1, activation='sigmoid')
    # self.d2 = Dense(9)

    self.data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(32,32,interpolation='bilinear'),
        tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 3)),
    ])
    self.base_model = tf.keras.applications.MobileNetV2(input_shape=(32,32,3),  # resize into 32x32 to satisfy the requirement of MobileNetV2
                                               include_top=False,
                                               weights='imagenet')
    self.base_model.trainable = False
    self.base_model.summary()

    self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.prediction_layer = tf.keras.layers.Dense(9)

  def call(self, x):
    x = self.data_augmentation(x)
    x = self.base_model(x, training=False)
    x = self.global_average_layer(x)
    x = self.dropout(x)

    return self.prediction_layer(x)


def train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS):
    print("Start training")
    #   for epoch in range(EPOCHS):
    #     print(f"This is epoch {epoch}")
    #     train_loss.reset_states()
    #     train_accuracy.reset_states()
    
    #     for images, labels in train_ds:
    #       with tf.GradientTape() as tape:
    #         print(images.shape)
    #         predictions = model(images, training=True)
    #         # print(predictions)
    #         # print(labels)
    #         loss = loss_object(labels, predictions)

    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    
    #     print(
    #     f'Epoch {epoch + 1}, '
    #     f'Loss: {train_loss.result()}, '
    #     f'Accuracy: {train_accuracy.result() * 100}, '
    #   )

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_ds,
                    epochs=100,
                    validation_data=val_ds)
    


def test(model, loss_object, test_loss, test_accuracy, test_ds):
#   test_loss.reset_states()
#   test_accuracy.reset_states()

#   for images, labels in test_ds:
#     predictions = model(images, training=False)
#     t_loss = loss_object(labels, predictions)

#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
    loss0, accuracy0 = model.evaluate(test_ds)

#   print(
#     f'Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}'
#   )

  
# def train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS):
#     print("Start training")
#     for epoch in range(EPOCHS):
#         print(f"This is epoch {epoch}")
#         train_loss.reset_states()
#         train_accuracy.reset_states()
    
#         for images, labels in train_ds:
#           with tf.GradientTape() as tape:
#             # print(images.shape)
#             predictions = model(images, training=True)
#             # print(predictions)
#             # print(labels)
#             loss = loss_object(labels, predictions)

#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#         train_loss(loss)
#         train_accuracy(labels, predictions)
    
#         print(
#         f'Epoch {epoch + 1}, '
#         f'Loss: {train_loss.result()}, '
#         f'Accuracy: {train_accuracy.result() * 100}, '
#       )

#     # base_learning_rate = 0.0001
#     # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
#     #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     #             metrics=['accuracy'])

#     # history = model.fit(train_ds,
#     #                 epochs=10,
#     #                 validation_data=val_ds)
    


# def test(model, loss_object, test_loss, test_accuracy, test_ds):
#   test_loss.reset_states()
#   test_accuracy.reset_states()

#   for images, labels in test_ds:
#     predictions = model(images, training=False)
#     t_loss = loss_object(labels, predictions)

#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
#     # loss0, accuracy0 = model.evaluate(test_ds)

#   print(
#     f'Loss: {test_loss.result()}, '
#     f'Test Accuracy: {test_accuracy.result() * 100}'
#   )

  