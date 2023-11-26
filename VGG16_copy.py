import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

class VGG16_copy(Model):
  def __init__(self):
    super(VGG16_copy, self).__init__()

    self.data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Resizing(32,32,interpolation='bilinear'),
        tf.keras.layers.Rescaling(1./255, input_shape=(32, 32, 3)),
    ])
    self.base_model =tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                  input_shape=(32, 32, 3)) 
    self.base_model.trainable = True
    print("Number of layers in the base model: ", len(self.base_model.layers))
    # fine_tune_at = 6

    # Freeze all the layers before the `fine_tune_at` layer
    
    # for layer in self.base_model.layers[:fine_tune_at]:
    #     layer.trainable = False

    self.base_model.summary()

    self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    self.dropout = tf.keras.layers.Dropout(0.2)
    self.prediction_layer = tf.keras.layers.Dense(1)

  def call(self, x):
    x = self.data_augmentation(x)
    x = self.base_model(x, training=True)
    x = self.global_average_layer(x)
    x = self.dropout(x)

    return self.prediction_layer(x)


  def train(self,model, train_ds, val_ds):
      print("Start training")
      base_learning_rate = 0.001
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

      history = model.fit(train_ds,
                      epochs=20,
                      validation_data=val_ds)
      

  def test(self,model, test_ds):
      loss0, accuracy0 = model.evaluate(test_ds)

