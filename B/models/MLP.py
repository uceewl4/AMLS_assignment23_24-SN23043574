# -*- encoding: utf-8 -*-
"""
@File    :   MLP.py
@Time    :   2023/12/16 22:07:02
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for customized network of MLP, including network initialization.
    construction and entire process of training, validation and testing.
"""

# here put the import lib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorboardX import SummaryWriter
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout


class MLP(Model):

    """
    description: This function includes all initialization of MLP, like layers used for construction,
      loss function object, optimizer, measurement of accuracy and loss.
    param {*} self
    param {*} task: task A or B
    param {*} method: MLP
    param {*} multilabel: whether under multilabel setting
    param {*} lr: learning rate
    """

    def __init__(self, task, method, multilabel=False, lr=0.001):
        super(MLP, self).__init__()
        # network layers definition
        self.multilabel = multilabel
        self.flatten = Flatten(input_shape=(28, 28, 3))
        self.d1 = Dense(2048, activation="relu")
        self.d2 = Dense(2048, activation="relu")
        self.do1 = Dropout(0.4)
        self.d3 = Dense(1024, activation="relu")
        self.d4 = Dense(512, activation="relu")
        self.do2 = Dropout(0.4)
        self.d5 = Dense(256, activation="relu")
        self.d6 = Dense(128, activation="relu")
        self.do3 = Dropout(0.2)
        self.d7 = Dense(64, activation="relu")
        self.d8 = Dense(9)  # 9-class classification

        # objective function: sparse categorical cross entropy for mutliclass classification
        # notice that here the loss is calculated from logits, no need to set activation function for the output layer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.lr = lr

        # adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # loss and accuracy
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

        self.val_loss = tf.keras.metrics.Mean(name="eval_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy"
        )

        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )

        self.method = method
        self.task = task

    """
  description: This function is the actual construction process of customized network.
  param {*} self
  param {*} x: input 
  return {*}: output logits
  """

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.do1(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.do2(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.do3(x)
        x = self.d7(x)
        return self.d8(x)

    """
  description: This function is used for the entire process of training. 
    Notice that loss of both train and validation are backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} train_ds: loaded train dataset as batches
  param {*} val_ds: loaded validation dataset as batches
  param {*} EPOCHS: number of epochs
  return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
  """

    def train(self, model, train_ds, val_ds, EPOCHS):
        print("Start training......")
        if not os.path.exists("Outputs/images/nn_curves/"):
            os.makedirs("Outputs/images/nn_curves/")
        writer = SummaryWriter(
            f"Outputs/images/nn_curves/{self.method}_task{self.task}"
        )

        # train
        for epoch in range(EPOCHS):
            train_pred = []  # label prediction
            ytrain = []  # ground truth
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for step, (train_images, train_labels) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    predictions = model(train_images, training=True)  # logits
                    if self.multilabel == False:
                        train_prob = tf.nn.softmax(predictions)  # probabilities
                        train_pred += np.argmax(train_prob, axis=1).tolist()
                    else:  # multilabel
                        train_prob_multilabel = tf.nn.sigmoid(
                            predictions
                        )  # sigmoid probability for each class
                        train_prob = tf.nn.softmax(
                            predictions
                        )  # still calculate multiclass probability

                        # set threshold of multilabel probability as 0.6
                        train_pred_multilabel = np.zeros_like(predictions)
                        train_pred_multilabel[
                            train_prob_multilabel >= 0.6
                        ] = 1  # multilabel predicted labels

                        # based on multiclass classification result, if original label belong to multilabel prediction
                        # then label it with the correct one in multilabel prediction
                        tmp = np.argmax(train_prob, axis=1).tolist()  # ground truth
                        for index, (pred, label) in enumerate(
                            zip(train_pred_multilabel, train_labels)
                        ):
                            if pred[int(label)] == 1:
                                tmp[index] = label
                        train_pred += tmp

                    ytrain += np.array(train_labels).tolist()
                    loss = self.loss_object(train_labels, predictions)

                # backward propagation
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )

                self.train_loss(loss)
                self.train_accuracy(train_labels, predictions)

                # validation
                if step % 600 == 0:
                    val_pred = []
                    yval = []
                    self.val_loss.reset_states()
                    self.val_accuracy.reset_states()

                    for val_images, val_labels in val_ds:
                        with tf.GradientTape() as tape:
                            predictions = model(val_images, training=True)
                            if self.multilabel == False:
                                val_prob = tf.nn.softmax(predictions)
                                val_pred += np.argmax(val_prob, axis=1).tolist()
                            else:  # multilabel
                                val_prob_multilabel = tf.nn.sigmoid(predictions)
                                val_prob = tf.nn.softmax(predictions)

                                val_pred_multilabel = np.zeros_like(predictions)
                                val_pred_multilabel[
                                    val_prob_multilabel >= 0.6
                                ] = 1  # threshold cannot be set to 0.5 considering the distribution

                                tmp = np.argmax(val_prob, axis=1).tolist()
                                for index, (pred, label) in enumerate(
                                    zip(val_pred_multilabel, val_labels)
                                ):
                                    if pred[int(label)] == 1:
                                        tmp[index] = label
                                val_pred += tmp

                            yval += np.array(val_labels).tolist()
                            val_loss = self.loss_object(val_labels, predictions)

                            self.val_loss(val_loss)
                            self.val_accuracy(val_labels, predictions)

                        # backward propagation
                        gradients = tape.gradient(val_loss, model.trainable_variables)
                        self.optimizer.apply_gradients(
                            zip(gradients, model.trainable_variables)
                        )

                        self.val_loss(val_loss)
                        self.val_accuracy(val_labels, predictions)

                    val_res = {
                        "val_loss": np.array(self.val_loss.result()).tolist(),
                        "val_acc": round(np.array(self.val_accuracy.result()) * 100, 4),
                    }
                    print(f"Epoch: {epoch + 1}, Step: {step} ", val_res)

            train_res = {
                "train_loss": np.array(self.train_loss.result()).tolist(),
                "train_acc": round(np.array(self.train_accuracy.result()) * 100, 4),
            }
            print(f"Epoch: {epoch + 1}", train_res)

            writer.add_scalars(
                "loss",
                {
                    "train_loss": np.array(self.train_loss.result()).tolist(),
                    "val_loss": np.array(self.val_loss.result()).tolist(),
                },
                epoch,
            )
            writer.add_scalars(
                "accuracy",
                {
                    "train_accuracy": np.array(self.train_accuracy.result()).tolist(),
                    "val_accuracy": np.array(self.val_accuracy.result()).tolist(),
                },
                epoch,
            )

            train_pred = np.array(train_pred)
            val_pred = np.array(val_pred)

        print("Finish training.")
        writer.close()

        if self.multilabel == False:
            return train_res, val_res, train_pred, val_pred, ytrain, yval
        else:
            return (
                train_res,
                val_res,
                train_pred,
                train_pred_multilabel,
                val_pred,
                val_pred_multilabel,
                ytrain,
                yval,
            )

    """
  description: This function is used for the entire process of testing. 
    Notice that loss of testing is not backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} test_ds: loaded test dataset as batches
  return {*}: accuracy and loss result, predicted labels (multilabel if necessary) and ground truth of test dataset
  """

    def test(self, model, test_ds):
        print("Start testing......")
        test_pred = []  # predicted labels
        ytest = []  # ground truth
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

        for test_images, test_labels in test_ds:
            predictions = model(test_images, training=False)  # logits

            if self.multilabel == False:
                test_prob = tf.nn.softmax(predictions)
                test_pred += np.argmax(test_prob, axis=1).tolist()
            else:  # multilabel
                test_prob_multilabel = tf.nn.sigmoid(
                    predictions
                )  # multilabel probability
                test_prob = tf.nn.softmax(predictions)  # multiclass probability

                # set threshold of multilabel probability as 0.6
                test_pred_multilabel = np.zeros_like(predictions)
                test_pred_multilabel[test_prob_multilabel >= 0.6] = 1

                # based on multiclass classification result, if original label belong to multilabel prediction
                # then label it with the correct one in multilabel prediction
                tmp = np.argmax(test_prob, axis=1).tolist()
                for index, (pred, label) in enumerate(
                    zip(test_pred_multilabel, test_labels)
                ):
                    if pred[int(label)] == 1:
                        tmp[index] = label
                test_pred += tmp

            ytest += np.array(test_labels).tolist()  # ground truth

            t_loss = self.loss_object(test_labels, predictions)
            self.test_loss(t_loss)
            self.test_accuracy(test_labels, predictions)

        test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
            "test_acc": round(np.array(self.test_accuracy.result()) * 100, 4),
        }
        print("Finish testing.")
        test_pred = np.array(test_pred)

        if self.multilabel == False:
            return test_res, test_pred, ytest
        else:
            return test_res, test_pred, test_pred_multilabel, ytest
