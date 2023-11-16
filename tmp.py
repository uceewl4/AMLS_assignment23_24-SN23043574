import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from A.models.MLP import MLP as A_MLP
from B.models.MLP import MLP as B_MLP
from A.models.MLP import test as A_test
from A.models.MLP import train as A_train
from B.models.MLP import test as B_test
from B.models.MLP import train as B_train
# from B.models.CNN import CNN, train, test
from A.models.CNN import CNN as A_CNN
from A.models.CNN import train as A_CNN_train
from A.models.CNN import test as A_CNN_test
# from B.models.MobileNetV2 import MobileNetV2, train, test
# from A.models.MobileNetV2 import MobileNetV2, train, test
from B.models.VGG16 import VGG16, train, test
# from B.models.ResNet50 import ResNet50, train, test
from B.models.DenseNet201 import DenseNet201, train, test

from A.models.baselines import Baselines as A_baselines
from B.models.baselines import Baselines as B_baselines
from utils import visual4cm


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from A.models.VGG16_SVM import VGG16_SVM, train, test

# # data = np.load('/Users/anlly/Desktop/ucl/Applied Machine Learning Systems-I/AMLS assignment/AMLS_assignment23_24-SN23043574/Datasets/pneumoniamnist.npz')
# # print(f"Train data length: {len(data['train_images'])}, label 0: {np.count_nonzero(data['train_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['train_labels'].flatten() == 1)}")
# # print(f"Validation data length: {len(data['val_images'])}, label 0: {np.count_nonzero(data['val_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['val_labels'].flatten() == 1)}")                                                               
# # print(f"Test data length: {len(data['test_images'])}, label 0: {np.count_nonzero(data['test_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['test_labels'].flatten() == 1)}")
# # Generally the ratio of train:val:test should be 3:1:1, here first use the dataset and no need for train test split


# visualization, result are all separate from model

# path = 'Outputs/pneumoniamnist/preprocessed_data'
# file=os.listdir(path)
# Xtest = []
# ytest = []
# Xtrain = []
# ytrain = []
# Xval = []
# yval = []

# for index,f in enumerate(file):
#         if not os.path.isfile(os.path.join(path,f)):
#             continue
#         else:
#             img = cv2.imread(os.path.join(path,f))
#             imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             if "test" in f:
#                 Xtest.append(imgGray)
#                 ytest.append(f.split("_")[1][0])
#             elif "train" in f:
#                 Xtrain.append(imgGray)
#                 ytrain.append(f.split("_")[1][0])
#             elif "val" in f:
#                 Xval.append(imgGray)
#                 yval.append(f.split("_")[1][0])

# path = 'Outputs/pneumoniamnist/preprocessed_data'
# file=os.listdir(path)
# Xtest = []
# ytest = []
# Xtrain = []
# ytrain = []
# Xval = []
# yval = []

# for index,f in enumerate(file):
#         if not os.path.isfile(os.path.join(path,f)):
#             continue
#         else:
#             img = cv2.imread(os.path.join(path,f))
#             # imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             if "test" in f:
#                 Xtest.append(img)
#                 ytest.append(f.split("_")[1][0])
#             elif "train" in f:
#                 Xtrain.append(img)
#                 ytrain.append(f.split("_")[1][0])
#             elif "val" in f:
#                 Xval.append(img)
#                 yval.append(f.split("_")[1][0])

# n,h,w,c = np.array(Xtrain).shape
# Xtrain = np.array(Xtrain)
# Xval = np.array(Xval)
# Xtest = np.array(Xtest)

# n,h,w = np.array(Xtrain).shape
# Xtrain = np.array(Xtrain).reshape(n,h*w) # need to reshape gray picture into two-dimensional ones
# Xval = np.array(Xval).reshape(len(Xval),h*w)
# Xtest = np.array(Xtest).reshape(len(Xtest),h*w)

# n,h,w = np.array(Xtrain).shape
# Xtrain = np.array(Xtrain)
# Xval = np.array(Xval)
# Xtest = np.array(Xtest)
# # logistic regression
# # LR = A_baselines("LR")  # lbfgs will have warning about max iteration limitation
# # LR.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = LR.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # knn = A_baselines("KNN")  
# # res = knn.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = knn.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # svm = A_baselines("SVM")  
# # svm.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = svm.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# dt = A_baselines("DT")  
# dt.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = dt.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))
# visual4cm(np.array(ytest).astype(int), pred_test.astype(int))

# # nb = A_baselines("NB")  
# # nb.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = nb.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # rf = A_baselines("RF")  
# # rf.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = rf.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# # abc = A_baselines("ABC")  
# # abc.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# # pred_train, pred_val, pred_test = abc.test(Xtrain, ytrain, Xval, yval, Xtest)
# # acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# # pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# # recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# # f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# # print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# path = 'Outputs/pathmnist/preprocessed_data'
# file=os.listdir(path)
# Xtest = []
# ytest = []
# Xtrain = []
# ytrain = []
# Xval = []
# yval = []

# for index,f in enumerate(file):
#         if not os.path.isfile(os.path.join(path,f)):
#             continue
#         else:
#             img = cv2.imread(os.path.join(path,f))
#             if "test" in f:
#                 Xtest.append(img)
#                 ytest.append(f.split("_")[1][0])
#             elif "train" in f:
#                 Xtrain.append(img)
#                 ytrain.append(f.split("_")[1][0])
#             elif "val" in f:
#                 Xval.append(img)
#                 yval.append(f.split("_")[1][0])

# # n,h,w,c = np.array(Xtrain).shape
# # Xtrain = np.array(Xtrain).reshape(n,h*w*c) # need to reshape gray picture into two-dimensional ones
# # Xval = np.array(Xval).reshape(len(Xval),h*w*c)
# # Xtest = np.array(Xtest).reshape(len(Xtest),h*w*c)

# n,h,w,c = np.array(Xtrain).shape
# Xtrain = np.array(Xtrain) # need to reshape gray picture into two-dimensional ones
# Xval = np.array(Xval)
# Xtest = np.array(Xtest)

# logistic regression


# knn = B_baselines("KNN")  
# res = knn.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = knn.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# svm = B_baselines("SVM")  
# svm.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = svm.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# dt = A_baselines("DT")  
# dt.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = dt.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))
# visual4cm(np.array(ytest).astype(int), pred_test.astype(int))

# nb = A_baselines("NB")  
# nb.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = nb.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# rf = A_baselines("RF")  
# rf.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = rf.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))

# abc = A_baselines("ABC")  
# abc.train(Xtrain, ytrain, Xval, yval, gridSearch=False)
# pred_train, pred_val, pred_test = abc.test(Xtrain, ytrain, Xval, yval, Xtest)
# acc = accuracy_score(np.array(ytest).astype(int), pred_test.astype(int))
# pre = precision_score(np.array(ytest).astype(int), pred_test.astype(int))
# recall = recall_score(np.array(ytest).astype(int), pred_test.astype(int))
# f1 = f1_score(np.array(ytest).astype(int), pred_test.astype(int))
# print("acc: {%.4f}, pre: {%.4f}, rec: {%.4f}, f1: {%.4f}" % ((acc, pre, recall, f1)))



# # 整体思路框架
# # 数据问题
# # 需要在这里选择模型，是task A还是task B，所有模型都放在A/B对应的构建中


# # MLP for task A
# batch_size = 32
# img_height = 28
# img_width = 28


# train_ds = tf.data.Dataset.from_tensor_slices(
#     (Xtrain, np.array(ytrain).astype(int))).batch(batch_size)
# val_ds = tf.data.Dataset.from_tensor_slices((Xval, np.array(yval).astype(int))).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((Xtest, np.array(ytest).astype(int))).batch(batch_size)
# normalization_layer = layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
# # whether go through the softmax or sigmoid function automatically, if False, means you need to calculate by yourself

# # you can use two ways to solve with loss
# # from_logits=True, dense layer without activation (more professional)
# # from_logits=False, dense layer with activation or just add a sigmoid layer

# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

# model = A_MLP()
# EPOCHS = 10
# A_train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS)
# A_test(model, loss_object, test_loss, test_accuracy, test_ds)


# # MLP for task B
# batch_size = 32
# img_height = 28
# img_width = 28
# img_depth = 3


# train_ds = tf.data.Dataset.from_tensor_slices(
#     (Xtrain, np.array(ytrain).astype(int))).batch(batch_size)
# val_ds = tf.data.Dataset.from_tensor_slices((Xval, np.array(yval).astype(int))).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((Xtest, np.array(ytest).astype(int))).batch(batch_size)
# normalization_layer = layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
# # whether go through the softmax or sigmoid function automatically, if False, means you need to calculate by yourself

# # you can use two ways to solve with loss
# # from_logits=True, dense layer without activation (more professional)
# # from_logits=False, dense layer with activation or just add a sigmoid layer

# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# model = B_MLP()
# EPOCHS = 10
# B_train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS)
# B_test(model, loss_object, test_loss, test_accuracy, test_ds)


# CNN for task A
# batch_size = 32
# img_height = 28
# img_width = 28
# img_depth =1


# train_ds = tf.data.Dataset.from_tensor_slices(
#     (Xtrain[..., tf.newaxis], np.array(ytrain).astype(int))).batch(batch_size)
# val_ds = tf.data.Dataset.from_tensor_slices((Xval[..., tf.newaxis], np.array(yval).astype(int))).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((Xtest[..., tf.newaxis], np.array(ytest).astype(int))).batch(batch_size)
# normalization_layer = layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
# print(train_ds)

# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)  
# whether go through the softmax or sigmoid function automatically, if False, means you need to calculate by yourself

# you can use two ways to solve with loss
# from_logits=True, dense layer without activation (more professional)
# from_logits=False, dense layer with activation or just add a sigmoid layer

# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

# model = A_CNN()
# EPOCHS = 35
# A_CNN_train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS)
# A_CNN_test(model, loss_object, test_loss, test_accuracy, test_ds)


# # CNN for task B
# batch_size = 32
# img_height = 28
# img_width = 28
# img_depth = 3


# train_ds = tf.data.Dataset.from_tensor_slices(
#     (Xtrain, np.array(ytrain).astype(int))).batch(batch_size)
# val_ds = tf.data.Dataset.from_tensor_slices((Xval, np.array(yval).astype(int))).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((Xtest, np.array(ytest).astype(int))).batch(batch_size)
# normalization_layer = layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
# print(train_ds)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  
# # whether go through the softmax or sigmoid function automatically, if False, means you need to calculate by yourself

# # you can use two ways to solve with loss
# # from_logits=True, dense layer without activation (more professional)
# # from_logits=False, dense layer with activation or just add a sigmoid layer

# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# model = CNN()
# EPOCHS = 10
# train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS)
# test(model, loss_object, test_loss, test_accuracy, test_ds)



# batch_size = 32
# img_height = 28
# img_width = 28
# img_depth = 3


# # train_ds = tf.data.Dataset.from_tensor_slices(
# #     (Xtrain[:256,:,:,:], np.array(ytrain[:256]).astype(int))).batch(batch_size)
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (Xtrain, np.array(ytrain).astype(int))).batch(batch_size)
# val_ds = tf.data.Dataset.from_tensor_slices((Xval, np.array(yval).astype(int))).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((Xtest, np.array(ytest).astype(int))).batch(batch_size)
# normalization_layer = layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
# print(train_ds)


# model = MobileNetV2()
# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
# EPOCHS = 10
# train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS)
# test(model, loss_object, test_loss, test_accuracy, test_ds)

# batch_size = 32
# img_height = 28
# img_width = 28
# img_depth = 3

# # Mobile task A
# # train_ds = tf.data.Dataset.from_tensor_slices(
# #     (Xtrain[:256,:,:,:], np.array(ytrain[:256]).astype(int))).batch(batch_size)
# train_ds = tf.data.Dataset.from_tensor_slices(
#     (Xtrain, np.array(ytrain).astype(int))).batch(batch_size)
# val_ds = tf.data.Dataset.from_tensor_slices((Xval, np.array(yval).astype(int))).batch(batch_size)
# test_ds = tf.data.Dataset.from_tensor_slices((Xtest, np.array(ytest).astype(int))).batch(batch_size)
# normalization_layer = layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
# print(train_ds)


# model = VGG16()
# optimizer = tf.keras.optimizers.Adam()

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
# EPOCHS = 10
# train(model, train_ds, val_ds, train_loss, train_accuracy, loss_object, optimizer, EPOCHS)
# test(model, loss_object, test_loss, test_accuracy, test_ds)

# path = 'Outputs/pneumoniamnist/preprocessed_data'
# file=os.listdir(path)
# Xtest = []
# ytest = []
# Xtrain = []
# ytrain = []
# Xval = []
# yval = []

# for index,f in enumerate(file):
#         if not os.path.isfile(os.path.join(path,f)):
#             continue
#         else:
#             img = cv2.imread(os.path.join(path,f))
#             # imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#             if "test" in f:
#                 Xtest.append(img)
#                 ytest.append(f.split("_")[1][0])
#             elif "train" in f:
#                 Xtrain.append(img)
#                 ytrain.append(f.split("_")[1][0])
#             elif "val" in f:
#                 Xval.append(img)
#                 yval.append(f.split("_")[1][0])

# n,h,w,c = np.array(Xtrain).shape
# Xtrain = np.array(Xtrain)
# Xval = np.array(Xval)
# Xtest = np.array(Xtest)

# batch_size = 32
# img_height = 28
# img_width = 28
# img_depth = 3


# vgg_svm = VGG16_SVM()
# train(vgg_svm, Xtrain, ytrain)
# test(vgg_svm, Xtest, ytest)

path = 'Outputs/pneumoniamnist/preprocessed_data'
file=os.listdir(path)
Xtest = []
ytest = []
Xtrain = []
ytrain = []
Xval = []
yval = []

for index,f in enumerate(file):
        if not os.path.isfile(os.path.join(path,f)):
            continue
        else:
            img = cv2.imread(os.path.join(path,f))
            # imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if "test" in f:
                Xtest.append(img)
                ytest.append(f.split("_")[1][0])
            elif "train" in f:
                Xtrain.append(img)
                ytrain.append(f.split("_")[1][0])
            elif "val" in f:
                Xval.append(img)
                yval.append(f.split("_")[1][0])

n,h,w,c = np.array(Xtrain).shape
Xtrain = np.array(Xtrain)
Xval = np.array(Xval)
Xtest = np.array(Xtest)

batch_size = 32
img_height = 28
img_width = 28
img_depth = 3


vgg_svm = DenseNet201()
train(vgg_svm, Xtrain, ytrain)
test(vgg_svm, Xtest, ytest)



if __name__ == '__main__':
     
    # data processing (haven't decide how to present yet)


    # model selection

    
