import os
import cv2
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils import get_metrics, hyperpara_selection, visual4cm, visual4auc, visual4tree, visual4KMeans
from A.data_preprocessing import data_preprocess4A, load_data_log4A
from B.data_preprocessing import data_preprocess4B, load_data_log4B
from utils import load_data, load_model


import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# # data = np.load('/Users/anlly/Desktop/ucl/Applied Machine Learning Systems-I/AMLS assignment/AMLS_assignment23_24-SN23043574/Datasets/pneumoniamnist.npz')
# # print(f"Train data length: {len(data['train_images'])}, label 0: {np.count_nonzero(data['train_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['train_labels'].flatten() == 1)}")
# # print(f"Validation data length: {len(data['val_images'])}, label 0: {np.count_nonzero(data['val_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['val_labels'].flatten() == 1)}")                                                               
# # print(f"Test data length: {len(data['test_images'])}, label 0: {np.count_nonzero(data['test_labels'].flatten() == 0)}, label 1: {np.count_nonzero(data['test_labels'].flatten() == 1)}")
# # Generally the ratio of train:val:test should be 3:1:1, here first use the dataset and no need for train test split

# export CUDA_VISIBLE_DEVICES=1
os.environ['CUDA_VISIBLE_DEVICES']='0'
# 8.9.6, 12.0.1
if tf.config.list_physical_devices('GPU'):
    print('Use GPU of UCL server: london.ee.ucl.ac.uk')
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print('Use CPU of your PC.')

if __name__ == '__main__':

    # argument processing
    parser = argparse.ArgumentParser(description='Argparse')
    parser.add_argument('--task',type=str, default = "A",required=True,help="")
    parser.add_argument('--method',type=str, default="", required=True,help='age of the programmer')
    parser.add_argument('--batch_size',type=int, default=32,help='age of the programmer')
    parser.add_argument('--epochs',type=int, default=10,help='age of the programmer')
    parser.add_argument('--lr', type=float, default=0.001, help="preprocess the data or use the file provided")
    parser.add_argument('--pre_data', type=bool, default=False, help="preprocess the data or use the file provided")
    parser.add_argument('--multilabel', type=bool, default=False, help="preprocess the data or use the file provided")
    args = parser.parse_args()
    task = args.task
    method = args.method
    pre_data = args.pre_data
    print(f"Method: {method} Task: {task} Multilabel: {args.multilabel}.") if task == "B" and method in ["MLP","CNN"] else print(f"Method: {method} Task: {task}.")

    if task == "A":
        raw_path = "Datasets/pneumoniamnist"
    else:
        raw_path = "Datasets/pathmnist"
     
    # data processing (haven't decide how to present yet)
    if pre_data:
        data_preprocess4A(raw_path) if task == "A" else data_preprocess4B(raw_path)
    else:
        load_data_log4A() if task == "A" else load_data_log4B()
    
    # load data
    print("Start loading data......")
    if task == "A":
        pre_path = 'Outputs/pneumoniamnist/preprocessed_data'
    else:
        pre_path = 'Outputs/pathmnist/preprocessed_data'
        
    if ("LR" in method) or ("KNN" in method) or ("SVM" in method) or ("DT" in method) \
        or ("NB" in method) or ("RF" in method) or ("ABC" in method) or ("KMeans" in method):
        Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data(task,pre_path,method)
        # A SVM 6988 784   densenet 6988 28 28 3
        # B  2352              densenet 28,28,3
    elif method in ["CNN","MLP","EnsembleNet"]:
        train_ds, val_ds, test_ds = load_data(task,pre_path,method,batch_size=args.batch_size)
    print("Load data successfully.")
    
    # model selection
    # didn't consider individual pre-trained currently
    print("Start loading model......")
    if method in ["MLP","CNN"]:
        model = load_model(task, method, args.multilabel,args.lr)
    else:
        model = load_model(task, method, args.multilabel)
    print("Load model successfully.")
    
    if method in ["LR","KNN","SVM","DT","NB","RF","ABC"]:  
        if method in ["KNN","DT","RF","ABC"]:
            cv_results_ = model.train(Xtrain, ytrain, Xval, yval, gridSearch=True)
        else:
            model.train(Xtrain, ytrain, Xval, yval)
        pred_train, pred_val, pred_test = model.test(Xtrain, ytrain, Xval, yval, Xtest)

    elif method in ["MLP","CNN"]:
        if args.multilabel == False:
            train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(model, train_ds, val_ds, args.epochs)
            test_res, pred_test, ytest = model.test(model, test_ds)
        else:  # multilabel
            train_res, val_res, pred_train,pred_train_multilabel, pred_val, pred_val_multilabel,ytrain, yval = model.train(model, train_ds, val_ds, args.epochs)
            test_res, pred_test, pred_test_multilabel, ytest = model.test(model, test_ds)
            print(pred_test_multilabel[:5,:])
    
    elif method == "EnsembleNet":
        model.train(train_ds, val_ds, args.epochs)
        train_res, val_res, pred_train, pred_val, ytrain, yval = model.weight_selection(train_ds,val_ds)
        test_res, pred_test, ytest = model.test(test_ds)
       
    elif (("VGG16" in method) or ("ResNet50" in method) or ("DenseNet201" in method) \
        or ("MobileNetV2" in method) or ("InceptionV3" in method)):
        if (("KNN" in method) or ("DT" in method) or ("RF" in method) or ("ABC" in method)):
            cv_results_ = model.train(model, Xtrain, ytrain, Xval, yval, Xtest, gridSearch=True)
        else:
            model.train(model, Xtrain, ytrain, Xval, yval, Xtest)
        pred_train, pred_val, pred_test = model.test(model, ytrain, yval)
    
    elif method == "KMeans":
        model.train(Xtrain, ytrain)
        pred_train, pred_val, pred_test = model.test(Xtrain, Xval, Xtest)
    
    # visualization
    if (("KNN" in method) or ("DT" in method) or ("RF" in method) or ("ABC" in method)):
        hyperpara_selection(task, method, cv_results_["mean_test_score"])
    if "DT" in method:
        visual4tree(task,method,model.model) if method == "DT" else visual4tree(task,method,model.clf)

    if method != "KMeans":
        res = {"train_res":get_metrics(task, ytrain, pred_train),
               "val_res":get_metrics(task, yval, pred_val),
               "test_res":get_metrics(task, ytest, pred_test)}
        for i in res.items():
            print(i)
        visual4cm(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test)
        if task == "A":
            visual4auc(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test)
    else:
        wrap_data = {"train":(Xtrain,ytrain), "val":(Xval,yval), "test": (Xtest,ytest),
            "train_clustering":(Xtrain, pred_train),
            "val_clustering":(Xval, pred_val),
            "test_clustering":(Xtest, pred_test)}
        visual4KMeans(task, wrap_data)
   

