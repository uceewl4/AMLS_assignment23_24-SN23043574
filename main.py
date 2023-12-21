# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2023/11/27 11:12:08
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This is the main file where the whole project starts to run.
"""

# here put the import lib
import os
import argparse
import warnings
import tensorflow as tf
from utils import (
    load_data,
    load_model,
    get_metrics,
    hyperpara_selection,
    visual4cm,
    visual4auc,
    visual4tree,
    visual4KMeans,
)
from A.data_preprocessing import data_preprocess4A, load_data_log4A
from B.data_preprocessing import data_preprocess4B, load_data_log4B

warnings.filterwarnings("ignore")

"""
    This is the part for CPU and GPU setting. Notice that part of the project 
    code is run on UCL server with provided GPU resources, especially for NNs 
    and pretrained models.
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export CUDA_VISIBLE_DEVICES=1  # used for setting specific GPU in terminal
if tf.config.list_physical_devices("GPU"):
    print("Use GPU of UCL server: london.ee.ucl.ac.uk")
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Use CPU of your PC.")

if __name__ == "__main__":
    """
    Notice that you can specify certain task and model for experiment by passing in
    arguments. Guidelines for running are provided in README.md and Github link.
    """
    # argument processing
    parser = argparse.ArgumentParser(description="Argparse")
    parser.add_argument("--task", type=str, default="A", help="task A or B")
    parser.add_argument("--method", type=str, default="LR", help="model chosen")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size of NNs like MLP and CNN"
    )
    parser.add_argument("--epochs", type=int, default=10, help="epochs of NNs")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of NNs")
    parser.add_argument(
        "--pre_data", type=bool, default=False, help="whether preprocess the dataset"
    )
    # notice that preprocessing the data may take a long time especially for task B,
    # codes with preprocessed data is provided in the backup project, you can replicate directly to here.
    # or run the backup project (almost the same with this one just with dataset provided).
    parser.add_argument("--npz", type=bool, default=False, help="whether download .npz")
    # notice that two files needed to download for the project, one need terminal command,
    # the other need to specify this argument as True for downloading .npz file.
    # my suggestion is to use my "Datasets" directory directly which is provided in the backup project,
    # but this argument is provided as well if you want to check code validation.
    # Simply run backup project will be better if you just want to test model implementation, where both raw
    # dataset and preprocessed dataset already exist for time consideration.
    parser.add_argument(
        "--multilabel",
        type=bool,
        default=False,
        help="whether consider multilabel setting for task B",
    )
    args = parser.parse_args()
    task = args.task
    method = args.method
    pre_data = args.pre_data
    npz = args.npz
    print(
        f"Method: {method} Task: {task} Multilabel: {args.multilabel}."
    ) if task == "B" and method in ["MLP", "CNN"] else print(
        f"Method: {method} Task: {task}."
    )

    if task == "A":
        raw_path = "Datasets/pneumoniamnist"
    else:
        raw_path = "Datasets/pathmnist"

    # data processing
    if pre_data:
        data_preprocess4A(raw_path) if task == "A" else data_preprocess4B(raw_path)
    else:
        load_data_log4A(npz) if task == "A" else load_data_log4B(npz)

    # load data
    print("Start loading data......")
    if task == "A":
        pre_path = "Outputs/pneumoniamnist/preprocessed_data"
    else:
        pre_path = "Outputs/pathmnist/preprocessed_data"
    if (
        ("LR" in method)
        or ("KNN" in method)
        or ("SVM" in method)
        or ("DT" in method)
        or ("NB" in method)
        or ("RF" in method)
        or ("ABC" in method)
        or ("KMeans" in method)
    ):
        Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data(task, pre_path, method)
    elif method in ["CNN", "MLP", "EnsembleNet"]:
        train_ds, val_ds, test_ds = load_data(
            task, pre_path, method, batch_size=args.batch_size
        )
    print("Load data successfully.")

    # model selection
    # didn't consider individual pre-trained currently
    print("Start loading model......")
    if method in ["MLP", "CNN", "EnsembleNet"]:
        model = load_model(task, method, args.multilabel, args.lr)
    else:
        model = load_model(task, method)
    print("Load model successfully.")

    """
        This part includes all training, validation and testing process with encapsulated functions.
        Detailed process of each method can be seen in corresponding classes.
    """
    if method in ["LR", "KNN", "SVM", "DT", "NB", "RF", "ABC"]:
        if method in ["KNN", "DT", "RF", "ABC"]:
            cv_results_ = model.train(Xtrain, ytrain, Xval, yval, gridSearch=True)
        else:
            model.train(Xtrain, ytrain, Xval, yval)
        pred_train, pred_val, pred_test = model.test(Xtrain, ytrain, Xval, yval, Xtest)

    elif method in ["MLP", "CNN"]:
        if args.multilabel == False:
            train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
                model, train_ds, val_ds, args.epochs
            )
            test_res, pred_test, ytest = model.test(model, test_ds)
        else:  # multilabel
            (
                train_res,
                val_res,
                pred_train,
                pred_train_multilabel,
                pred_val,
                pred_val_multilabel,
                ytrain,
                yval,
            ) = model.train(model, train_ds, val_ds, args.epochs)
            test_res, pred_test, pred_test_multilabel, ytest = model.test(
                model, test_ds
            )
            print(pred_test_multilabel[:5, :])

    elif method == "EnsembleNet":
        model.train(train_ds, val_ds, args.epochs)
        train_res, val_res, pred_train, pred_val, ytrain, yval = model.weight_selection(
            train_ds, val_ds
        )
        test_res, pred_test, ytest = model.test(test_ds)

    elif (
        ("VGG16" in method)
        or ("ResNet50" in method)
        or ("DenseNet201" in method)
        or ("MobileNetV2" in method)
        or ("InceptionV3" in method)
    ):
        if (
            ("KNN" in method)
            or ("DT" in method)
            or ("RF" in method)
            or ("ABC" in method)
        ):
            cv_results_ = model.train(
                model, Xtrain, ytrain, Xval, yval, Xtest, gridSearch=True
            )
        else:
            model.train(model, Xtrain, ytrain, Xval, yval, Xtest)
        pred_train, pred_val, pred_test = model.test(model, ytrain, yval)

    elif method == "KMeans":
        model.train(Xtrain, ytrain)
        pred_train, pred_val, pred_test = model.test(Xtrain, Xval, Xtest)

    # metrics and visualization
    # hyperparameters selection
    if ("KNN" in method) or ("DT" in method) or ("RF" in method) or ("ABC" in method):
        hyperpara_selection(task, method, cv_results_["mean_test_score"])

    # decision tree
    if "DT" in method:
        visual4tree(task, method, model.model) if method == "DT" else visual4tree(
            task, method, model.clf
        )

    # confusion matrix, auc roc curve, metrics calculation
    if method != "KMeans":
        res = {
            "train_res": get_metrics(task, ytrain, pred_train),
            "val_res": get_metrics(task, yval, pred_val),
            "test_res": get_metrics(task, ytest, pred_test),
        }
        for i in res.items():
            print(i)
        visual4cm(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test)
        if task == "A":
            visual4auc(
                task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test
            )
    else:  # clustering 3D figure
        wrap_data = {
            "train": (Xtrain, ytrain),
            "val": (Xval, yval),
            "test": (Xtest, ytest),
            "train_clustering": (Xtrain, pred_train),
            "val_clustering": (Xval, pred_val),
            "test_clustering": (Xtest, pred_test),
        }
        visual4KMeans(task, wrap_data)
