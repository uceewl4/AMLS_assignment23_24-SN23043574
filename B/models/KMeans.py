import numpy as np
from sklearn.cluster import KMeans as kmeans



class KMeans:

    def __init__(self):
        self.model = kmeans(n_clusters=9) 
        
    def train(self, Xtrain, ytrain):

        print(f"Start training for KMeans......")
        self.model.fit(Xtrain, ytrain)  
        print(f"Finish training for KMeans.")

    def test(self, Xtrain, Xval, Xtest):
        pred_train = self.model.predict(Xtrain)

        print(f"Start evaluating for KMeans......")
        pred_val = self.model.predict(Xval)
        print(f"Finish evaluating for KMeans.")

        print(f"Start testing for KMeans......")
        pred_test = self.model.predict(Xtest)
        print(f"Finish testing for KMeans.")
        
        return pred_train, pred_val, pred_test



