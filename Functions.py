import math, random
import pandas as pd
#from sklearn.datasets import load_files
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn import metrics ###for accuracy calculation we import scikit-learn metrics module
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import sys
sys.path.append(".")
from ImageData import *

def LoadData(type="DataFrame,Array",use_PCA="Yes,No", PCA_Features_num = 900):
    # Load Data
    train_folder = "data/train/"
    X, y =Load_Image_Dataset(train_folder)

    # Get training Images Features 
    X = extractfeatures(X)

    ##### Use Features Reduction
    if(use_PCA=="Yes"):
        X = PCA_FeatureReduction(X ,Features_num , X)

    if(type=="DataFrame"):
        ### Convert to DataFrame
        X= pd.DataFrame(X)
        y= pd.DataFrame(y)

        ### Convert the columns from Integer to String
        X.columns = X.columns.map(str)
        y.columns = y.columns.map(str)
    elif(type=="Array"):
        X = np.array(X)
        y = np.array(y)

    return X, y

def Test_Model_with_Folds(model, X, y, n_splits=10):
    ## https://towardsdatascience.com/how-to-check-if-a-classification-model-is-overfitted-using-scikit-learn-148b6b19af8b
    X= pd.DataFrame(X)
    y= pd.DataFrame(y)
    kf = KFold(n_splits, random_state=100, shuffle=True)
    mae_train = []
    mae_test = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train.values.ravel())
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        mae_train.append(mean_absolute_error(y_train, y_train_pred))
        mae_test.append(mean_absolute_error(y_test, y_test_pred))

    folds = range(1, kf.get_n_splits() + 1)
    plt.plot(folds, mae_train, 'o-', color='green', label='train')
    plt.plot(folds, mae_test, 'o-', color='red', label='test')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of fold')
    plt.ylabel('Mean Absolute Error')
    plt.show()

def Test_Model_Accuracy_with_Folds(model, X, y, n_splits=10):
    ## https://towardsdatascience.com/how-to-check-if-a-classification-model-is-overfitted-using-scikit-learn-148b6b19af8b
    X= pd.DataFrame(X)
    y= pd.DataFrame(y)
    kf = KFold(n_splits, random_state=100, shuffle=True)
    accuracy_score_train = []
    accuracy_score_test = []
    for train_index, test_index in kf.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train.values.ravel())
        y_pred_Train = model.predict(X_train)
        y_pred_Test = model.predict(X_test)
        accuracy_score_train.append(metrics.accuracy_score(y_train, y_pred_Train)*100)
        accuracy_score_test.append(metrics.accuracy_score(y_test, y_pred_Test)*100)

    folds = range(1, kf.get_n_splits() + 1)
    plt.plot(folds, accuracy_score_train, 'o-', color='green', label='train')
    plt.plot(folds, accuracy_score_test, 'o-', color='red', label='test')
    plt.legend()
    plt.grid()
    plt.xlabel('Number of fold')
    plt.ylabel('Accuracy Score')
    plt.show()


def Test_Accuracy_vs_DataSize(model,X,y):

   test_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
   accuracy_score_train = []
   accuracy_score_test = []
   for size in test_size:
      X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, test_size=size, random_state=42)
      tree_clas.fit(X_Train, y_Train.values.ravel())
      y_pred_Train = tree_clas.predict(X_Train)
      y_pred_Test = tree_clas.predict(X_Test)

      # Model Accuracy
      accuracy_score_train.append(metrics.accuracy_score(y_Train, y_pred_Train)*100)
      accuracy_score_test.append(metrics.accuracy_score(y_Test, y_pred_Test)*100)

   plt.plot(np.multiply(test_size,100), accuracy_score_train, 'o-', color='green', label='train')
   plt.plot(np.multiply(test_size,100), accuracy_score_test, 'o-', color='red', label='test')
   plt.legend()
   plt.grid()
   plt.xlabel('% Test Data size')
   plt.ylabel('Accuracy Score')
   plt.show()