'''
A module to collect useful functions we reuse in various notebooks
'''


# We will reuse this function. Hence we copy it to mads_dl.py


# from 03_1_mlp_intro
def count_params(model):
    '''
    Return the number of trainable parameters of a PyTorch Module (model)
    Iterate each of the modules parameters and counts them 
    if they require grad (if they are trainable)
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)                                               


from sklearn import metrics
import torch

# from 03_2_binary_classification
def compute_acc(model, X, y):
    '''
    compute the accuracy of a model for given features X and labels y
    '''
    model.eval()
    with torch.no_grad():
        y_pred=model.predict(X)
    return metrics.accuracy_score(y, y_pred)


# 03_and_04_regression_insurance
def predict(model, X):
    '''
    Use the model to predict for the values in the test set.
    Return the prediction
    '''
    model.eval()
    with torch.no_grad():
        return model(X)


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd

def add_regression_eval(results, algorithm, y_train, y_train_pred, y_test, y_test_pred, num_params):
    '''
    Create a table with evaluation results
    of a regression experiment
    '''
    for dataset, actual, predicted in zip(("train", "test"), (y_train, y_test), (y_train_pred, y_test_pred)):
        results= pd.concat([results, pd.DataFrame([{
            "algorithm": algorithm, 
            "dataset": dataset,
            "MSE": mean_squared_error(actual, predicted),
            "MAE": mean_absolute_error(actual, predicted),
            "MAPE": mean_absolute_percentage_error(actual, predicted)*100, # implemented is relative to 1 not to 100
            "params": num_params
        }])], ignore_index=True)   
    return results