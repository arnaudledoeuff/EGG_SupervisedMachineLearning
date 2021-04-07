

# general python libraries
from scipy.io import arff
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# sklearn functions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def My_Cross_Validation (features, target, model):
    scores = list()
    kfold = KFold(10, shuffle=False)
    for train, validate in kfold.split(features):
        # define train/validate sets
        X_cross_train = features[train, :]
        y_cross_train = target[train]
        X_cross_validate = features[validate, :]
        y_cross_validate =  target[validate]
        # fit model on train set
        model.fit(X_cross_train, y_cross_train)
        # forecast test set
        prediction = model.predict(X_cross_validate)
        # evaluate predictions
        score = accuracy_score(y_cross_validate, prediction)
        # store
        ponderation = (X_cross_validate.shape[0]/features.shape[0])*10
        scores.append(score*ponderation)
    # calculate mean score across each run
    return(np.mean(scores))

def My_Walk_Forward_Validation (features, target, model, time_window):
    # define train/validate sets
    split_size = round(features.shape[0]*0.9)
    X_wf_train = features[:split_size, :]
    y_wf_train = target[:split_size]
    X_wf_validate = features[split_size:, :]
    y_wf_validate = target[split_size:]
    
    # walk-forward validation
    historyX, historyy = [x for x in X_wf_train], [x for x in y_wf_train]
    predictions = list()
    for i in range(len(y_wf_validate)):
        # fit model on a small subset of the train set
        tmpX, tmpy = np.array(historyX)[-time_window:,:], np.array(historyy)[-time_window:]
        model.fit(tmpX, tmpy)
        # forecast the next time step
        prediction = model.predict([X_wf_validate[i, :]])[0]
        # store prediction
        predictions.append(prediction)
        # add real observation to history
        historyX.append(X_wf_validate[i, :])
        historyy.append(y_wf_validate[i])
    # evaluate predictions
    score = accuracy_score(y_wf_validate, predictions)
    return score