import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor
from sklearn.datasets import make_moons

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_one_score(model, X_train, X_test, y_train, y_test, F1=False):
    if isinstance(model, Sequential):
        if F1:
            y_pred_train = model.predict(X_train)
            y_pred_test = pd.DataFrame(model.predict(X_test),columns=y_test.columns)
            y_pred_train = pd.DataFrame(np.where(y_pred_train > 0.5, 1, 0),columns=y_train.columns)
            y_pred_test = pd.DataFrame(np.where(y_pred_test > 0.5, 1, 0),columns=y_train.columns)
            f1_train = f1_score(y_train, y_pred_train, average='weighted')
            f1_test = f1_score(y_test, y_pred_test, average='weighted')
            return f1_train, f1_test
        else:
            accuracy_train = model.evaluate(X_train, y_train, verbose=0)[1]
            accuracy_test = model.evaluate(X_test, y_test, verbose=0)[1]
            return accuracy_train, accuracy_test
    elif hasattr(model, 'score'):
        if F1:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            f1_train = f1_score(y_train, y_pred_train, average='weighted')
            f1_test = f1_score(y_test, y_pred_test, average='weighted')
            return f1_train, f1_test
        else:
            accuracy_train = model.score(X_train, y_train)
            accuracy_test = model.score(X_test, y_test)
            return accuracy_train, accuracy_test
    else:
        raise ValueError("Model does not support evaluation with `evaluate` or `score`.")


def get_mean_score(models,data_train, data_test, F1=False):
    accuracy_train_list = []
    accuracy_test_list = []
    for i, _ in enumerate(data_train):
        X_train, y_train = data_train[i].values()
        X_test, y_test = data_test[i].values()
        accuracy_train, accuracy_test = get_one_score(models[i], X_train, X_test, y_train, y_test, F1)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
    accuracy_train_mean = np.mean(accuracy_train_list)
    accuracy_test_means = np.mean(accuracy_test_list)
    return accuracy_train_mean, accuracy_test_means

def compare_mean_score(models_list, data_train, data_test, F1=False):
    accuracy_train_means = []
    accuracy_test_means = []
    for models in models_list:
        accuracy_train_mean, accuracy_test_mean = get_mean_score(models, data_train, data_test,F1)
        accuracy_train_means.append(accuracy_train_mean)
        accuracy_test_means.append(accuracy_test_mean)
    return accuracy_train_means, accuracy_test_means
