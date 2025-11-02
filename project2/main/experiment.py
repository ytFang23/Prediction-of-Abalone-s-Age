import numpy as np
import pandas as pd
import json

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
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

from data_process import get_data, data_init, one_hot_encoding
from fit_model import decision_tree, decision_tree_hyper_search

from imblearn.over_sampling import SMOTE

def experiment_on_decision_tree(dataset_name, file_path):
    X, y, dataset = get_data(dataset_name, file_path)
    X_copy = pd.DataFrame(columns=X.columns)
    X, X_copy = one_hot_encoding(X, X_copy)

    best_model, model_params, model_scores = decision_tree_hyper_search(X, y)

    sorted_indices = np.argsort(model_scores)
    start_idx = int(len(model_scores) * 0.4)
    end_idx = int(len(model_scores) * 0.9)
    idx = np.linspace(start_idx, end_idx, num=10, dtype=int)
    indices = [sorted_indices[i] for i in idx]
    params_list = [model_params[i] for i in indices]
    with open(f'../settings/decision_tree_{dataset_name}.json', 'w') as file:
        json.dump(params_list, file, indent=4)

def best_decision_tree_setting(dataset_name, file_path):
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    df = df.apply(lambda col: col.astype(int) if col.dtype in ['float64', 'float32', 'int64', 'int32'] else col)
    best_setting = df.loc[df['accuracy_test_mean'].idxmax(),:]
    best_setting.to_json(f'../settings/best_tree_{dataset_name}.json')


def main(dataset_name,First=True):
    if First:
        experiment_on_decision_tree(dataset_name,
                                    f'../data/{dataset_name}.data')
    else:
        best_decision_tree_setting(dataset_name,
                                   f'../tables/{dataset_name}/decision_tree_{dataset_name}_accuracy.csv')



if __name__ == '__main__':
    pass

    main('abalone',False)
    main('adult',False)
    main('cmc',False)

