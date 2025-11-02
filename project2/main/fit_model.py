import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import make_moons

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC, Precision, Recall

from main.evaluate import get_one_score


def decision_tree(X_train, y_train, params):
    parameters = {'ccp_alpha', 'class_weight', 'criterion', 'max_depth', 'max_features', 'max_leaf_nodes',
 'min_impurity_decrease', 'min_samples_leaf', 'min_samples_split',
 'min_weight_fraction_leaf', 'monotonic_cst', 'random_state', 'splitter'}

    common_keys = set(params.keys()) & parameters
    params = {key: params[key] for key in common_keys}

    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    return model


def decision_tree_hyper_search(X_train, y_train, n_iter=400):
    param_dist = {
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy']
    }

    clf = DecisionTreeClassifier()
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=8,
        n_jobs=1
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    model_params = random_search.cv_results_['params']
    model_scores = random_search.cv_results_['mean_test_score']

    return best_model, model_params, model_scores


def decision_tree_post_pruning(X_train, y_train, X_test, y_test, model, dataset_name, save_figure=False):

    path = model.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    train_f1s = []
    test_f1s = []
    for clf in clfs:
        f1_train, f1_test = get_one_score(clf, X_train, X_test, y_train, y_test, True)
        train_f1s.append(f1_train)
        test_f1s.append(f1_test)
    (train_score,
     test_score,
     train_f1,
     test_f1,
     ccp_alpha, best_clf) = max(zip(train_scores,
                                    test_scores,
                                    train_f1s,
                                    test_f1s,
                                    ccp_alphas,
                                    clfs), key=lambda x: x[1])

    figs = []
    if save_figure:
        fig1, ax1 = plt.subplots()
        ax1.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
        ax1.set_xlabel("effective alpha")
        ax1.set_ylabel("total impurity of leaves")
        ax1.set_title(f"{dataset_name} Total Impurity vs effective alpha for training set")
        ax1.axvline(x=ccp_alpha, color='grey', linestyle='--', label=f'alpha_best={ccp_alpha:.2e}')
        ax1.legend()
        figs.append(fig1)

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]
        fig2, ax2 = plt.subplots(2, 1)
        ax2[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
        ax2[0].set_xlabel("alpha")
        ax2[0].set_ylabel("number of nodes")
        ax2[0].set_title(f"{dataset_name} Number of nodes vs alpha")
        ax2[0].axvline(x=ccp_alpha, color='grey', linestyle='--', label=f'alpha_best={ccp_alpha:.2e}')
        ax2[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
        ax2[1].set_xlabel("alpha")
        ax2[1].set_ylabel("depth of tree")
        ax2[1].set_title(f"{dataset_name} Depth vs alpha")
        ax2[1].axvline(x=ccp_alpha, color='grey', linestyle='--', label=f'alpha_best={ccp_alpha:.2e}')
        ax2[0].legend()
        ax2[1].legend()
        fig2.tight_layout()
        figs.append(fig2)

        fig3, ax3 = plt.subplots()
        ax3.set_xlabel("alpha")
        ax3.set_ylabel("accuracy")
        ax3.set_title(f"{dataset_name} Accuracy vs alpha for training and testing sets")
        ax3.plot(ccp_alphas, train_scores, marker='o', label="train",
                drawstyle="steps-post")
        ax3.plot(ccp_alphas, test_scores, marker='o', label="test",
                drawstyle="steps-post")
        ax3.axvline(x=ccp_alpha, color='grey', linestyle='--', label=f'alpha_best={ccp_alpha:.2e}')
        ax3.axhline(y=test_score, color='grey', linestyle='--', label=f'best accuracy={test_score:.5f}')
        ax3.legend()
        figs.append(fig3)

    return best_clf, train_score, test_score, train_f1, test_f1, ccp_alpha, figs


def random_forest(X_train, y_train, n_estimators=None):
    rnd_clf = RandomForestClassifier(n_estimators=n_estimators)
    rnd_clf.fit(X_train, y_train)
    return rnd_clf

def gradient_boost(X_train, y_train, learning_rate=0.1, n_estimators=None):
    gb_clf = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
    gb_clf.fit(X_train, y_train)
    return gb_clf

def xg_boost(X_train, y_train, reg_lambda=1.0, n_estimators=None):
    xgb_clf = XGBClassifier(n_estimators=n_estimators, reg_lambda=reg_lambda, learning_rate=0.3)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

def nural_network(X_train,y_train,adam=False,l2_parameter:float=0,dropout_parameter:float=0):
    D_i = X_train.shape[1]
    D_o = y_train.shape[1]
    D_h = round(np.sqrt(D_i * D_o))

    model = Sequential()
    model.add(Input(shape=(D_i,)))
    if l2_parameter:
        model.add(Dense(D_h, activation='relu', kernel_regularizer=l2(l2_parameter)))
        model.add(Dense(D_h, activation='relu', kernel_regularizer=l2(l2_parameter)))
    else:
        model.add(Dense(D_h, activation='relu'))
        model.add(Dense(D_h, activation='relu'))
    if dropout_parameter:
        model.add(Dropout(dropout_parameter))
    model.add(Dense(D_o, activation='softmax'))

    if adam:
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        min_delta=0.001,
        mode='min',
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=90,
        batch_size=32,
        callbacks=[early_stopping]
    )

    return model
