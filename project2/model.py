import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import json
import sys
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier,GradientBoostingRegressor
from sklearn.datasets import make_moons

from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from main.data_process import data_init, plot_dataset, get_abalone_raw_data, get_cmc_raw_data, get_adult_raw_data
from main.fit_model import (decision_tree,
                            decision_tree_post_pruning,
                            random_forest,
                            gradient_boost,
                            xg_boost,
                            nural_network)
from main.evaluate import get_mean_score, compare_mean_score
from main.output import list_to_csv


Demo_Mode = False
Loop_Time = 10


def load_settings(dataset_name):
    with open(f'./settings/decision_tree_{dataset_name}.json', 'r') as file:
        decision_tree_setting = json.load(file)
    with open(f'./settings/best_tree_{dataset_name}.json', 'r') as file:
        best_tree_setting = json.load(file)
    return decision_tree_setting, best_tree_setting, None

def choose_decision_tree(data_train, parameters):
    models = []
    for i,d_train in enumerate(data_train):
        X_train, y_train = d_train.values()
        model_decision_tree = decision_tree(X_train, y_train, parameters)
        models.append(model_decision_tree)
    return models

def choose_post_pruning(data_train,data_test,dataset_name,parameters):
    models = []
    accuracy_train_list = []
    accuracy_test_list = []
    f1_train_list = []
    f1_test_list = []

    _figs = []
    for i,d_train in enumerate(data_train):
        X_train, y_train = d_train.values()
        model_decision_tree = decision_tree(X_train, y_train, parameters)
        X_train, y_train = data_train[i].values()
        X_test, y_test = data_test[i].values()
        (model,
         accuracy_train,
         accuracy_test,
         f1_train,
         f1_test,
         ccp_alpha,
         figs) = decision_tree_post_pruning(X_train, y_train, X_test, y_test, model_decision_tree, dataset_name,
                                                                                save_figure=not bool(i))
        if figs:
            _figs = figs
        models.append(model)
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        f1_train_list.append(f1_train)
        f1_test_list.append(f1_test)
    accuracy_train_mean, accuracy_test_mean = np.mean(accuracy_train_list), np.mean(accuracy_test_list)
    f1_train_mean, f1_test_mean = np.mean(f1_train_list), np.mean(f1_test_list)
    return models, accuracy_train_mean, accuracy_test_mean, f1_train_mean, f1_test_mean, _figs

def choose_ensemble(data_train, n_estimators, parameter=0.1, ensemble='random_forest'):
    models = []
    for i,d_train in enumerate(data_train):
        X_train, y_train = d_train.values()
        y_train = y_train.squeeze()
        if ensemble == 'random_forest':
            model_ensemble = random_forest(X_train, y_train, n_estimators)
        elif ensemble == 'gradient_boost':
            model_ensemble = gradient_boost(X_train, y_train, parameter, n_estimators)
        elif ensemble == 'xg_boost':
            model_ensemble = xg_boost(X_train, y_train, parameter, n_estimators)
        else:
            raise ValueError(f'invalid ensemble Method: {ensemble}')
        models.append(model_ensemble)
    return models

def multi_ensemble(data_train,
                   data_test,
                   dataset_name,
                   i_list,
                   _smote,
                   parameter:float=0,
                   ensemble='random_forest'):
    models_list = []
    for i in i_list:
        print(f'training_{ensemble}:n_estimators={i}')
        models = choose_ensemble(data_train, i, parameter, ensemble)
        models_list.append(models)


    accuracy_train_means, accuracy_test_means = compare_mean_score(models_list, data_train, data_test)
    f1_train_means, f1_test_means = compare_mean_score(models_list, data_train, data_test,True)
    list_to_csv([i_list, accuracy_train_means, accuracy_test_means, f1_train_means, f1_test_means],
                ['num_of_estimators(trees)', 'accuracy_train_mean', 'accuracy_test_mean', 'f1_train_mean', 'f1_test_mean'],
                f'./tables/{dataset_name}{'_smote' if _smote else ''}/{ensemble}_accuracy_{parameter}.csv')

    index = accuracy_test_means.index(max(accuracy_test_means))
    plt.plot(i_list, accuracy_train_means)
    plt.scatter(i_list, accuracy_train_means, label='accuracy_train')
    plt.plot(i_list, accuracy_test_means)
    plt.scatter(i_list, accuracy_test_means, label='accuracy_test')
    if i_list[index]% 1 != 0:
        text1 = f'{i_list[index]:.5f}'
    else:
        text1 = f'{i_list[index]}'
    if accuracy_test_means[index] %1 != 0:
        text2 = f'{accuracy_test_means[index]:.5f}'
    else:
        text2 = f'{accuracy_test_means[index]}'
    plt.text(i_list[index],
             accuracy_test_means[index],
             f'({text1}, {text2})',
             ha='center',
             va='bottom')
    plt.grid(color='gray', linewidth=0.5)
    if parameter:
        plt.scatter(i_list[0],accuracy_train_means[0],alpha=0,
                    label=f"{'λ' if ensemble=='xg_boost' else 'learning_rate'}={parameter}")
    plt.xlabel('num_of_estimators(trees)')
    plt.ylabel('accuracy')
    plt.title(f'{dataset_name} {ensemble} accuracy vs num_of_estimators')
    plt.legend()
    if not Demo_Mode:
        plt.savefig(f'./figures/{dataset_name}{'_smote' if _smote else ''}/{ensemble}_accuracy_{parameter}.png')
    else:
        plt.show()
    plt.clf()

    index1 = f1_test_means.index(max(f1_test_means))
    plt.plot(i_list, f1_train_means)
    plt.scatter(i_list, f1_train_means, label='f1_train')
    plt.plot(i_list, f1_test_means)
    plt.scatter(i_list, f1_test_means, label='f1_test')
    if i_list[index1] % 1 != 0:
        text1 = f'{i_list[index1]:.5f}'
    else:
        text1 = f'{i_list[index1]}'
    if f1_test_means[index1] % 1 != 0:
        text2 = f'{f1_test_means[index1]:.5f}'
    else:
        text2 = f'{f1_test_means[index1]}'
    plt.text(i_list[index1],
             f1_test_means[index1],
             f'({text1}, {text2})',
             ha='center',
             va='bottom')
    plt.grid(color='gray', linewidth=0.5)
    if parameter:
        plt.scatter(i_list[0], f1_train_means[0], alpha=0,
                    label=f"{'λ' if ensemble == 'xg_boost' else 'learning_rate'}={parameter}")
    plt.xlabel('num_of_estimators(trees)')
    plt.ylabel('f1-score')
    plt.title(f'{dataset_name} {ensemble} f1-score vs num_of_estimators')
    plt.legend()
    if not Demo_Mode:
        plt.savefig(f'./figures/{dataset_name}{'_smote' if _smote else ''}/{ensemble}_f1_{parameter}.png')
    else:
        plt.show()
    plt.clf()

    return (models_list[index],
            f'{ensemble}_{i_list[index]}_trees_{parameter}',
            accuracy_train_means[index],
            accuracy_test_means[index],
            f1_train_means[index],
            f1_test_means[index])

def choose_nural_network(data_train,adam=False,l2_parameter:float=0,dropout_parameter:float=0):
    models = []
    for i, d_train in enumerate(data_train):
        X_train, y_train = d_train.values()
        model_ensemble = nural_network(X_train, y_train, adam=False,l2_parameter=0,dropout_parameter=0)
        models.append(model_ensemble)
    return models

def multi_nural_network(data_train, data_test, parameters, dataset_name, _smote):
    method = parameters['method']
    if method == 'Adam':
        adam = True
    elif method == 'SGD':
        adam = False
    else:
        raise ValueError(f'invalid neural_net method: {method}')
    models_list = []
    method_list = []
    setting_list_l2 = []
    setting_list_dropout = []
    for l2_parameter in parameters['l2_parameter']:
        for dropout_parameter in parameters['dropout_parameter']:
            method_list.append(method)
            setting_list_l2.append(l2_parameter)
            setting_list_dropout.append(dropout_parameter)
            models = choose_nural_network(data_train,adam,l2_parameter,dropout_parameter)
            print(f'training_neural_network: {method}, l2={l2_parameter}, dropout={dropout_parameter}')
            models_list.append(models)
    accuracy_train_means, accuracy_test_means = compare_mean_score(models_list, data_train, data_test)
    f1_train_means, f1_test_means = compare_mean_score(models_list, data_train, data_test, F1=True)
    list_to_csv([setting_list_l2,
                 setting_list_dropout,
                 method_list,
                 accuracy_train_means,
                 accuracy_test_means,
                 f1_train_means,
                 f1_test_means],
                ['l2',
                 'dropout',
                 'method',
                 'accuracy_train_mean',
                 'accuracy_test_mean',
                 'f1_train_mean',
                 'f1_test_mean'],
                f"./tables/{dataset_name}{'_smote' if _smote else ''}/{method}_nn_{dataset_name}_accuracy_{parameters['i']}.csv")

    i_list = None
    len_l2 = len(parameters['l2_parameter'])
    len_dropout = len(parameters['dropout_parameter'])
    if len_l2==1 and len_dropout!=1:
        i_list = parameters['dropout_parameter']
    elif len_l2!=1 and len_dropout==1:
        i_list = parameters['l2_parameter']

    index = accuracy_test_means.index(max(accuracy_test_means))
    if i_list is not None:
        plt.plot(i_list, accuracy_train_means)
        plt.scatter(i_list, accuracy_train_means, label='accuracy_train')
        plt.plot(i_list, accuracy_test_means)
        plt.scatter(i_list, accuracy_test_means, label='accuracy_test')
        if i_list[index]% 1 != 0:
            text1 = f'{i_list[index]:.5f}'
        else:
            text1 = f'{i_list[index]}'
        if accuracy_test_means[index] %1 != 0:
            text2 = f'{accuracy_test_means[index]:.5f}'
        else:
            text2 = f'{accuracy_test_means[index]}'
        plt.text(i_list[index],
                 accuracy_test_means[index],
                 f'({text1}, {text2})',
                 ha='center',
                 va='bottom')
        plt.grid(color='gray', linewidth=0.5)
        plt.xlabel('l2 λ' if len_dropout==1 else 'dropout rate')
        plt.ylabel('accuracy')
        plt.title(f"{dataset_name} {method} neural network accuracy vs {'l2 λ' if len_dropout==1 else 'dropout rate'}")
        plt.legend()
        if not Demo_Mode:
            plt.savefig(f"./figures/{dataset_name}{'_smote' if _smote else ''}/{method}_nn_{dataset_name}_accuracy_{parameters['i']}.png")
        else:
            plt.show()
        plt.clf()


    index1 = f1_test_means.index(max(f1_test_means))
    if i_list is not None:
        plt.plot(i_list, f1_train_means)
        plt.scatter(i_list, f1_train_means, label='f1_train')
        plt.plot(i_list, f1_test_means)
        plt.scatter(i_list, f1_test_means, label='f1_test')
        if i_list[index1]% 1 != 0:
            text1 = f'{i_list[index1]:.5f}'
        else:
            text1 = f'{i_list[index1]}'
        if f1_test_means[index1] %1 != 0:
            text2 = f'{f1_test_means[index1]:.5f}'
        else:
            text2 = f'{f1_test_means[index1]}'
        plt.text(i_list[index1],
                 f1_test_means[index1],
                 f'({text1}, {text2})',
                 ha='center',
                 va='bottom')
        plt.grid(color='gray', linewidth=0.5)
        plt.xlabel('l2 λ' if len_dropout==1 else 'dropout rate')
        plt.ylabel('f1-score')
        plt.title(f"{dataset_name} {method} neural network f1 vs {'l2 λ' if len_dropout==1 else 'dropout rate'}")
        plt.legend()
        if not Demo_Mode:
            plt.savefig(f"./figures/{dataset_name}{'_smote' if _smote else ''}/{method}_nn_{dataset_name}_f1_{parameters['i']}.png")
        else:
            plt.show()
        plt.clf()


    return (models_list[index],
            f'nn_{method_list[index]}_l2_{setting_list_l2[index]}_dropout_{setting_list_dropout[index]}',
            accuracy_train_means[index],
            accuracy_test_means[index],
            f1_train_means[index],
            f1_test_means[index])


def main(dataset_name='abalone', nn=False, _smote=False, without_post=False):

    # init
    decision_tree_setting, best_tree_setting, _ = load_settings(dataset_name)
    tree_data_train, tree_data_test, nn_data_train, nn_data_test = data_init(dataset_name,
                                                                             f'./data/{dataset_name}.data',
                                                                             Loop_Time,_smote)
    models_lists = []
    names_lists = []
    accuracy_train_mean_lists = []
    accuracy_test_mean_lists = []
    f1_train_mean_lists = []
    f1_test_mean_lists = []
    models_list = []


    # decision tree
    if not nn:
        for i, parameters in enumerate(decision_tree_setting):
            models = choose_decision_tree(tree_data_train, parameters)
            models_list.append(models)
        accuracy_train_means, accuracy_test_means = compare_mean_score(models_list, tree_data_train, tree_data_test)
        f1_train_means, f1_test_means = compare_mean_score(models_list, tree_data_train, tree_data_test, True)
        list_to_csv([decision_tree_setting,
                     accuracy_train_means,
                     accuracy_test_means,
                     f1_train_means,
                     f1_test_means],
                    ['decision_tree_setting',
                     'accuracy_train_mean',
                     'accuracy_test_mean',
                     'F1_train_mean',
                     'F1_test_mean'],
                    f'./tables/{dataset_name}{'_smote' if _smote else ''}/decision_tree_{dataset_name}_accuracy.csv')

        # decision tree pre-pruning
        print('training_decision_tree')
        models = choose_decision_tree(tree_data_train,best_tree_setting)

        # plt.figure(figsize=(300, 200))
        # X_train_0, y_train_0 = tree_data_train[0].values()
        # feature_names = X_train_0.columns if hasattr(X_train_0, 'columns') else None
        # class_names = np.unique(y_train_0).astype(str)
        # plot_tree(models[0], filled=True, feature_names=feature_names, class_names=class_names)
        # plt.savefig(f'./figures/{dataset_name}_tree_plot.png')

        X_train_0, y_train_0 = tree_data_train[0].values()
        feature_names = X_train_0.columns if hasattr(X_train_0, 'columns') else None
        class_names = np.unique(y_train_0).astype(str)
        plot_tree(models[0],filled=True, 
         feature_names=feature_names, 
         class_names=class_names,
         max_depth=3)
        if Demo_Mode:
            plt.show()
        else:
            plt.savefig(f'./figures/{dataset_name}_tree_plot.png')

        models_lists.append(models)
        names_lists.append('decision_tree_pre_pruning')
        accuracy_train_mean, accuracy_test_mean = get_mean_score(models,
                                                                    tree_data_train,
                                                                    tree_data_test)
        f1_train_mean, f1_test_mean = get_mean_score(models,
                                                                    tree_data_train,
                                                                    tree_data_test,True)
        accuracy_train_mean_lists.append(accuracy_train_mean)
        accuracy_test_mean_lists.append(accuracy_test_mean)
        f1_train_mean_lists.append(f1_train_mean)
        f1_test_mean_lists.append(f1_test_mean)

        # decision tree post_pruning
        if without_post:
            print('training_decision_tree: post-pruning')
            (models,
             accuracy_train_mean,
             accuracy_test_mean,
             f1_train_mean,
             f1_test_mean,
             figs) = choose_post_pruning(tree_data_train,tree_data_test,dataset_name,best_tree_setting)
            # models_lists.append(models)
            names_lists.append('decision_tree_post_pruning')
            accuracy_train_mean_lists.append(accuracy_train_mean)
            accuracy_test_mean_lists.append(accuracy_test_mean)
            f1_train_mean_lists.append(f1_train_mean)
            f1_test_mean_lists.append(f1_test_mean)
            for j, fig in enumerate(figs):
                if Demo_Mode:
                    fig.show()
                else:
                    fig.savefig(f'./figures/{dataset_name}{'_smote' if _smote else ''}/decision_tree_post_pruning_{j}.png')
                fig.clf()

        # Ensemble Learning
        ensemble_setting = {
            'random_forest' : {
                'parameters': [0],
                'i_list' : [2, 3, 5, 7, 11, 13, 20, 30, 50, 70, 90, 120, 150, 200]
            },
            'gradient_boost' : {
                'parameters': [0.02, 0.05, 0.1, 0.3],
                'i_list' : [2, 3, 5, 7, 11, 13, 20, 30, 50, 70, 90, 120, 150, 200]
            },
            'xg_boost' : {
                'parameters':[0.02, 0.1, 1, 3, 5, 9, 15],
                'i_list' : [2, 3, 5, 7, 11, 13, 15, 17, 20, 25, 30, 40, 50]
            },
        }
        for key, value in ensemble_setting.items():
            for parameter in value['parameters']:
                (models,
                 name,
                 accuracy_train_mean,
                 accuracy_test_mean,
                 f1_train_mean,
                 f1_test_mean) = multi_ensemble(tree_data_train,
                                                tree_data_test,
                                                dataset_name,
                                                value['i_list'],
                                                _smote,
                                                parameter=parameter,
                                                ensemble=key)
        #         models_lists.append(models)
                names_lists.append(name)
                accuracy_train_mean_lists.append(accuracy_train_mean)
                accuracy_test_mean_lists.append(accuracy_test_mean)
                f1_train_mean_lists.append(f1_train_mean)
                f1_test_mean_lists.append(f1_test_mean)

    # neural network
    else:
        nn_setting = [
            {
                'i':0,
                'l2_parameter': [0],
                'dropout_parameter': [0]
            },
            {
                'i':1,
                'l2_parameter': [0],
                'dropout_parameter': np.arange(0,0.95,0.2)
            },
            {
                'i':2,
                'l2_parameter': np.insert(np.logspace(-5, -3, num=5), 0, 0),
                'dropout_parameter': [0]
            },
            {
                'i':3,
                'l2_parameter': np.logspace(-5, -3, num=3),
                'dropout_parameter': np.arange(0.1,1,0.3)
            },
        ]
        for method in ['Adam','SGD']:
            for i,each in enumerate(nn_setting):
                each['method'] = method
                (models,
                 name,
                 accuracy_train_mean,
                 accuracy_test_mean,
                 f1_train_mean,
                 f1_test_mean) = multi_nural_network(nn_data_train,
                                                     nn_data_test,
                                                     each,
                                                     dataset_name,
                                                     _smote)
                # models_lists.append(models)
                names_lists.append(name)
                accuracy_train_mean_lists.append(accuracy_train_mean)
                accuracy_test_mean_lists.append(accuracy_test_mean)
                f1_train_mean_lists.append(f1_train_mean)
                f1_test_mean_lists.append(f1_test_mean)

    list_to_csv([names_lists, accuracy_train_mean_lists, accuracy_test_mean_lists, f1_train_mean_lists, f1_test_mean_lists],
                ['model_name', 'accuracy_train_mean', 'accuracy_test_mean', 'f1_train_mean', 'f1_test_mean'],
                f'./tables/{dataset_name}{'_smote' if _smote else ''}/best_models_{'nn' if nn else 'tree'}.csv')


    pass

def plot_datasets():
    X1, y1, dataset1 = get_abalone_raw_data('./data/abalone.data')
    X2, y2, dataset2 = get_cmc_raw_data('./data/cmc.data')
    X3, y3, dataset3 = get_adult_raw_data('./data/adult.data')

    plot_dataset(dataset1, 'abalone')
    plot_dataset(dataset2, 'cmc')
    plot_dataset(dataset3, 'adult')

if __name__ == '__main__':
    Demo_Mode = True
    if Demo_Mode:
        Loop_Time = 1

    # Warning:
    # Run just one line of following code at one time. 
    # Ortherwise, the terminal will Kill this process.

    plot_datasets()
    main(dataset_name='abalone', nn=False, _smote=False)
    # main(dataset_name='abalone', nn=True, _smote=False)
    # main(dataset_name='cmc', nn=False, _smote=False)
    # main(dataset_name='cmc', nn=True, _smote=False)
    # main(dataset_name='adult', nn=False, _smote=False, without_post=True)
    # main(dataset_name='adult', nn=True, _smote=False)

