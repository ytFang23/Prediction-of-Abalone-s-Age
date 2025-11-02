import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, mean_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, roc_auc_score
import itertools
import os

Show_Mode = False
Quick_Mode = False
Demo_Mode = False
Loop_Time = 2


def get_data(file_path):
    data_set = pd.read_csv('./data/abalone_data',
                           header=None,
                           names=['Sex',
                                  'Length',
                                  'Diameter',
                                  'Height',
                                  'Whole_Weight',
                                  'Shucked_Weight',
                                  'Viscera_Weight',
                                  'Shell_Weight',
                                  'Ring_Age'])

    data_set['Male'] = np.where(data_set['Sex'] == 'M', 1, 0)
    data_set['Female'] = np.where(data_set['Sex'] == 'F', 1, 0)
    data_set['Infant'] = np.where(data_set['Sex'] == 'I', 1, 0)
    data_set = data_set.drop(columns=['Sex'])

    matrix_x = data_set.drop(columns=['Ring_Age'])
    y = data_set['Ring_Age']

    return matrix_x, y, data_set


def data_statistic(data_set):
    if not os.path.exists('./figure/data_statistic'):
        os.makedirs('./figure/data_statistic')
    plt.figure(figsize=(12, 10))

    # Correlation Matrix Heatmap
    correlation_matrix = data_set.corr()
    mask = correlation_matrix == 1
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', mask=mask, cmap='coolwarm', square=True, cbar_kws={"shrink": .8},
                xticklabels=data_set.columns, yticklabels=data_set.columns)
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('./figure/data_statistic/heatmap.png', dpi=300)
    if Show_Mode:
        plt.show()
    plt.clf()

    plt.figure()
    # Diameter Scatter Plot
    plt.scatter(data_set['Diameter'], data_set['Ring_Age'], color='blue', marker='o', s=1)
    plt.title('Diameter and Ring Age')
    plt.xlabel('Diameter')
    plt.ylabel('Ring Age')
    plt.tight_layout()
    plt.savefig('./figure/data_statistic/diameter_and_ring_age_scatter.png', dpi=300)
    if Show_Mode:
        plt.show()
    plt.clf()

    # Shell_Weight Scatter Plot
    plt.scatter(data_set['Shell_Weight'], data_set['Ring_Age'], color='blue', marker='o', s=1)
    plt.title('Shell Weight and Ring Age')
    plt.xlabel('Shell Weight')
    plt.ylabel('Ring Age')
    plt.tight_layout()
    plt.savefig('./figure/data_statistic/shell_weight_and_ring_age_scatter.png', dpi=300)
    if Show_Mode:
        plt.show()
    plt.clf()

    # Histogram
    sns.histplot(data_set['Diameter'], bins=30, kde=True, color='blue', alpha=0.6)
    plt.title('Diameter Histogram')
    plt.tight_layout()
    plt.savefig('./figure/data_statistic/diameter_histogram.png', dpi=300)
    if Show_Mode:
        plt.show()
    plt.clf()

    sns.histplot(data_set['Shell_Weight'], bins=30, kde=True, color='blue', alpha=0.6)
    plt.title('Shell Weight Histogram')
    plt.tight_layout()
    plt.savefig('./figure/data_statistic/shell_weight_histogram.png', dpi=300)
    if Show_Mode:
        plt.show()
    plt.clf()

    sns.histplot(data_set['Ring_Age'], bins=np.arange(-0.5, 29.5, 1), kde=True, color='blue', alpha=0.6)
    plt.title('Ring Age Histogram')
    plt.tight_layout()
    plt.savefig('./figure/data_statistic/ring_age_histogram.png', dpi=300)
    if Show_Mode:
        plt.show()
    plt.clf()


def process_data(matrix_x_train, matrix_x_test, method='MinMaxScaler'):
    if method == 'MinMaxScaler':
        processor = MinMaxScaler()
    elif method == 'StandardScaler':
        processor = StandardScaler()
    elif method == 'Normalizer':
        processor = Normalizer()
    else:
        return matrix_x_train, matrix_x_test

    processor.fit(matrix_x_train)
    matrix_x_train = processor.transform(matrix_x_train)
    matrix_x_test = processor.transform(matrix_x_test)

    return matrix_x_train, matrix_x_test


def plot_actual_predict(y_test, y_predict, file_path, title, show_figure=False):
    df = pd.DataFrame({
        'y1': y_test,
        'y2': y_predict,
    })
    x_plot = np.arange(len(y_test))
    df = df.sort_values(by=['y1', 'y2'])
    df['x'] = x_plot

    plt.scatter(df['x'], df['y1'], label='Actual Age', s=1)
    plt.scatter(df['x'], df['y2'], label='Predict Age', s=1)
    plt.title(title)
    plt.ylim(0,35)
    plt.xlabel('Samples')
    plt.xticks([])
    plt.ylabel('Ring Age')
    plt.legend(loc='upper left')

    directory, filename = os.path.split(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.tight_layout()
    plt.savefig(file_path)
    print(f'({file_path} SAVED)')
    if show_figure and Show_Mode:
        plt.show()
    plt.clf()


def plot_residuals(y_test, y_predict, file_path, title, show_figure=False):
    # !!!!! maybe弃用，一开始写复杂了，而且图的表意不如plot_linear_regression清晰，不过可能plot_residuals的图更好看
    residuals = y_test - y_predict
    positive_residuals = np.where(residuals >= 0, residuals, 0)
    negative_residuals = np.where(residuals < 0, -residuals, 0)
    y_plot = y_test - negative_residuals
    x_plot = np.arange(len(y_plot))
    df = pd.DataFrame({
        'y': y_plot,
        'positive_residuals': positive_residuals,
        'negative_residuals': negative_residuals,
        'actual': y_test,
        'residuals': residuals
    })
    df = df.sort_values(by=['actual', 'residuals'])
    df['x'] = x_plot
    plt.bar(df['x'], df['y'], color='#307aff', width=1)
    plt.bar(df['x'], df['negative_residuals'], bottom=df['y'], color='orange', label='negative_residuals', width=1)
    plt.bar(df['x'], df['positive_residuals'], bottom=df['y'] + df['negative_residuals'], color='blue',
            label='positive_residuals', width=1)
    plt.title(title)
    plt.xlabel('Samples')
    plt.xticks([])
    plt.ylabel('Ring Age')
    plt.legend(loc='upper left')

    directory, filename = os.path.split(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.tight_layout()
    plt.savefig(file_path, dpi=600)
    print(f'({file_path} SAVED)')
    if show_figure and Show_Mode:
        plt.show()
    plt.clf()


def plot_confusion_matrix(y_test, y_predict, file_path, title, show_figure=False):
    cmat = confusion_matrix(y_test, y_predict, labels=[0, 1])
    cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
    classes = ['under7', 'over7']

    plt.imshow(cmat, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cmat.max() / 2.
    for i, j in itertools.product(range(cmat.shape[0]), range(cmat.shape[1])):
        plt.text(j, i, format(cmat[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cmat[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    directory, filename = os.path.split(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.tight_layout()
    plt.savefig(file_path)
    print(f'({file_path} SAVED)')
    if show_figure and Show_Mode:
        plt.show()
    plt.clf()


def plot_roc(y_test, y_predict_prob, file_path, title, show_figure=False):
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
    auc_value = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='red', label='ROC (AUC = {:.2f})'.format(auc_value))
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(which='both')

    directory, filename = os.path.split(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.tight_layout()
    plt.savefig(file_path)
    print(f'({file_path} SAVED)')
    if show_figure and Show_Mode:
        plt.show()
    plt.clf()


def plot_pr(y_test, y_predict_prob, file_path, title, show_figure=False):
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
    auc_pr = auc(recall, precision)

    plt.plot(recall, precision, color='blue', label=f'PR curve (area = {auc_pr:.2f})')
    # plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')
    plt.ylim(0.88,1.00)
    plt.xlim(0,1.05)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc='lower left')
    plt.grid(which='both')

    directory, filename = os.path.split(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.tight_layout()
    plt.savefig(file_path)
    print(f'({file_path} SAVED)')
    if show_figure and Show_Mode:
        plt.show()
    plt.clf()


def regression_correct_metrics(y_test, y_predict):
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    r_squared = r2_score(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)

    return rmse, r_squared, mae


def classification_correct_metrics(y_test, y_predict, y_predict_prob):
    accuracy = accuracy_score(y_test, y_predict)

    precision = precision_score(y_test, y_predict)

    recall = recall_score(y_test, y_predict)

    f1 = f1_score(y_test, y_predict)

    log_loss_value = log_loss(y_test, y_predict_prob)

    auc_value = roc_auc_score(y_test, y_predict_prob)

    return accuracy, precision, recall, f1, log_loss_value, auc_value


def linear_regression(matrix_x_train, matrix_x_test, y_train, y_test, i, filepath):
    model = LinearRegression()
    model.fit(matrix_x_train, y_train)
    y_predict = model.predict(matrix_x_test)
    y_predict_train = model.predict(matrix_x_train)

    # plot
    if (Quick_Mode and i == 0) or not Quick_Mode:
        plot_actual_predict(y_test,
                            y_predict,
                            f"./figure/{filepath}/actual_predict/{i}.png" if not Demo_Mode
                            else f"./f/actual_predict/{filepath}_test.png",
                            f'Predicted Age and Actual Age (test) - {filepath} {i}',
                            i == 0)
        plot_actual_predict(y_train,
                            y_predict_train,
                            f"./figure/{filepath}/actual_predict_train/{i}.png" if not Demo_Mode
                            else f"./f/actual_predict/{filepath}_train.png",
                            f'Predicted Age and Actual Age (train)- {filepath} {i}',
                            i == 0)

        # Another kind of plot, take very long time but.
        # plot_residuals(y_test,
        #                y_predict,
        #                f"./figure/{filepath}/residual/{i}.png",
        #                f'Predicted Age with Residuals - {filepath} {i}',
        #                i == 0)

    # calculate correct_metrics
    correct_metrics = regression_correct_metrics(y_test, y_predict)
    correct_metrics_train = regression_correct_metrics(y_train, y_predict_train)

    return model, correct_metrics, correct_metrics_train


def logistic_regression(matrix_x_train, matrix_x_test, y_train, y_test, i, filepath):
    # fit model
    model = LogisticRegression()
    model.fit(matrix_x_train, y_train)

    # predict
    y_predict = model.predict(matrix_x_test)
    y_predict_prob = model.predict_proba(matrix_x_test)[:, 1]
    y_predict_train = model.predict(matrix_x_train)
    y_predict_prob_train = model.predict_proba(matrix_x_train)[:, 1]

    correct_metrics = classification_correct_metrics(y_test, y_predict, y_predict_prob)
    correct_metrics_train = classification_correct_metrics(y_train, y_predict_train, y_predict_prob_train)

    if (Quick_Mode and i == 0) or not Quick_Mode:
        # plot confusion matrix
        plot_confusion_matrix(y_test,
                              y_predict,
                              f"./figure/{filepath}/confusion_matrix/{i}.png" if not Demo_Mode
                              else f"./f/confusion_matrix_train/{filepath}_test.png",
                              f'Confusion Matrix (test)- {filepath} {i}',
                              i == 0)
        plot_confusion_matrix(y_train,
                              y_predict_train,
                              f"./figure/{filepath}/confusion_matrix_train/{i}.png" if not Demo_Mode
                              else f"./f/confusion_matrix_train/{filepath}_train.png",
                              f'Confusion Matrix (train)- {filepath} {i}',
                              i == 0)
        # plot ROC,PR
        plot_roc(y_test,
                 y_predict_prob,
                 f"./figure/{filepath}/ROC/{i}.png" if not Demo_Mode else f"./f/ROC/{filepath}_test.png",
                 f'Receiver Operating Characteristic (test)- {filepath} {i}',
                 i == 0)
        plot_roc(y_train,
                 y_predict_prob_train,
                 f"./figure/{filepath}/ROC_train/{i}.png" if not Demo_Mode else f"./f/ROC/{filepath}_train.png",
                 f'Receiver Operating Characteristic (train)- {filepath} {i}',
                 i == 0)
        plot_pr(y_test,
                y_predict_prob,
                f"./figure/{filepath}/PR/{i}.png" if not Demo_Mode else f"./f/PR/{filepath}_test.png",
                f'Precision-Recall Curve (test)- {filepath} {i}',
                i == 0)
        plot_pr(y_train,
                y_predict_prob_train,
                f"./figure/{filepath}/PR_train/{i}.png" if not Demo_Mode else f"./f/PR/{filepath}_train.png",
                f'Precision-Recall Curve (train)- {filepath} {i}',
                i == 0)

    return model, correct_metrics, correct_metrics_train


def save_table(df, name):

    print(df.mean())
    print(df.mean())
    
    if not os.path.exists('./table'):
        os.makedirs('./table')

    mean_row = df.mean().to_frame().T
    mean_row.index = ['Mean']
    df = pd.concat([df, mean_row], ignore_index=False)
    std_row = df.std().to_frame().T
    std_row.index = ['Std']
    df = pd.concat([df, std_row], ignore_index=False)

    df.to_csv(f'./table/{name}.csv', index=True)
    print(f'({name}.csv SAVED)')



import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, precision_recall_curve, auc


def network_regression(matrix_x_train,
                       matrix_x_test,
                       y_train,
                       y_test,
                       i,
                       filepath,
                       partly = False):
    if partly:
        layer_1 = Dense(units=2, activation='relu')
        layer_2 = Dense(units=4, activation='relu')
        layer_3 = Dense(units=1, activation=None)  # Linear activation for regression
        model = Sequential([layer_1, layer_2, layer_3])
        model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
                      loss=MeanSquaredError(),
                      metrics=['mae'])  # Mean Absolute Error for regression
    else:
        layer_1 = Dense(units=10, activation='relu')
        layer_2 = Dense(units=8, activation='relu')
        layer_3 = Dense(units=1, activation=None)  # Linear activation for regression
        model = Sequential([layer_1, layer_2, layer_3])
        model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True),
                      loss=MeanSquaredError(),
                      metrics=['mae'])  # Mean Absolute Error for regression

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # Train regression model
    history = model.fit(matrix_x_train, y_train, validation_data=(matrix_x_test, y_test), epochs=100, batch_size=32,
                        callbacks=[early_stopping])
    y_pred_test = model.predict(matrix_x_test).flatten()
    y_pred_train = model.predict(matrix_x_train).flatten()

    # plot
    if (Quick_Mode and i == 0) or not Quick_Mode:
        plot_actual_predict(y_test,
                            y_pred_test,
                            f"./figure/{filepath}/actual_predict/{i}.png" if not Demo_Mode
                              else f"./f/actual_predict/{filepath}_test.png",
                            f'Predicted Age and Actual Age (test) - {filepath} {i}',
                            i == 0)
        plot_actual_predict(y_train,
                            y_pred_train,
                            f"./figure/{filepath}/actual_predict_train/{i}.png" if not Demo_Mode
                              else f"./f/actual_predict/{filepath}_train.png",
                            f'Predicted Age and Actual Age (train)- {filepath} {i}',
                            i == 0)
        # plot_residuals(y_test,
        #                y_predict,
        #                f"./figure/{filepath}/residual/{i}.png",
        #                f'Predicted Age with Residuals - {filepath} {i}',
        #                i == 0)

    # calculate correct_metrics
    correct_metrics = regression_correct_metrics(y_test, y_pred_test)
    correct_metrics_train = regression_correct_metrics(y_train, y_pred_train)

    return model, correct_metrics, correct_metrics_train


def network_classification(matrix_x_train,
                           matrix_x_test,
                           y_train,
                           y_test,
                           i,
                           filepath,
                           partly = False):
    if partly:
        layer_1 = Dense(units=2, activation='sigmoid')
        layer_2 = Dense(units=4, activation='relu')
        layer_3 = Dense(units=1, activation='sigmoid')
        model = Sequential([layer_1, layer_2, layer_3])
        model.compile(optimizer=SGD(learning_rate=0.09, momentum=0.9, nesterov=True),
                      loss=BinaryCrossentropy(),
                      metrics=['accuracy'])
    else:
        layer_1 = Dense(units=10, activation='sigmoid')
        layer_2 = Dense(units=12, activation='relu')
        layer_3 = Dense(units=1, activation='sigmoid')
        model = Sequential([layer_1, layer_2, layer_3])
        model.compile(optimizer=SGD(learning_rate=0.09, momentum=0.9, nesterov=True),
                      loss=BinaryCrossentropy(),
                      metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(matrix_x_train, y_train, validation_data=(matrix_x_test, y_test), epochs=100, batch_size=32,
                        callbacks=[early_stopping])

    # predict
    y_predict_prob = model.predict(matrix_x_test).flatten()
    y_predict = (y_predict_prob >= 0.5).astype(int)
    y_predict_prob_train = model.predict(matrix_x_train).flatten()
    y_predict_train = (y_predict_prob_train >= 0.5).astype(int)

    correct_metrics = classification_correct_metrics(y_test, y_predict, y_predict_prob)
    correct_metrics_train = classification_correct_metrics(y_train, y_predict_train, y_predict_prob_train)

    if (Quick_Mode and i == 0) or not Quick_Mode:
        # plot confusion matrix
        plot_confusion_matrix(y_test,
                              y_predict,
                              f"./figure/{filepath}/confusion_matrix/{i}.png" if not Demo_Mode
                              else f"./f/confusion_matrix_train/{filepath}_test.png",
                              f'Confusion Matrix (test)- {filepath} {i}',
                              i == 0)
        plot_confusion_matrix(y_train,
                              y_predict_train,
                              f"./figure/{filepath}/confusion_matrix_train/{i}.png" if not Demo_Mode
                              else f"./f/confusion_matrix_train/{filepath}_train.png",
                              f'Confusion Matrix (train)- {filepath} {i}',
                              i == 0)
        # plot ROC,PR
        plot_roc(y_test,
                 y_predict_prob,
                 f"./figure/{filepath}/ROC/{i}.png" if not Demo_Mode else f"./f/ROC/{filepath}_test.png",
                 f'Receiver Operating Characteristic (test)- {filepath} {i}',
                 i == 0)
        plot_roc(y_train,
                 y_predict_prob_train,
                 f"./figure/{filepath}/ROC_train/{i}.png" if not Demo_Mode else f"./f/ROC/{filepath}_train.png",
                 f'Receiver Operating Characteristic (train)- {filepath} {i}',
                 i == 0)
        plot_pr(y_test,
                y_predict_prob,
                f"./figure/{filepath}/PR/{i}.png" if not Demo_Mode else f"./f/PR/{filepath}_test.png",
                f'Precision-Recall Curve (test)- {filepath} {i}',
                i == 0)
        plot_pr(y_train,
                y_predict_prob_train,
                f"./figure/{filepath}/PR_train/{i}.png" if not Demo_Mode else f"./f/PR/{filepath}_train.png",
                f'Precision-Recall Curve (train)- {filepath} {i}',
                i == 0)

    return model, correct_metrics, correct_metrics_train




def choose_mode(matrix_x,
                y,
                neural_network=False,
                logistic=False,
                normalising=None,
                part_of_features=False
                ):
    if neural_network:
        if logistic:
            filepath = 'network_classification_'
        else:
            filepath = 'network_regression_'
    else:
        if logistic:
            filepath = 'logistic_regression_'
        else:
            filepath = 'linear_regression_'
    if normalising:
        filepath += normalising
    else:
        filepath += 'Basic'
    if part_of_features:
        filepath += '_partly'

    correct_metrics_list = []
    correct_metrics_train_list = []

    for i in range(Loop_Time):
        matrix_x_train, matrix_x_test, y_train, y_test = train_test_split(matrix_x, y, test_size=0.4, random_state=i)
        if part_of_features:
            matrix_x_train = matrix_x_train[['Diameter', 'Shell_Weight']]
            matrix_x_test = matrix_x_test[['Diameter', 'Shell_Weight']]
        matrix_x_train, matrix_x_test = process_data(matrix_x_train, matrix_x_test, normalising)
        if logistic:
            y_train = np.where(y_train >= 7, 1, 0)
            y_test = np.where(y_test >= 7, 1, 0)
            if neural_network:
                _, correct_metrics, correct_metrics_train = network_classification(matrix_x_train,
                                                                                   matrix_x_test,
                                                                                   y_train,
                                                                                   y_test,
                                                                                   i,
                                                                                   filepath,
                                                                                   part_of_features)
            else:
                _, correct_metrics, correct_metrics_train = logistic_regression(matrix_x_train,
                                                                                matrix_x_test,
                                                                                y_train,
                                                                                y_test,
                                                                                i,
                                                                                filepath)
            correct_metrics_list.append(correct_metrics)
            correct_metrics_train_list.append(correct_metrics_train)
        else:
            if neural_network:
                _, correct_metrics, correct_metrics_train = network_regression(matrix_x_train,
                                                                               matrix_x_test,
                                                                               y_train,
                                                                               y_test,
                                                                               i,
                                                                               filepath,
                                                                               part_of_features)
            else:
                _, correct_metrics, correct_metrics_train = linear_regression(matrix_x_train,
                                                                              matrix_x_test,
                                                                              y_train,
                                                                              y_test,
                                                                              i,
                                                                              filepath)
            correct_metrics_list.append(correct_metrics)
            correct_metrics_train_list.append(correct_metrics_train)

    if logistic:
        df_test = pd.DataFrame(correct_metrics_list)
        df_train = pd.DataFrame(correct_metrics_train_list)
        df_test.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Log-Loss-Value', 'AUC']
        df_train.columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'Log-Loss-Value', 'AUC']
    else:
        df_test = pd.DataFrame(correct_metrics_list)
        df_train = pd.DataFrame(correct_metrics_train_list)
        df_test.columns = ['RMSE', 'R-squared', 'MAE']
        df_train.columns = ['RMSE', 'R-squared', 'MAE']
        # do not need: 'Accuracy', 'Precision', 'Recall', 'F1', 'Log-Loss-Value', 'AUC'

    if not Demo_Mode:
        print(f'--------------------------{filepath}--------------------------')
        print('!!! TEST !!!')
        save_table(df_test, filepath + '_test')
        print('!!! TRAIN !!!')
        save_table(df_train, filepath + '_train')
        print('------------------------------------------------------------------------------')



def main(quick_mode=True):
    # target: compare different models
    # 1. linear or logistic
    # 2. with or without normalising (with different normalising method)
    # 3. two selected input features or all input features

    matrix_x, y, data_set = get_data('./data/abalone_data')

    data_statistic(data_set)

    choose_mode(matrix_x, y, False, False, None, False)
    choose_mode(matrix_x, y, True, False, None, False)
    choose_mode(matrix_x, y, False, False, None, True)
    choose_mode(matrix_x, y, True, False, None, True)
    choose_mode(matrix_x, y, False, False, 'Normalizer', False)
    choose_mode(matrix_x, y, True, False, 'Normalizer', False)

    # choose_mode(matrix_x, y, False, False, 'MinMaxScaler', False)
    # choose_mode(matrix_x, y, False, False, 'Normalizer', True)
    # choose_mode(matrix_x, y, False, False, 'StandardScaler', False)
    # choose_mode(matrix_x, y, False, False, 'StandardScaler', True)
    # choose_mode(matrix_x, y, False, False, 'MinMaxScaler', False)
    # choose_mode(matrix_x, y, False, False, 'MinMaxScaler', True)

    choose_mode(matrix_x, y, False, True, None, False)
    choose_mode(matrix_x, y, True, True, None, False)
    choose_mode(matrix_x, y, False, True, None, True)
    choose_mode(matrix_x, y, True, True, None, True)
    choose_mode(matrix_x, y, False, True, 'MinMaxScaler', False)
    choose_mode(matrix_x, y, True, True, 'MinMaxScaler', False)

    # choose_mode(matrix_x, y, False, True, 'Normalizer', False)
    # choose_mode(matrix_x, y, False, True, 'Normalizer', False)
    # choose_mode(matrix_x, y, False, True, 'Normalizer', True)
    # choose_mode(matrix_x, y, False,True, 'StandardScaler', False)
    # choose_mode(matrix_x, y, False, True, 'StandardScaler', True)
    # choose_mode(matrix_x, y, False, True, 'MinMaxScaler', True)


    # choose_mode(matrix_x, y, True, True, 'Normalizer', False)
    # choose_mode(matrix_x, y, True, False, 'MinMaxScaler', False)


if __name__ == "__main__":
    Show_Mode = False
    Quick_Mode = True
    Demo_Mode = True
    Loop_Time = 2
    main()
