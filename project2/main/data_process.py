import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def get_data(dataset_name, file_path):
    if dataset_name == 'abalone':
        return get_abalone_data(file_path)
    elif dataset_name == 'cmc':
        return get_cmc_data(file_path)
    elif dataset_name == 'adult':
        return get_adult_data(file_path)
    else:
        raise ValueError(f"No dataset named {dataset_name}")


def get_abalone_data(file_path):
    data_set = pd.read_csv(file_path)

    data_set['Male'] = np.where(data_set['Sex'] == 'M', 1, 0)
    data_set['Female'] = np.where(data_set['Sex'] == 'F', 1, 0)
    data_set = data_set.drop(columns=['Sex'])

    data_set['Class'] = np.where(data_set['Ring_Age'] <=7, 0, 0)
    data_set['Class'] = np.where((8<= data_set['Ring_Age']) &  (data_set['Ring_Age']<=10), 1, data_set['Class'])
    data_set['Class'] = np.where((11<= data_set['Ring_Age']) & (data_set['Ring_Age']<=15), 2, data_set['Class'])
    data_set['Class'] = np.where(data_set['Ring_Age'] >15, 3, data_set['Class'])

    X = data_set.drop(columns=['Ring_Age','Class'])
    y = pd.DataFrame(data_set['Class'])
    return X, y, data_set

def get_cmc_data(file_path):
    data_set = pd.read_csv(file_path)
    cols = ['Wife_education',
            'Husband_edu',
            'Wife_religion',
            'Wife_work',
            'Husband_occupation',
            'Standard_living',
            'Media_exposure']
    data_set['method'] = np.where(data_set['method'] == 1, 0, data_set['method'])
    data_set['method'] = np.where(data_set['method'] == 2, 1, data_set['method'])
    data_set['method'] = np.where(data_set['method'] == 3, 2, data_set['method'])
    data_set[cols] = data_set[cols].astype(str)
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    y = pd.DataFrame(y)
    return X, y, data_set

def get_adult_data(file_path):
    data_set = pd.read_csv(file_path)
    data_set = data_set.drop(columns=['Education','Fnlwgt'])
    data_set['Target'] = np.where(data_set['Target']=='>50K', 1,0)
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    y = pd.DataFrame(y)
    return X, y, data_set

def get_abalone_raw_data(file_path):
    data_set = pd.read_csv(file_path)
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    y = pd.DataFrame(y)
    return X, y, data_set

def get_cmc_raw_data(file_path):
    data_set = pd.read_csv(file_path)
    cols = ['Wife_education',
            'Husband_edu',
            'Wife_religion',
            'Wife_work',
            'Husband_occupation',
            'Standard_living',
            'Media_exposure',
            'method']
    data_set['method'] = np.where(data_set['method'] == 1, 0, data_set['method'])
    data_set['method'] = np.where(data_set['method'] == 2, 1, data_set['method'])
    data_set['method'] = np.where(data_set['method'] == 3, 2, data_set['method'])
    data_set[cols] = data_set[cols].astype(str)
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    y = pd.DataFrame(y)
    return X, y, data_set

def get_adult_raw_data(file_path):
    data_set = pd.read_csv(file_path)
    data_set.drop(columns=['Workclass','Education','Marital-status','Occupation','Relationship','Race','Native-country'], inplace=True)
    columns_to_convert = ['Age','Fnlwgt','Education-num','Capital-gain','Capital-loss','Hours-per-week']
    data_set[columns_to_convert] = data_set[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    X = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1]
    y = pd.DataFrame(y)
    return X, y, data_set

def missing_data(X_train, X_test):
    X_train_filled = X_train.copy()
    X_test_filled = X_test.copy()

    if not X_train_filled.isnull().any().any() and not X_test_filled.isnull().any().any():
        print('Without Null')
        return X_train_filled, X_test_filled
    else:
        print('With Null')

    num_cols = X_train.select_dtypes(include="number").columns
    obj_cols = X_train.select_dtypes(exclude="number").columns

    num_imputer = SimpleImputer(strategy="median")
    obj_imputer = SimpleImputer(strategy="most_frequent")

    if not num_cols.empty:
        if X_train_filled.shape[0]:
            X_train_filled[num_cols] = num_imputer.fit_transform(X_train_filled[num_cols])
        if X_test_filled.shape[0]:
            X_test_filled[num_cols] = num_imputer.transform(X_test_filled[num_cols])
    if not obj_cols.empty:
        if X_train_filled.shape[0]:
            X_train_filled[obj_cols] = obj_imputer.fit_transform(X_train_filled[obj_cols])
        if X_test_filled.shape[0]:
            X_test_filled[obj_cols] = obj_imputer.transform(X_test_filled[obj_cols])

    return X_train_filled, X_test_filled


def rescale(X_train, X_test):

    num_cols = X_train.select_dtypes(include="number").columns
    X_train_rescaled = X_train.copy()
    X_test_rescaled = X_test.copy()

    scaler = MinMaxScaler()
    scaler.fit(X_train_rescaled[num_cols])

    X_train_rescaled[num_cols] = scaler.transform(X_train_rescaled[num_cols])
    X_test_rescaled[num_cols] = scaler.transform(X_test_rescaled[num_cols])

    # print(X_train_rescaled, X_test_rescaled)
    return X_train_rescaled, X_test_rescaled

def one_hot_encoding(train_set, test_set):
    combined_set = pd.concat([train_set, test_set], axis=0)

    if combined_set.shape[1] == 1:
        combined_set_encoded = pd.get_dummies(combined_set, columns=combined_set.columns, drop_first=False)
    else:
        obj_cols = combined_set.select_dtypes(exclude='number').columns
        combined_set_encoded = pd.get_dummies(combined_set, columns=obj_cols, dtype=int, drop_first=True)

    train_set_encoded = combined_set_encoded.iloc[:len(train_set), :].reset_index(drop=True)
    test_set_encoded = combined_set_encoded.iloc[len(train_set):, :].reset_index(drop=True)

    return train_set_encoded, test_set_encoded


def data_init(dataset_name, file_path, loop_time=1, _smote=False, ):

    tree_data_train = []
    tree_data_test = []
    nn_data_train = []
    nn_data_test = []
    for i in range(0,loop_time):
        X, y, dataset = get_data(dataset_name, file_path)

        X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X, y, test_size=0.4, random_state=i)
        X_train_tree, X_test_tree = missing_data(X_train_tree, X_test_tree)
        # print(X_train_tree)

        X_train_tree, X_test_tree = one_hot_encoding(X_train_tree, X_test_tree)

        if _smote:
            smote = SMOTE()
            X_train_tree, y_train_tree = smote.fit_resample(X_train_tree, y_train_tree)
        y_train_nn, y_test_nn = one_hot_encoding(y_train_tree, y_test_tree)
        X_train_nn, X_test_nn = rescale(X_train_tree, X_test_tree)

        tree_data_train.append({'X_train': X_train_tree,'y_train': y_train_tree})
        tree_data_test.append({'X_test': X_test_tree,'y_test': y_test_tree})
        nn_data_train.append({'X_train': X_train_nn,'y_train': y_train_nn})
        nn_data_test.append({'X_test': X_test_nn,'y_test': y_test_nn})

    return tree_data_train, tree_data_test, nn_data_train, nn_data_test





def plot_dataset(dataset,dataset_name):

    dataset_copy = pd.DataFrame(columns=dataset.columns).astype(dataset.dtypes)
    dataset, dataset_copy = missing_data(dataset, dataset_copy)
    dataset, dataset_copy = one_hot_encoding(dataset, dataset_copy)

    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix Heatmap - {dataset_name} Dataset')
    plt.savefig(f'./figures/analysis/{dataset_name}_correlation_heatmap.png', bbox_inches='tight')
    plt.close()

    dataset.hist(figsize=(15, 10), bins=20)
    plt.suptitle(f'Feature Distribution - {dataset_name} Dataset')
    plt.savefig(f'./figures/analysis/{dataset_name}_histograms.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    dataset.boxplot()
    plt.title(f'Box Plot for Outlier Detection - {dataset_name} Dataset')
    plt.xticks(rotation=45)
    plt.savefig(f'./figures/analysis/{dataset_name}_boxplot.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    X1, y1, dataset1 = get_abalone_raw_data('../data/abalone.data')
    X2, y2, dataset2 = get_cmc_raw_data('../data/cmc.data')
    X3, y3, dataset3 = get_adult_raw_data('../data/adult.data')
    tree_data_train1, tree_data_test1, nn_data_train1, nn_data_test1 = data_init('abalone', '../data/abalone.data')
    tree_data_train2, tree_data_test2, nn_data_train2, nn_data_test2 = data_init('cmc', '../data/cmc.data')
    tree_data_train3, tree_data_test3, nn_data_train3, nn_data_test3 = data_init('adult', '../data/adult.data')

    print(0)