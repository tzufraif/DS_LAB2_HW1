import pandas as pd
import random
import time
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import PolynomialFeatures
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split




def preprocess(mode='train'):
    X = np.array(np.zeros((1, 78)))
    for file_name in os.listdir(path=mode):
        df = pd.read_csv(f"{mode}/{file_name}", sep='|')
        df = df.drop(columns=df.iloc[:, 7:34].columns, axis=1)
        df = df.drop(['Unit1', 'Unit2'], axis=1)
        df = df.interpolate(method='linear')
        df = df.fillna(df.mean())
        indices = df[df['SepsisLabel'] == 1].index
        label = 0 if len(indices) == 0 else 1
        df = df.drop(indices[1:])
        describe = df.iloc[:, :-1].describe()  # ignores labels
        row = describe.T.iloc[:, 1:]
        row = row.to_numpy().flatten()
        row = np.append(row, label)
        X = np.vstack([X, row])

    print('@@@@@@@@@@@@@@@@@@@@ Finished! @@@@@@@@@@@@@@@@@@@@')
    X = np.delete(X, 0, 0)
    X = pd.DataFrame(X)
    X.iloc[:, :-1] = (X.iloc[:, :-1] - X.iloc[:, :-1].mean()) / X.iloc[:, :-1].std()


    X = X.interpolate(method='linear')
    X = X.fillna(X.mean())
    X.drop(57, axis=1, inplace=True)
    if mode == 'train':
        X = down_sampling(X)

    return X.iloc[:, :-1], X.iloc[:, -1]


def poly_transformation(df):
    """
    Polynomial transformation of given dataframe
    :param df: Pandas DataFrame object
    :return: Transformed DataFrame
    """
    degree = 2
    polynomialFeatures = PolynomialFeatures(degree)
    X_transformed = polynomialFeatures.fit_transform(df)
    X_transformed = np.append(X_transformed, df, axis=1)
    print(f'The polynomial feature engneering added {X_transformed.shape[1] - df.shape[1]} features')
    return X_transformed


def plots(Y):
    """
    Generate plots
    :param Y: Test and preds tuple
    """
    y_test1, y_pred1, y_test2, y_pred2, y_test3, y_pred3, y_test4, y_pred4 = Y
    plt.figure(0).clf()

    # SVM
    fpr, tpr, _ = metrics.roc_curve(y_test1, y_pred1)
    auc = round(metrics.roc_auc_score(y_test1, y_pred1), 4)
    plt.plot(fpr, tpr, label="SVM, AUC=" + str(auc))

    # Decision Tree
    fpr, tpr, _ = metrics.roc_curve(y_test2, y_pred2)
    auc = round(metrics.roc_auc_score(y_test2, y_pred2), 4)
    plt.plot(fpr, tpr, label="Decision Tree, AUC=" + str(auc))

    # Random Forest
    fpr, tpr, _ = metrics.roc_curve(y_test3, y_pred3)
    auc = round(metrics.roc_auc_score(y_test3, y_pred3), 4)
    plt.plot(fpr, tpr, label="Random Forest, AUC=" + str(auc))

    # Gradient Boosting
    fpr, tpr, _ = metrics.roc_curve(y_test4, y_pred4)
    auc = round(metrics.roc_auc_score(y_test4, y_pred4), 4)
    plt.plot(fpr, tpr, label="Gradient Boosting, AUC=" + str(auc))

    # Legend
    plt.legend()
    plt.title('ROC Curve')
    plt.savefig('output3.jpg')
    plt.show()


def down_sampling(df):
    """
    Performs down sampling for the given df
    :param df: Pandas DataFrame object
    :return: Df after down sampling procedure
    """
    columns = df.columns
    feature_names = columns[:-1]
    target_name = columns[-1]
    X, y = (RandomUnderSampler(sampling_strategy=1 / 3, random_state=42)
            .fit_resample(df.drop(target_name, axis=1), df[target_name]))

    rus_df = pd.DataFrame(np.concatenate((X, y[:, None]), axis=1),
                          columns=list(feature_names) + [target_name])
    print('# Data:', len(rus_df))
    print('% Undersampled + Dropped NA / Original:', 100 * len(rus_df) / len(df))
    print('% Positive target:', rus_df[target_name].mean() * 100)
    return rus_df


if __name__ == '__main__':
    seed = 42
    random.seed(42)
    train_prep = time.time()
    X_train, y_train = preprocess('train')
    # down sampling occured in perprocess
    X_train.to_csv('X_train_no_poly.csv')
    # X_train = poly_transformation(X_train)
    # X_train.to_csv('X_train_with_poly.csv')
    y_train.to_csv('y_train.csv')
    # X_train = pd.read_csv('X_train.csv')
    # y_train = pd.read_csv('y_train.csv')
    # X_train.to_csv('X_train.csv')
    # y_train.to_csv('y_train.csv')
    """models = [(SVC(random_state=42), 'SVM'), (DecisionTreeClassifier(random_state=42), 'Decision Tree'),
              (RandomForestClassifier(n_estimators=500, random_state=42), 'Random Forest'),
              (GradientBoostingClassifier(n_estimators=500, random_state=42), 'Gradient Boosting')]"""
    Y = []
    X_test, y_test = preprocess('test')
    X_test.to_csv('X_test_no_poly.csv')
    # X_test = poly_transformation(X_test)
    # X_test.to_csv('X_test_with_poly.csv')
    y_test.to_csv('y_test.csv')

