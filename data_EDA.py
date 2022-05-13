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

"""
def preprocess(mode='train'):
    X = pd.DataFrame()
    for file_name in os.listdir(path=mode):
         df = pd.read_csv(f"{mode}/{file_name}", sep='|')
         df = df.drop(columns=df.iloc[:,7:34].columns, axis =1)
         df = df.drop(['Unit1', 'Unit2'], axis = 1)
         df = df.interpolate(method = 'linear')
         df = df.fillna(df.mean())
         indices = df[df['SepsisLabel']==1].index
         label = 0 if len(indices) == 0 else 1
         df = df.drop(indices[1:])
  # df = df.iloc[:,:-1].mean()
         describe = df.iloc[:,:-1].describe() #ignores labels
         features = describe.index[1:] #ignores count
         x = pd.DataFrame()
         for feature in features:
             x = pd.concat([x, describe.loc[feature,:]], axis =0)
             row = x.transpose()
             row['SepsisLabel'] = label
         X = pd.concat([X, row], axis = 1) # now df is a vector with mean representation for each column
    # X.to_csv('X_train_clean.csv')
    print('@@@@@@@@@@@@@@@@@@@@ Finished! @@@@@@@@@@@@@@@@@@@@')
    X.iloc[:, :-1] = (X.iloc[:, :-1] - X.iloc[:, :-1].mean()) / X.iloc[:, :-1].std()
    X = X.interpolate(method='linear')
    X = X.fillna(X.mean())
    X.dropna(axis=1, inplace = True)
    if mode == 'train':
        X = down_sampling(X)

    return X.iloc[:, :-1], X.iloc[:, -1]
"""


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
    # now df is a vector with mean representation for each column
    print('@@@@@@@@@@@@@@@@@@@@ Finished! @@@@@@@@@@@@@@@@@@@@')
    X = np.delete(X, 0, 0)
    X = pd.DataFrame(X)
    X.iloc[:, :-1] = (X.iloc[:, :-1] - X.iloc[:, :-1].mean()) / X.iloc[:, :-1].std()

    # for col in X:
    #     X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.interpolate(method='linear')
    X = X.fillna(X.mean())
    # df.dropna(axis=1, inplace = True)
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
"""
    for model, model_name in models:
        clf = make_pipeline(StandardScaler(), model)
        clf.fit(X_train, y_train)

        test_prep = time.time()
        # X_test = pd.read_csv('X_test.csv')
        # y_test = pd.read_csv('y_test.csv')
        print(f'train preprocess time: {time.time() - test_prep}')

        # X_train.to_csv('X_test.csv')
        # y_train.to_csv('y_test.csv')
        y_pred = clf.predict(X_test)
        print(f'{model_name} - F1 score is:{f1_score(y_test, y_pred)}')

        target_names = ['class 0', 'class 1']
        # print(classification_report(y_test, y_pred, target_names=target_names))
        Y.append(y_test)
        Y.append(y_pred)

    plots(Y)
    print(f'Total time: {time.time() - train_prep}')
"""
