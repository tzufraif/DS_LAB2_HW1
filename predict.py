import pickle
import pandas as pd
import numpy as np
from preprocessing import preprocess
import argparse

if __name__ == '__main__':
    # Parsing script arguments
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()
    print('-----------------------------------------')
    print(f'given data path: {args.input_folder}...')
    print('-----------------------------------------')
    print(f'Working on preprocessing...')
    print('-----------------------------------------')
    X, y = preprocess(args.input_folder, 'test')
    print('Finished preprocessing...')
    models_names = ['SVM', 'Random Forest', 'Gradient Boosting']
    predictions = []
    models_dataframe = pd.DataFrame()
    for model_name in models_names:
        print('-----------------------------------------')
        print(f'Current model load: {model_name}...')
        clf = pickle.load(open(model_name, 'rb'))
        print('Start predictions')
        y_pred = clf.predict(X)
        predictions.append(y_pred)
    for prediction, model_name in zip(predictions, models_names):
        models_dataframe[model_name] = np.array(prediction)
    y_pred_new = [1 if np.sum(row) >= 2 else 0 for row in models_dataframe.values]
    final_df = pd.DataFrame()
    final_df['Id'] = X.index.values
    final_df['SepsisLabel'] = np.array(y_pred_new)
    print('-----------------------------------------')
    print('Creating CSV with predictions results')
    final_df.to_csv('prediction.csv', index=False, header=False)
    print('Finish running')