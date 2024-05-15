import argparse
import pandas as pd
import catboost
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor, Pool
import json
import os
import sys


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path',
                        help='Path to the YearPredictionMSD dataset',
                        required=True,
                        type=str,
                        dest='dataset_path')
    parser.add_argument('-ot', '--output-test-path',
                        help='Output directory for the subset of 1000 samples '
                             'of the YearPredictionMSD test dataset',
                        type=str,
                        default='datasets',
                        dest='output_test_path')
    parser.add_argument('-om', '--output-model-path',
                        help='Output directory to save the CatBoost model '
                             'trained on the YearPredictionMSD train dataset',
                        type=str,
                        default='models',
                        dest='output_model_path')

    args = parser.parse_args()

    return args


def save_test_data(output_test_path, test, model):
    test = test.head(1000)
    test.to_csv(os.path.join(output_test_path, 'yearpredictionmsd-1000.csv'),
                encoding="utf-8", index=False)

    y_pred = model.predict(Pool(test))
    np.save(os.path.join(output_test_path, 'yearpredictionmsd-1000-prediction.npy'),
            y_pred)


def main():
    args = cli_argument_parser()

    df = pd.read_csv(args.dataset_path, sep=",", header=None, index_col=None)
    train = df.iloc[:463715]
    test = df.iloc[463715:]
    train_target = train.iloc[:,0]
    test_target = test.iloc[:,0]

    train.drop(columns=train.columns[0], axis=1, inplace=True)
    test.drop(columns=test.columns[0], axis=1, inplace=True)

    y_valid_pred = 0 * train_target
    y_test_pred = 0

    model = CatBoostRegressor(loss_function="MAE",
                               eval_metric="RMSE",
                               learning_rate=0.30,
                               iterations=10000,
                               random_seed=2023,
                               od_type="Iter",
                               depth=6,
                               early_stopping_rounds=1000
                              )
    n_split = 5
    kf = KFold(n_splits=n_split, random_state=42, shuffle=True)
    for idx, (train_index, valid_index) in enumerate(kf.split(train)):
        y_train, y_valid = train_target.iloc[train_index], train_target.iloc[valid_index]
        X_train, X_valid = train.iloc[train_index,:], train.iloc[valid_index,:]
        _train = Pool(X_train, label=y_train)
        _valid = Pool(X_valid, label=y_valid)
        print( "\nFold ", idx)
        fit_model = model.fit(_train,
                              eval_set=_valid,
                              use_best_model=True,
                              verbose=200,
                              plot=True
                             )
        pred = fit_model.predict(X_valid)
        print( "  MAE = ", mean_absolute_error(y_valid, pred) )
        y_valid_pred.iloc[valid_index] = pred
        y_test_pred += fit_model.predict(test)
    y_test_pred /= n_split

    print(model.get_best_score())

    y_pred = model.predict(Pool(test))
    error = np.sqrt(mean_squared_error(test_target, y_pred))
    print(error)

    model.save_model(os.path.join(args.output_model_path, 'yearpredictionmsd_v1.cbm'),
               format="cbm",
               export_parameters=None,
               pool=None)

    save_test_data(args.output_test_path, test, model)


if __name__ == "__main__":
    sys.exit(main() or 0)
