import pandas as pd
import catboost
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import json
from sklearn.model_selection import train_test_split
import os
import sys
import argparse


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path',
                        help='Path to the covertype dataset',
                        required=True,
                        type=str,
                        dest='dataset_path')
    parser.add_argument('-ot', '--output-test-path',
                        help='Output directory for the subset of 1000 samples '
                             'of the covertype test dataset',
                        type=str,
                        default='datasets',
                        dest='output_test_path')
    parser.add_argument('-om', '--output-model-path',
                        help='Output directory to save the CatBoost model '
                             'trained on the covertype train dataset',
                        type=str,
                        default='models',
                        dest='output_model_path')

    args = parser.parse_args()

    return args


def save_test_data(output_test_path, test, test_target, model):
    np.savez_compressed(os.path.join(output_test_path, 'covertype/test.npz'),
                        data=test, target=test_target)
    test = test.head(1000)
    test.to_csv(os.path.join(output_test_path, 'covertype-1000.csv'),
                encoding="utf-8", index=False)

    pred_proba = model.predict_proba(Pool(test))
    np.save(os.path.join(output_test_path, 'covertype-1000-proba.npy'), pred_proba)


def main():
    args = cli_argument_parser()

    df = pd.read_csv(args.dataset_path, sep=",", header=None, index_col=None)
    train, test = train_test_split(df, test_size=0.3, random_state=2023)
    train_target = train.iloc[:,-1]
    test_target = test.iloc[:,-1]

    train.drop(columns=train.columns[-1], axis=1, inplace=True)
    test.drop(columns=test.columns[-1], axis=1, inplace=True)

    y_valid_pred = 0 * train_target
    y_test_pred = 0


    model = CatBoostClassifier(loss_function="MultiClass",
                               eval_metric="Accuracy",
                               learning_rate=0.5,
                               iterations=5000,
                               random_seed=2023,
                               od_type="Iter",
                               depth=8,
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
                              verbose=1000,
                              plot=True
                             )
        pred = fit_model.predict_proba(X_valid)
        print( "  acc = ", accuracy_score(y_valid, pred.argmax(axis=1) + 1) )
        y_valid_pred.iloc[valid_index] = pred.argmax(axis=1) 

    print(model.get_best_score())

    y_pred = model.predict(Pool(test))
    score = accuracy_score(test_target, y_pred)
    print(score)


    model.save_model(os.path.join(args.output_model_path, 'covertype_v1.cbm'),
                     format="cbm",
                     export_parameters=None,
                     pool=None)

    save_test_data(args.output_test_path, test, test_target, model)


if __name__ == "__main__":
    sys.exit(main() or 0)
