import pandas as pd
import catboost
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
import json
import os
import sys
import argparse


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path',
                        help='Path to the Santander customer transaction '
                             'dataset',
                        required=True,
                        type=str,
                        dest='dataset_path')
    parser.add_argument('-ot', '--output-test-path',
                        help='Output directory for the subset of 1000 samples '
                             'of the Santander customer transaction test dataset',
                        type=str,
                        default='datasets',
                        dest='output_test_path')
    parser.add_argument('-om', '--output-model-path',
                        help='Output directory to save the CatBoost model '
                             'trained on the Santander customer transaction train dataset',
                        type=str,
                        default='models',
                        dest='output_model_path')
 
    args = parser.parse_args()

    return args


def save_test_data(output_test_path, dataset_path, test, model):
    test = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    test = test.head(1000)
    test.to_csv(os.path.join(output_test_path, 'santander-customer-transaction-1000.csv'),
       encoding="utf-8", index=False)

    pred_proba = model.predict_proba(Pool(test.drop(columns=['ID_code'])))
    np.save(os.path.join(output_test_path, 'santander-customer-transaction-1000-proba.npy'),
       pred_proba)



def main():
    args = cli_argument_parser()

    train = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))
    test = pd.read_csv(os.path.join(args.dataset_path, 'test.csv'))
    train_id = train.ID_code
    test_id = test.ID_code
    target = train.target
    train.drop(columns=["ID_code", "target"], inplace=True)
    test.drop(columns=["ID_code"], inplace=True)
    y_valid_pred = 0 * target
    y_test_pred = 0

    model = CatBoostClassifier(loss_function="Logloss",
                               eval_metric="AUC",
                               learning_rate=0.01,
                               iterations=10000,
                               random_seed=2023,
                               od_type="Iter",
                               depth=1,
                               early_stopping_rounds=1000
                              )

    n_split = 5
    kf = KFold(n_splits=n_split, random_state=42, shuffle=True)
    for idx, (train_index, valid_index) in enumerate(kf.split(train)):
        y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]
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
        pred = fit_model.predict_proba(X_valid)[:,1]
        print( "  auc = ", roc_auc_score(y_valid, pred) )
        y_valid_pred.iloc[valid_index] = pred
        y_test_pred += fit_model.predict_proba(test)[:,1]
    y_test_pred /= n_split

    print(model.get_best_score())

    y_pred = model.predict(Pool(train))
    score = accuracy_score(target, y_pred)
    print(score)

    model.save_model(os.path.join(args.output_model_path,
                     'santander-customer-transaction-prediction_v1.cbm'),
               format="cbm",
               export_parameters=None,
               pool=None)

    save_test_data(args.output_test_path, args.dataset_path, test, model)


if __name__ == "__main__":
    sys.exit(main() or 0)
