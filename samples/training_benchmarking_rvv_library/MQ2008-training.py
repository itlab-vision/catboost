import argparse
import pandas as pd
import catboost
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import ndcg_score
from catboost import CatBoostRanker, Pool
from sklearn.datasets import load_svmlight_file
import os
import sys


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path',
                        help='Path to the MQ2008 dataset',
                        required=True,
                        type=str,
                        dest='dataset_path')
    parser.add_argument('-ot', '--output-test-path',
                        help='Output directory for the subset of 1000 samples '
                             'of the MQ2008 test dataset',
                        type=str,
                        default='datasets',
                        dest='output_test_path')
    parser.add_argument('-om', '--output-model-path',
                        help='Output directory to save the CatBoost model '
                             'trained on the MQ2008 train dataset',
                        type=str,
                        default='models',
                        dest='output_model_path')
 
    args = parser.parse_args()

    return args


def process_libsvm_file(file_name):
    X, y, queries = load_svmlight_file(file_name, query_id=True)
    return np.array(X.todense(),dtype=float), np.array(y, dtype=int),
                    np.array(queries, dtype=int)


def save_test_data(output_test_path, test, test_queries, ranker):
    test_1000 = test[:1000]
    test_target_1000 = test[:1000]
    test_queries_1000 = test_queries[:1000]

    np.save(os.path.join(output_test_path, 'mq2008_test_1000.npy'), test_1000)
    np.save(os.path.join(output_test_path, 'mq2008_test_target_1000.npy'), test_target_1000)
    np.save(os.path.join(output_test_path, 'mq2008_test_queries_1000.npy'), test_queries_1000)

    pred = ranker.predict(Pool(test_1000, label=test_target_1000, group_id=test_queries_1000))
    np.save(os.path.join(output_test_path, 'mq2008_test_1000-prediction.npy'), pred)


def main():
    args = cli_argument_parser()

    dataset_path = args.dataset_path

    train, train_target, train_queries = process_libsvm_file(os.path.join(dataset_path, "train.txt"))
    valid, valid_target, valid_queries = process_libsvm_file(os.path.join(dataset_path, "vali.txt"))
    test, test_target, test_queries = process_libsvm_file(os.path.join(dataset_path, "test.txt"))

    ranker = CatBoostRanker(
        loss_function='YetiRank',
        eval_metric = 'NDCG',
        learning_rate=0.02,
        iterations=10000,
        random_seed=2023,
        od_type="Iter",
        depth=6,
        early_stopping_rounds=1000,
        )

    _train = Pool(train, label=train_target, group_id=train_queries)
    _valid = Pool(valid, label=valid_target, group_id=valid_queries)

    fit_model = ranker.fit(_train,
                           eval_set=_valid,
                           use_best_model=True,
                           verbose=200,
                           plot=True
                          )

    res = ranker.eval_metrics(Pool(test, label=test_target, group_id=test_queries),
                              metrics = ['NDCG'])
    print(res['NDCG:type=Base'][-1])


    ranker.save_model(os.path.join(args.output_model_path, 'MQ2008-prediction_v1.cbm'),
               format="cbm",
               export_parameters=None,
               pool=None)

    save_test_data(args.output_test_path, test, test_queries, ranker)


if __name__ == "__main__":
    sys.exit(main() or 0)
