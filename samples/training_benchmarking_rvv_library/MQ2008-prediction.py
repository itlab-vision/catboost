import argparse
import pandas as pd
from catboost import CatBoostRanker, Pool
import numpy as np
import time
import sys


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model',
                        help='The CatBoost model trained on the MQ2008 dataset '
                             '(.cbm file)',
                        required=True,
                        type=str,
                        dest='model_path')
    parser.add_argument('-td', '--test-dataset',
                        help='File name of the MQ2008 test dataset '
                             '(small dataset which consists of 1000 samples '
                             'of the MQ2008 test dataset)',
                        required=True,
                        type=str,
                        dest='test_dataset')
    parser.add_argument('-tdt', '--test-target',
                        help='File name of the MQ2008 test target'
                             '(small dataset which consists of 1000 samples '
                             'of the MQ2008 test dataset)',
                        required=True,
                        type=str,
                        dest='test_target')
    parser.add_argument('-tdq', '--test-queries',
                        help='File name of the MQ2008 test queries'
                             '(small dataset which consists of 1000 samples '
                             'of the MQ2008 test dataset)',
                        required=True,
                        type=str,
                        dest='test_queries')
    parser.add_argument('-rp', '--reference-prediction',
                        help='Prediction vector for these 1000 samples computed '
                             'on the x86 platform to compare predicted values '
                             'with the corresponding ones',
                        required=True,
                        type=str,
                        dest='reference_prediction')

    args = parser.parse_args()

    return args


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r: %2.4f sec' % (method.__name__, te-ts))
        return result

    return timed

@timeit
def predict(model, test, test_target, test_queries):
    return model.predict(Pool(test, label=test_target, group_id=test_queries,
                              thread_count=1), thread_count=1)


def main():
    args = cli_argument_parser()

    # Load the subset of 1000 samples of the MQ2008 dataset
    test = np.load(args.test_dataset)
    test_target = np.load(args.test_target)
    test_queries = np.load(args.test_queries)

    # Load reference prediction values computed on the x86 platform
    reference_predict = np.load(args.reference_prediction)

    # Load the CatBoost model trained on the MQ2008 train dataset
    model = CatBoostRanker().load_model(args.model_path, format="cbm")

    # Compute predictions for the subset of MQ2008 on the current platform
    current_predict = predict(model, test, test_target, test_queries)

    # Calculate difference between predicted and reference values
    diff = abs(reference_predict - current_predict)

    # Print calculated difference
    print(f'Average diff: {np.mean(diff)}, summary diff: {np.sum(diff)}')


if __name__ == "__main__":
    sys.exit(main() or 0)
