import argparse
import pandas as pd
from catboost import CatBoostRegressor, Pool
import numpy as np
import time
import sys


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model',
                        help='The CatBoost model trained on the YearPredictionMSD '
                             'dataset (.cbm file)',
                        required=True,
                        type=str,
                        dest='model_path')
    parser.add_argument('-td', '--test-dataset',
                        help='File name of the YearPredictionMSD test dataset '
                             '(small dataset which consists of 1000 samples '
                             'of the YearPredictionMSD test dataset)',
                        required=True,
                        type=str,
                        dest='test_dataset')
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

        print ('%r: %2.4f sec' % (method.__name__, te-ts))
        return result

    return timed

@timeit
def predict(model, test_df):
    return model.predict(Pool(test_df, thread_count=1), thread_count=1)


def main():
    args = cli_argument_parser()

    # Load the subset of 1000 samples of the YearPredictionMSD dataset
    test_df = pd.read_csv(args.test_dataset, index_col = False)

    # Load reference prediction values computed on the x86 platform
    y_test = np.load(args.reference_prediction)

    # Load the CatBoost model trained on the YearPredictionMSD train dataset
    model = CatBoostRegressor().load_model(args.model_path, format="cbm")

    # Compute predictions for the subset of YearPredictionMSD on the current platform
    y_pred = predict(model, test_df)

    # Calculate difference between predicted and reference values
    diff = abs(y_pred - y_test)

    # Print calculated difference
    print(f'Average diff: {np.mean(diff)}, summary diff: {np.sum(diff)}')


if __name__ == "__main__":
    sys.exit(main() or 0)
