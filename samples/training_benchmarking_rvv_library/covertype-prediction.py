import argparse
import pandas as pd
from catboost import CatBoostClassifier, Pool
import numpy as np
import time
import sys


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model',
                        help='The CatBoost model trained on the covertype '
                             'dataset (.cbm file)',
                        required=True,
                        type=str,
                        dest='model_path')
    parser.add_argument('-td', '--test-dataset',
                        help='File name of the covertype test dataset'
                             '(small subset which consists of 1000 samples)',
                        required=True,
                        type=str,
                        dest='test_dataset')
    parser.add_argument('-rp', '--reference-prediction',
                        help='Prediction vector for these 1000 samples '
                             'calculated on the x86 platform to compare '
                             'predicted values with the corresponding ones',
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
def predict(model, test):
    return model.predict_proba(test, thread_count=1)


def main():
    args = cli_argument_parser()

    # Load the subset of 1000 samples of the covertype dataset
    test = pd.read_csv(args.test_dataset)

    # Load reference predictions calculated on the x86 platform
    reference_proba = np.load(args.reference_prediction)

    # Load the CatBoost model trained on the covertype train dataset
    model = CatBoostClassifier().load_model(args.model_path, format="cbm")

    # Calculate predictions for the subset of covertype on the current platform
    current_proba = predict(model, test)

    # Calculate difference between predicted and reference values
    diff = abs(reference_proba - current_proba)

    # Print calculated difference
    print(f'Average diff: {np.mean(diff)}, summary diff: {np.sum(diff)}')


if __name__ == "__main__":
    sys.exit(main() or 0)
