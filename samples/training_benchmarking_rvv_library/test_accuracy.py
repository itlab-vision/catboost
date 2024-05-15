import os
import sys
import argparse
import logging as log
import utils
import numpy as np
from catboost import Pool
import utils
import timeit


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset',
                        help='Dataset name',
                        choices = ['MQ2008', 'covertype', 'image-embeddings',
                                   'santander-customer-transaction',
                                   'yearpredictionmsd'],
                        default='santander-customer-transaction',
                        dest='dataset')

    args = parser.parse_args()
    return args


def measure_accuracy(dataset_name, model_path, dataset_path, metric_name, log):
    # Load model
    model = utils.load_catboost_model(dataset_name, model_path, log)
    # Load dataset
    dataset = utils.load_npz_dataset(dataset_name, dataset_path, log)

    # Run accuracy check
    start = timeit.default_timer()
    if dataset_name == 'MQ2008':
        result = model.eval_metrics(Pool(dataset['data'],
            label=dataset['data_target'], group_id=dataset['data_queries']),
            metrics = [metric_name])
    elif dataset_name == 'image-embeddings':
        result = model.eval_metrics(utils.make_image_embedding_pool(dataset),
            metrics = [metric_name])
    else:
        result = model.eval_metrics(Pool(dataset['data'],
            label=dataset['data_target']), metrics = [metric_name])
    end = timeit.default_timer()
    log.info(f'Accuracy check time: {end-start}')

    metric_value = result[metric_name][-1]

    log.info(f'Accuracy test results: {metric_name}: {metric_value}')
    return {'accuracy_metric': metric_name,
            'accuracy_value' : metric_value}


def main():
    log.basicConfig(
        format='[ %(levelname)s ] %(message)s',
        level=log.INFO,
        stream=sys.stdout,
    )
    args = cli_argument_parser()
    dataset_params = utils.DATASET_PARAMS[args.dataset]

    measure_accuracy(args.dataset,
                     dataset_params['model_path'],
                     dataset_params['dataset_path'],
                     dataset_params['metric'],
                     log)


if __name__ == '__main__':
    sys.exit(main() or 0)
