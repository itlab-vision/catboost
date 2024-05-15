import argparse
import logging as log
import os
import sys
import test_accuracy
import test_performance
import utils
import pandas as pd


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--result_file',
                        help='Output file for saving accuracy and performance '
                             'results',
                        default='results.csv',
                        type=str,
                        dest='result_file')

    args = parser.parse_args()

    return args


def main():
    log.basicConfig(
        format='[ %(levelname)s ] %(message)s',
        level=log.INFO,
        stream=sys.stdout,
    )
    args = cli_argument_parser()

    # List of datasets
    datasets = ['santander-customer-transaction',
                'covertype',
                'yearpredictionmsd',
                'MQ2008',
                'image-embeddings',]

    results = []
    for dataset_name in datasets:
        dataset_params = utils.DATASET_PARAMS[dataset_name]
        log.info(f'__________[{dataset_name}]____________')
        log.info(f'Start processing dataset')
        row = {}
        cpu_name = utils.get_device_name()
        row.update(cpu_name)
        dataset_info = {'dataset': dataset_name}
        row.update(dataset_info)
        performance = test_performance.measure_performance(
            dataset_name,
            dataset_params['model_path'],
            dataset_params['dataset_path'],
            dataset_params['sample_number'],
            dataset_params['iter_count'],
            log)
        row.update(performance)
        accuracy = test_accuracy.measure_accuracy(
            dataset_name,
            dataset_params['model_path'],
            dataset_params['dataset_path'],
            dataset_params['metric'],
            log)
        row.update(accuracy)
        results.append(row)
    log.info(f'Writing results to file {args.result_file}')
    pd.DataFrame(results).to_csv(args.result_file, index = None)


if __name__ == '__main__':
    sys.exit(main() or 0)
