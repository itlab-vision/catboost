import os
import sys
import argparse
import logging as log
import utils
import numpy as np
from catboost import Pool
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


def measure_latency_mode(dataset_name, model, dataset, sample_number, log):
    if sample_number == 0 or sample_number > dataset.shape[0]:
        sample_number = dataset.shape[0]
    log.info(f'Measure latency mode for {sample_number} rows')

    # Warmup
    if dataset_name == 'image-embeddings':
        start = timeit.default_timer()
        model.predict(utils.make_image_embedding_pool(dataset, need_target=False, subsample=1)) # warm up model data in cache
        model.predict(utils.make_image_embedding_pool(dataset, need_target=False, subsample=1))
        model.predict(utils.make_image_embedding_pool(dataset, need_target=False, subsample=1))
        end = timeit.default_timer()
    else:
        dataset = dataset[:sample_number]

        start = timeit.default_timer()
        model.predict(Pool(dataset[:1])) # warm up model data in cache
        model.predict(Pool(dataset[:1]))
        model.predict(Pool(dataset[:1]))
        end = timeit.default_timer()

    log.info(f"Warmup time {end-start}")

    # Measure
    if dataset_name == 'image-embeddings':
        times = []
        for row in dataset:
            row = np.expand_dims(row, axis=0)
            start = timeit.default_timer()
            result = model.predict(utils.make_image_embedding_pool(row, need_target=False))
            end = timeit.default_timer()
            times.append(end-start)
        return {'row_average_time' : np.average(times),
            'row_median_time' : np.median(times)}

    times = []
    for row in dataset:
        row = np.expand_dims(row, axis=0)
        start = timeit.default_timer()
        result = model.predict(Pool(row))
        end = timeit.default_timer()
        times.append(end-start)

    return {'row_average_time' : np.average(times),
            'row_median_time' : np.median(times)}


def measure_throughtput_mode(dataset_name, model , dataset, iter_count, log):
    log.info('Measure throughput mode')

    if dataset_name == 'image-embeddings':
        # Warmup
        start = timeit.default_timer()
        model.predict(utils.make_image_embedding_pool(dataset, need_target=False)) # warm up model data in cache
        end = timeit.default_timer()
        log.info(f"Warmup time {end-start}")

        # Measure
        times = []
        for i in range(iter_count):
            start = timeit.default_timer()
            result = model.predict(utils.make_image_embedding_pool(dataset, need_target=False))
            end = timeit.default_timer()
            times.append(end-start)

        return {'full_average_time' : np.average(times),
            'full_median_time' : np.median(times)}



    # Warmup
    start = timeit.default_timer()
    model.predict(Pool(dataset)) # warm up model data in cache
    end = timeit.default_timer()
    log.info(f"Warmup time {end-start}")

    # Measure
    times = []
    for i in range(iter_count):
        start = timeit.default_timer()
        result = model.predict(Pool(dataset))
        end = timeit.default_timer()
        times.append(end-start)

    return {'full_average_time' : np.average(times),
            'full_median_time' : np.median(times)}


def measure_performance(dataset_name, model_path, dataset_path, sample_number, iter_count, log):
    # Load model
    model = utils.load_catboost_model(dataset_name, model_path, log)
    # Load dataset
    dataset = utils.load_npz_dataset(dataset_name, dataset_path, log)
    dataset = dataset['data']

    # Run latency mode
    latency_result = measure_latency_mode(dataset_name, model, dataset, sample_number, log)
    # Run throughput mode
    throughput_result = measure_throughtput_mode(dataset_name, model, dataset, iter_count, log)

    #performance_results = {'latency_mode' : latency_result, 'throughput_mode' : throughtput_result}
    performance_results = {}
    performance_results.update(latency_result)
    performance_results.update(throughput_result)

    log.info(f'Performance test results: {performance_results}')

    return performance_results


def main():
    log.basicConfig(
        format='[ %(levelname)s ] %(message)s',
        level=log.INFO,
        stream=sys.stdout,
    )
    args = cli_argument_parser()
    dataset_params = utils.DATASET_PARAMS[args.dataset]

    measure_performance(
        args.dataset,
        dataset_params['model_path'],
        dataset_params['dataset_path'],
        dataset_params['sample_number'],
        dataset_params['iter_count'],
        log)


if __name__ == '__main__':
    sys.exit(main() or 0)
