import numpy as np
import pandas as pd
import catboost
import platform
import timeit
import urllib.request
import os


# DATASET_PARAMS is a dictionary of experiment parameters to estimate accuracy
# and performance of applying CatBoost models:
#   'key' is  dataset name
#   'value' is a list of corresponding parameters
#     'model_path' is a file name of the trained model
#     'dataset_path' is a file name of test dataset
#     'metric' is an accuracy metric
#     'sample_number' is a number of test samples used to calculate accuracy
#     'iter_count' is a number of iterations to repeat applying the CatBoost
#                  model during calculating performance metric
DATASET_PARAMS = {
    'santander-customer-transaction' : {
        'model_path' : 'models/santander-customer-transaction-prediction_v1.cbm',
        'dataset_path' : 'datasets/santander-customer-transaction-prediction/train.npz',
        'metric' : 'Accuracy',
        'sample_number' : 1000,
        'iter_count' : 10},
    'covertype' : {
        'model_path' : 'models/covertype_v1.cbm',
        'dataset_path' : 'datasets/covertype/test.npz',
        'metric' : 'Accuracy',
        'sample_number' : 1000,
        'iter_count' : 10},
    'yearpredictionmsd':{
        'model_path' : 'models/yearpredictionmsd_v1.cbm',
        'dataset_path' : 'datasets/yearpredictionmsd/test.npz',
        'metric' : 'RMSE',
        'sample_number' : 1000,
        'iter_count' : 10},
    'MQ2008':{
        'model_path' : 'models/MQ2008-prediction_v1.cbm',
        'dataset_path' : 'datasets/MQ2008/test.npz',
        'metric' : 'NDCG:type=Base',
        'sample_number' : 1000,
        'iter_count' : 10},
    'image-embeddings':{
        'model_path' : 'models/image-embeddings_v1.cbm',
        'dataset_path' : 'datasets/image-embeddings/test.npz',
        'metric' : 'Accuracy',
        'sample_number' : 1000,
        'iter_count' : 10}
}


def load_catboost_model(model_name, path_to_weights, log):
    log.info(f'Loading model {model_name} from {path_to_weights}')
    if model_name == 'yearpredictionmsd':
        return catboost.CatBoostRegressor().load_model(path_to_weights, format="cbm")
    elif model_name == 'MQ2008':
        return catboost.CatBoostRanker().load_model(path_to_weights, format="cbm")
    return catboost.CatBoostClassifier().load_model(path_to_weights, format="cbm")


def load_npz_dataset(model_name, path_to_dataset, log):
    log.info(f'Loading dataset from {path_to_dataset}')
    loaded = np.load(path_to_dataset)
    if model_name == 'MQ2008':
        return {'data': loaded['data'],
                'data_target': loaded['target'],
                'data_queries': loaded['queries']}
    return {'data': loaded['data'],
            'data_target': loaded['target']}


def get_device_name():
    return {'Name': platform.node()}


def make_image_embedding_pool(dataset, need_target = True, subsample = 0):
    test_df = pd.DataFrame()
    if (type(dataset) == dict):
        test_df['embed'] = dataset['data'].tolist()
    else:
        test_df['embed'] = dataset.tolist()

    if subsample > 0:
        if need_target:
            pool = catboost.Pool(test_df.head(subsample),
                             label=dataset['data_target'][:subsample],
                             embedding_features=['embed'])
        else:
            pool = catboost.Pool(test_df.head(subsample),
                             embedding_features=['embed'])
    else:
        if need_target:
            pool = catboost.Pool(test_df, label=dataset['data_target'], embedding_features=['embed'])
        else:
            pool = catboost.Pool(test_df, embedding_features=['embed'])
    return pool
