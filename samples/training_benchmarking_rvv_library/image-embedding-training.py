import pandas as pd
import catboost
import numpy as np
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
import os
import sys
import argparse


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ds', '--dataset-path',
                        help='Path to the image embeddings dataset',
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


def save_test_data(output_test_path, dataset_path, X_test, y_test, pd, model):
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    np.save(os.path.join(output_test_path, 'image_embeddings_test_1000.npy'), X_test)


    X1000_test_df = pd.DataFrame()
    X1000_test_df['embed'] = X_test.tolist()
    test1000_pool = Pool(X1000_test_df, embedding_features=['embed']) 

    y_pred = model.predict_proba(test1000_pool)
    np.save(os.path.join(output_test_path, 'image_embeddings_test-1000-proba.npy'), y_pred)
    test = np.load(os.path.join(dataset_path, 'resnet18_test_X.npy'))
    test_target = np.load(os.path.join(dataset_path, 'resnet18_test_y.npy'))    
    np.savez_compressed(os.path.join(dataset_path, 'test.npz'), data=test, target=test_target)



def main():
    args = cli_argument_parser()


    X_train = np.load(os.path.join(args.dataset_path, 'resnet18_train_X.npy'))
    y_train = np.load(os.path.join(args.dataset_path, 'resnet18_train_y.npy'))
        
    X_test = np.load(os.path.join(args.dataset_path, 'resnet18_test_X.npy'))
    y_test = np.load(os.path.join(args.dataset_path, 'resnet18_test_y.npy'))


    X_train_df = pd.DataFrame()
    X_train_df['embed'] = X_train.tolist()
        
    X_test_df = pd.DataFrame()
    X_test_df['embed'] = X_test.tolist()

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='Accuracy',
        iterations=10000,
        learning_rate = 0.05,
        random_seed=2023,
        depth = 4,
        od_type="Iter",
        early_stopping_rounds=1000
        )

    train_pool = Pool(X_train_df, y_train, embedding_features= ['embed'])
    test_pool = Pool(X_test_df, y_test, embedding_features= ['embed'])

    fit_model = model.fit(train_pool,
                          eval_set=test_pool,
                          use_best_model=True,
                          verbose=1000,
                          plot=True
                         )

    pred = fit_model.predict_proba(X_test_df)
    print( "  acc = ", accuracy_score(y_test, pred.argmax(axis=1)) )

    print(model.get_best_score())

    y_pred = model.predict(test_pool)
    score = accuracy_score(y_test, y_pred.argmax(axis=1))
    print(score)

    model.save_model(os.path.join(args.output_model_path,
                                  'image-embeddings_v1.cbm'),
               format="cbm",
               export_parameters=None,
               pool=None)
    save_test_data(args.output_test_path, args.dataset_path, 
                   X_test, y_test, pd, model)


if __name__ == "__main__":
    sys.exit(main() or 0)
