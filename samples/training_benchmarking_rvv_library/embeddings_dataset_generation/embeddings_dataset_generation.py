import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
import torchvision.transforms as transforms
from datasets.loader import VOC
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import torchvision.datasets as datasets
from pprint import pprint
import sys
import argparse


def cli_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-od', '--output-dataset-path',
                        help='Output directory for image embeddings datasets',
                        type=str,
                        default='../datasets/image-embeddings',
                        dest='output_dataset_path')

    args = parser.parse_args()

    return args

def main():
    args = cli_argument_parser()


    train_transformer = transforms.Compose(
        [ transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transformer = transforms.Compose(
        [ transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    voc = VOC(batch_size=1, year="2007")
    train_loader = voc.get_loader(transformer=train_transformer, datatype='trainval')
    valid_loader = voc.get_loader(transformer=val_transformer, datatype='test')
    embed_models = {
        'resnet34' : torch.nn.Sequential(*(list(models.resnet34(pretrained=True).children())[:-1])),
    }
    best_results = {}    

    def get_embeddings_with_skip_multilabel(model, dataloader):
        X = []
        y = []
        for i, (images, targets) in tqdm(enumerate(train_loader), total = len(train_loader.dataset)):
            images = images
            targets = targets.cpu().detach().numpy()[0]
            # forward
            pred = model(images).cpu().detach().numpy()
            if np.sum(targets) > 1:
                continue
            X.append(np.squeeze(pred))
            y.append(np.argmax(targets))    
        return X, y


    for model_name in embed_models:
        print(model_name)
        
        # Load model
        cur_model = embed_models[model_name]
        
        # Get embeddings for train
        X_train, y_train = get_embeddings_with_skip_multilabel(cur_model, train_loader)
            
        # Get embeddings for test
        X_test, y_test = get_embeddings_with_skip_multilabel(cur_model, valid_loader)    
            
        # Save embeddings
        np.save(os.path.join(args.output_dataset_path, f"{model_name}_train_X.npy"), X_train)
        np.save(os.path.join(args.output_dataset_path, f"{model_name}_train_y.npy"), y_train)
        np.save(os.path.join(args.output_dataset_path, f"{model_name}_test_X.npy"), X_test)
        np.save(os.path.join(args.output_dataset_path, f"{model_name}_test_y.npy"), y_test)

    for model_name in embed_models:
        print(model_name)
        
        # Prepare dataframe for catboost
        X_train = np.load(os.path.join(args.output_dataset_path, f"{model_name}_train_X.npy"))
        y_train = np.load(os.path.join(args.output_dataset_path, f"{model_name}_train_y.npy"))
        
        X_test = np.load(os.path.join(args.output_dataset_path, f"{model_name}_test_X.npy"))
        y_test = np.load(os.path.join(args.output_dataset_path, f"{model_name}_test_y.npy"))
        
        
        X_train_df = pd.DataFrame()
        X_train_df['embed'] = X_train.tolist()
        
        X_test_df = pd.DataFrame()
        X_test_df['embed'] = X_test.tolist()    
        
        
        train_pool = Pool(X_train_df, y_train, embedding_features= ['embed'])
        test_pool = Pool(X_test_df, y_test, embedding_features= ['embed'])
        
        # Train classifier 
        clf = CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='Accuracy',
            iterations=10000,
            learning_rate = 0.01,
            depth = 4,
            od_type="Iter",
            early_stopping_rounds=1000
            )
        clf.fit(train_pool, eval_set=test_pool, metric_period=10, plot=True, verbose=50)
        
        # Save results
        best_results[model_name] = clf.get_best_score()
        
    pprint(best_results)


if __name__ == "__main__":
    sys.exit(main() or 0)
