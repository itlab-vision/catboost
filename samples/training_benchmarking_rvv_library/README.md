# CatBoost samples

## Directory structure

1. Scripts to train CatBoost models named `<dataset-name>-training.py`.
1. Scripts to apply trained CatBoost models named `<dataset-name>-prediction.py`.
1. Script to calculate accuracy and estimate performance named `main_benchmark.py`.
   This script calls functions from `test_accuracy.py`, `test_performance.py`
   and `utils.py`.
1. The `embeddings_dataset_generation` directory contains auxiliary scripts
   to generate one of the datasets used in CatBoost samples.

The `<dataset-name>` parameter takes one of the following values: `covertype`,
`MQ2008`, `santander-customer-transaction`, `yearpredictionmsd`,
or `image-embedding`.

## How to get datasets?

To train or apply CatBoost models, please, download datasets using
the following links. It is further assumed that the dataset `<dataset-name>`
is located in the directory `./datasets/<dataset-name>`.

1. [Covertype][covertype].
1. [MQ2008][MQ2008].
1. [Santander Customer Transaction][Santander-Customer-Transaction].
1. [Year Prediction MSD][Year-Prediction-MSD].
1. Image embedding dataset is a dataset extracted from
   the [PASCAL VOC 2007 dataset][PASCAL-VOC-2007] using the Python-script
   located in the `embeddings_dataset_generation` directory. To generate embeddings,
   download the PASCAL VOC dataset (VOCtrainval_06-Nov-2007.tar and VOCtest_06-Nov-2007.tar archives)
   from the official [website][PASCAL-VOC-2007]. Unzip dataset into the
   `datasets/voc/trainval/VOC2007` and `datasets/voc/test/VOC2007` directories respectively, so that 5
   subfolders appear in the VOC2007 directory. Further run the command line given below.

   ```bash
   python embeddings_dataset_generation.py
   ```

## How to run samples?

### Train CatBoost models

It is recommended to use powerfull CPU or GPU to train machine learning models.
To train Catboost models, please, run one of the following command lines
depending on the dataset you are interested in. Each
`<dataset-name>-training.py` script reads a dataset, trains a CatBoost model
on the train subset of samples, saves a subset of test samples consisting
of the first 1000 examples and corresponding predictions obtained for these
samples using the CatBoost model. We use 1000 samples to check implementation
correctness and to calculate accuracy on low-power hardware (for example,
on RISC-V devices). It necessary you can change the specified number
of samples.

```bash
python covertype-training.py           \
   -ds datasets/covertype/covtype.data
```

```bash
python MQ2008-training.py -ds ./datasets/MQ2008/Fold1/
```

```bash
python  santander-customer-transaction-training.py \
   -ds datasets/santander-customer-transaction-prediction
```

```bash
python yearpredictionmsd-training.py                    \
   -ds datasets/yearpredictionmsd/YearPredictionMSD.txt
```

```bash
python image-embedding-training.py -ds datasets/image-embeddings
```

### Check correctness, calculate accuracy and estimate performance of applying CatBoost models

To check correctness and estimate performance of applying CatBoost models
when you train model on one platform (x86, for example) and apply it on
the another one (RISC-V, for example), please, run the following command lines.
Each script `<dataset-name>-prediction.py` calculates the difference between
predictions obtained after applying model on the target platform (RISC-V)
and reference results saved after training (x86), and estimates performance
of applying model.

```bash
python  covertype-prediction.py           \
    -m  models/covertype_v1.cbm           \
    -td datasets/covertype-1000.csv       \
    -rp datasets/covertype-1000-proba.npy
```

```bash
python   MQ2008-prediction.py                     \
    -m   models/MQ2008-prediction_v1.cbm          \
    -td  datasets/mq2008_test_1000.npy            \
    -tdt datasets/mq2008_test_target_1000.npy     \
    -tdq datasets/mq2008_test_queries_1000.npy    \
    -rp  datasets/mq2008_test_1000-prediction.npy
```

```bash
python  santander-customer-transaction-prediction.py             \
    -m  models/santander-customer-transaction-prediction_v1.cbm  \
    -td datasets/santander-customer-transaction-1000.csv         \
    -rp datasets/santander-customer-transaction-1000-proba.npy
```

```bash
python  yearpredictionmsd-prediction.py                 \
    -m  models/yearpredictionmsd_v1.cbm                 \
    -td datasets/yearpredictionmsd-1000.csv             \
    -rp datasets/yearpredictionmsd-1000-prediction.npy
```

```bash
python  image-embedding-prediction.py                 \
    -m  models/image-embeddings_v1.cbm                \
    -td datasets/image_embeddings_test_1000.npy       \
    -rp datasets/image_embeddings_test-1000-proba.npy
```

To check accuracy and performance of Catboost models, please, run
the following command line.

```bash
python main_benchmark.py
```


<!-- LINKS -->
[covertype]: https://archive.ics.uci.edu/dataset/31/covertype
[MQ2008]: https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval
[Santander-Customer-Transaction]: https://www.kaggle.com/competitions/santander-customer-transaction-prediction
[Year-Prediction-MSD]: https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
[PASCAL-VOC-2007]: http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007
