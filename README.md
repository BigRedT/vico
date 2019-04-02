# ViCo: Word Embeddings from Visual Co-occurrences

We will assume we are currently in the root directory (which contains the `README.md`). All `bash` or `python` scripts described below will be executed from the root directory.

Before we begin, let us create a directory called `symlinks` in the root directory to store our data and experiments. 
```
mkdir symlinks
```

Because of hardware constraints, I keep my code and experiment outputs on the nfs, and datasets on local disks. So I create directories called `vico_data` and `vico_exp` as per need and create symlinks called `data` and `exp` pointing to these in the `symlinks` directory as follows
```
mkdir path_to_store_datasets/vico_data
ln -s path_to_store_datasets/vico_data symlinks/data

mkdir path_to_store_experiment_outputs/vico_exp
ln -s path_to_store_experiment_outputs/vico_exp symlinks/exp
```

To verify, run `ls -alh symlinks` and you should see something like
```
data -> path_to_store_datasets/vico_data
exp -> path_to_store_experiment_outputs/vico_exp
```

While you can choose any directory for storing datasets and experiments, the code base assumes symlinks to be called `data` and `exp`.

# Code structure

* `./data` contains scripts for downloading and preprocessing:
    * ImageNet and VisualGenome datasets which are used for computing co-occurrences
    * CIFAR-100 which is used for a zero-shot-like analysis
    * Data for Discriminative Attributes Task (SemEval 2018 Task 10) which is a word-only downstream task. 
* `./exp` contains scripts for computing co-occurrence statistics, learning ViCo embeddings, and evaluating embeddings. It also contains training and evaluation scripts for the Discriminative Attributes Task.

# Steps for Learning ViCo embeddings

## Step 1: Create co-occurrence matrices

We compute the following types of co-occurrences from different sources:
* Object-Attribute (VisualGenome)
* Attribute-Attribute (VisualGenome)
* Context (VisualGenome)
* Object-Hypernym (ImageNet)
* Synonyms (WordNet)

If you have already downloaded the datasets, simply run:
```
bash exp/multi_sense_cooccur/create_cooccur.sh
```

This will create the following file which will be used to learn ViCo embeddings:
```
symlinks/exp/multi_sense_cooccur/imagenet_genome_gt/merged_cooccur.csv
```

For inspection, we provide a simple command line utility to load the csv into a pandas dataframe and interactively display all co-occurrences for a given word with other words sorted by co-occurrence types in ascending order (to avoid having to scroll to the top). To launch this utility run:
```
python -m exp.multi_sense_cooccur.explore_merged_cooccur
```
Follow the prompted instructions to interactively explore co-occurrences. To see the instructions again, simply call `usage()`.

## Step 2: Train ViCo's multitask log-bilinear model

## Step 3: Extract embeddings from the saved model

## Step 4: Concat with GloVe

## Step 5: You embeddings are ready to use :metal:

# Evaluation

We provide scripts for the following:
1. Unsupervised clustering analysis
2. Supervised partitioning analysis
3. Zero-Shot-like (visual generalization) analysis
4. Discriminative attributes task evaluation

