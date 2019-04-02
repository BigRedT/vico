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

1. Create co-occurrence matrices for different co-occurrence types:
    * Object-Attribute (VisualGenome)
    * Attribute-Attribute (VisualGenome)
    * Context (VisualGenome)
    * Object-Hypernym (ImageNet)
    * Synonyms (WordNet)
2. Train ViCo's multitask log-bilinear model and save model
3. Extract embeddings from the saved model
4. Concat with GloVe
5. Rock and Roll :metal:

# Evaluation

We provide scripts for the following:
1. Unsupervised clustering analysis
2. Supervised partitioning analysis
3. Zero-Shot-like (visual generalization) analysis
4. Discriminative attributes task evaluation

