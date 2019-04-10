# ViCo: Word Embeddings from Visual Co-occurrences

# Contents
- [Setup](#setup)
- [Code Structure](#code-structure)
    - [Directories](#directories)
    - [A note on `run.py` files](#a-note-on-`run.py`-files)
- [Steps for Learning ViCo embeddings](#steps-for-learning-vico-embeddings)
    - [Quick Start](#quick-start)
    - [Step 1: Create co-occurrence matrices](#step-1-create-co-occurrence-matrices)
    - [Step 2: Train ViCo's multitask log-bilinear model](#step-2-train-vicos-multitask-log-bilinear-model)
        - [Start Training](#start-training)
        - [Finetune](#finetune)
        - [Monitor Losses](#monitor-losses)
        - [Time and Memory](#time-and-memory)
    - [Step 3: Extract embeddings from the saved model](#step-3-extract-embeddings-from-the-saved-model)
    - [Step 4: Concat with GloVe](#step-4-concat-with-glove)
        - [A note on memory vs speed trade-off for slicing/indexing](#a-note-on-memory-vs-speed-trade-off-for-slicingindexing)
    - [Step 5: Be Awesome :metal:](#step-5-use-vico-embeddings-in-your-awesome-project)
- [Evaluation](#evaluation)
    - [Unsupervised Clustering Analysis](#unsupervised-clustering-analysis)
    - [Supervised Partitioning Analysis](#supervised-partitioning-analysis)
    - [Zero-Shot Analysis](#zero-shot-analysis)

# Setup 

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

[Back to Contents](#contents)

# Code structure

## Directories
* `./data` contains scripts for downloading and preprocessing:
    * ImageNet and VisualGenome datasets which are used for computing co-occurrences
    * CIFAR-100 which is used for a zero-shot-like analysis
    * Data for Discriminative Attributes Task (SemEval 2018 Task 10) which is a word-only downstream task. 
* `./exp` contains scripts for computing co-occurrence statistics, learning ViCo embeddings, and evaluating embeddings. It also contains training and evaluation scripts for the Discriminative Attributes Task.
    
## A note on `run.py` files 
* Each directory in the `./exp` folder typically contains a `run.py` file which contains functions for launching sub-experiments or parts of the same experiment like training, embedding extraction, evaluation etc. 
* The name of these sub-experiment functions, which we refer to as *experiment runners*, begin with `exp_` like `exp_train`, `exp_extract_embeddings` etc. 
* An *experiment runner* specifies all parameters/constants relevant to launching an experiment. Typically a *runner* contains 3 types of constants -
    * `exp_const`: experiment constants like experiment name, optimizer etc
    * `data_const`: data constants like path to data files
    * `model_const`: model constants like network architecture hyperparameters 
* Executing a run file without any arguments, lists all available experiments
    ```
    $ python -m exp.genome_attributes.run
    ---------------------------------------------------------
    Select one of the following exp to run using flag --exp:
    ---------------------------------------------------------
    exp_create_gt_obj_attr_cooccur
    exp_create_gt_attr_attr_cooccur
    exp_create_gt_context_cooccur
    ```
* A specific experiment can be selected and executed using `--exp` argument
    ```
    $ python -m exp.genome_attributes.run --exp exp_create_gt_context_cooccur
    ```

[Back to Contents](#contents)

# Steps for Learning ViCo embeddings

## Quick Start
We provide a simple bash script `./exp/multi_sense_cooccur/scripts/helper.sh` that can be used to run all steps involved in learning ViCo. Simply modify the script to specify the `GPU` on which you want to run the experiments, and the steps you want to run using the `MODE` variable (multiple steps can be specified at once) and execute:
```
bash ./exp/multi_sense_cooccur/scripts/helper.sh
``` 

Each step is explained in detail below along with commands which are the same as the script but with explicit arguments for ViCo(linear,100) as a running example.

## Step 1: Create co-occurrence matrices

We compute the following types of co-occurrences from different sources:
* Object-Attribute (VisualGenome + WordNet)
* Attribute-Attribute (VisualGenome + WordNet)
* Context (VisualGenome + WordNet)
* Object-Hypernym (ImageNet + WordNet)
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

We train ViCo initially with the Adam optimizer for faster convergence and fine-tune with Adagrad.

### Start Training
To start training with Adam, run:
```
CUDA_VISIBLE_DEVICES=0 python \
    -m exp.multi_sense_cooccur.run \
    --exp exp_train \
    --embed_dim 100 \
    --xform linear \
    --syn True
```
`embed_dim` is the ViCo embedding dimension and `xform` is the transformation function to be used in the multitask log-bilinear model. `xform` can alternatively be set to 'select'. It is easy to extend ViCo with other transforms as shown in `./exp/multi_sense_cooccur/models/logbilinear.py`. 

Here, we turned on the use of synonym co-occurrences during training using `--syn`.  However, we empirically found ViCo trained w/o synonym co-occurrences to perform slightly better (see tables in the evaluation section). The version of embeddings used in the paper are trained w/o synonym co-occurrences. If that is the version you are after, go ahead and set `--syn False` here and in all commands below.

### Finetune
To finetune with Adagrad starting from a saved model, say at iteration number 80000, run:
```
CUDA_VISIBLE_DEVICES=0 python \
    -m exp.multi_sense_cooccur.run \
    --exp exp_train \
    --embed_dim 100 \
    --xform linear \
    --model_num 80000 \
    --syn True
```
Experiments data including hyperparameters/constants, tensorboard logs, and models are saved in `./symlinks/exp/multi_sense_cooccur/linear_100/`. Models, by default, are saved every 10000 iterations. Any model number other than `-1` (default) automatically selects Adagrad optimizer. 

### Monitor Losses
Losses can be monitored on tensorboard using
```
tensorboard --logdir=./symlinks/exp/multi_sense_cooccur/
```
<img src="imgs/tensorboard.png" alt="tensorboard logs" width="500">

Note the steep decrease in loss around 80000 iterations due to change in optimizer (learing rate is unchanged). `Neg_*` denotes the losses due to the **max** term while others correspond to the **log-bilinear** loss. `Total_Loss` denotes the sum of all losses.  

### Time and Memory
- Initial training + finetuning: ~8 hours 
- GPU memory usage: ~1GB
- RAM usage: ~7GB 

## Step 3: Extract embeddings from the saved model

To extract embeddings from the embedding layer of the saved model, run:
```
python \
    -m exp.multi_sense_cooccur.run \
    --exp exp_extract_embeddings \
    --embed_dim 100 \
    --xform linear \
    --model_num 160000 \
    --syn True
```
This saves the following files in the `./symlinks/exp/multi_sense_cooccur/linear_100/` directory:
- `visual_embeddings.npy`: VxD matrix of ViCo embeddings where V is the vocabulary size, and D is the embedding dimension
- `visual_embeddings_xformed.npy`: VxD' matrix constructed by concatenating all transformed embeddings. Default configuration results in a concatenation of 50 dimensional transformed embeddings for each co-occurrence type in the following order:
    - Synonyms
    - Attribute-Attribute
    - Object-Attribute
    - Object-Hypernym
    - Context

    Note that for ViCo(select), `visual_embeddings.npy` and `visual_embeddings_xformed.npy` are the same while for ViCo(linear,100), D=100 and D'=250 (=50x5).
- `word_to_idx.json`: A dictionary mapping words to indices in the above npy files

## Step 4: Concat with GloVe

To concatenate GloVe with ViCo, run:
```
python \
    -m exp.multi_sense_cooccur.run \
    --exp exp_concat_with_glove \
    --embed_dim 100 \
    --xform linear \
    --glove_dim 300
```
This creates the following files in the `./symlinks/exp/multi_sense_cooccur/linear_100/concat_with_glove_300` directory:
- `visual_word_vecs.h5py`: An hdf5 file that contains Nx(300+D) dimensional embeddings where N is the GloVe vocabulary (assuming N > V, the visual co-occurrence vocabulary). If a word in GloVe's vocabulary does not exist in the visual vocabulary, the mean of ViCo embeddings is used instead.
- `visual_word_vecs_idx.json`: A dictionary mapping words to indices in the above h5py files.

The embedding for any word can be accessed as follows:
```python
import h5py
import json

# Load json file
word_to_idx = json.load(open('visual_word_vecs_idx.json', 'r'))

# Load h5py file
embeddings = h5py.File('visual_word_vecs.h5py','r')['embeddings']

word = 'green'
idx = word_to_idx[word]
word_embedding = embeddings[idx] # (400,) numpy.ndarray
glove_component = word_embedding[:300]
vico_component = word_embedding[300:]
```

### A note on memory vs speed trade-off for slicing/indexing

The advantage of h5py is that the variable `embeddings` refers to an h5py Dataset object which can be indexed to get individual rows without loading the entire embedding matrix (400000x400) into memory. However, every slice operation like `embeddings[idx]` would have to read from disk. This may not be an issue if slicing is infrequent but could be slow otherwise (for example if used in a dataloader to create mini-batches for a captioning-task).

Therefore, if RAM is not an issue, it is recommended to load the full matrix using:
```python
  embeddings = h5py.File('visual_word_vecs.h5py','r')['embeddings'][()]
  # Notice the '[()]' at the end
```
In this case, `embeddings` is the full embedding matrix loaded in RAM as a numpy array. Since the entire matrix is in RAM, slicing operations are fast. 

## Step 5: Use ViCo embeddings in your awesome project
Build on it, analyze it, improve it ...

OR

Run it through our evaluation protocol as described in the next section to better understand it.

[Back to Contents](#contents)

# Evaluation

We provide scripts for the following:
- Unsupervised clustering analysis
- Supervised partitioning analysis
- Zero-Shot-like (visual generalization) analysis
- Discriminative attributes task evaluation (SemEval 2018, Task 10)


## Unsupervised Clustering Analysis

If you have the ViCo(linear,100) embeddings trained and concatenated with GloVe as described above, you can run the analysis by executing
```
python -m exp.multi_sense_cooccur.run --exp exp_unsupervised_clustering
```
This saves plots in `./symlinks/exp/multi_sense_cooccur/analysis/unsupervised_clustering` directory and prints average performance across cluster numbers in the terminal, which can directly be copied to a latex file

- Clustering performance on fine categories

    | Embedding | V-Measure | ARI |
    |:---------|:---------:|:---:|
    | GloVe | 0.50 | 0.15 |
    | ViCo(linear,100) | 0.59 | 0.22 |
    | GloVe+ViCo(linear,100) | 0.60 | 0.22 |
    | GloVe+ViCo(linear,100) (w/o syn) | **0.61** | **0.23** |

- Clustering performance on coarse categories

    | Embedding | V-Measure | ARI |
    |:---------|:---------:|:---:|
    | GloVe | 0.52 | 0.38 |
    | ViCo(linear,100) | 0.61 | 0.40 |
    | GloVe+ViCo(linear,100) | **0.67** | **0.51** |
    | GloVe+ViCo(linear,100) (w/o syn) | 0.65 | 0.48 |


### What if you want to compare other embeddings?
Usually we want to run such an analysis to compare various embeddings. Runner `exp_unsupervised_clustering` in `./exp/multi_sense_cooccur/run.py` file creates a dictionary called `data_const.embed_info` which contains all necessary information about embeddings which need to be evaluated. The information is provides through an instance of a python class which must have a `get_embedding` method and the following attributes/properties:
- `word_vecs_h5py`: Path to the h5py file
- `word_to_idx_json`: Path to the json file which maps words to indices in the h5py file

See the class `EmbedInfo` for example. Since all of my embeddings followed a consistent naming convention, I could use the same class to automatically construct these attributes.

The `get_embedding` method is used to extract parts of the embedding such as -- the GloVe component, the ViCo component, or keep both.


## Supervised Partitioning Analysis

The runner for the partitioning analysis is similar to the clustering case and can be executed using
```
python -m exp.multi_sense_cooccur.run --exp exp_supervised_partitioning
```
This saves plots in `./symlinks/exp/multi_sense_cooccur/analysis/supervised_partitioning` directory and prints average performance across various tree depths in the terminal, which can directly be copied to a latex file

- Partitioning performance on fine categories

    |Embedding | V-Measure | ARI | Accuracy |
    |:---------|:---------:|:---:|:-------:|
    |GloVe | 0.70 | 0.47 | 0.64 |
    |ViCo(linear,100) | 0.75 | 0.53 | 0.68 |
    |GloVe+ViCo(linear,100) | 0.76 | 0.53 | 0.70 |
    |GloVe+ViCo(linear,100) (w/o syn) | **0.78** | **0.61** | **0.72** |

- Partitioning performance on coarse categories

    |Embedding | V-Measure | ARI | Accuracy |
    |:---------|:---------:|:---:|:-------:|
    |GloVe | 0.77 | 0.74 | 0.84 |
    |ViCo(linear,100) | 0.78 | 0.74 | 0.85 |
    |GloVe+ViCo(linear,100) | 0.79 | 0.77 | 0.85 |
    |GloVe+ViCo(linear,100) (w/o syn) | **0.81** | **0.78** | **0.87** |

## Zero-Shot Analysis

To launch the zero-shot analysis (a.k.a visual generalization analysis) for ViCo(linear,100), run:
```
bash exp/cifar100/scripts/run.sh <num_held_out_classes> <gpu_id> <run>
```
where `num_held_out_classes` is one of {20,40,60,80}, `gpu_id` refers to the GPU on which to run the evaluation, and `run` is an 

| Embeddings | Run 0 | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Run 7 | Mean | Std |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GloVe | 18.20 | 20.0 | 20.50 | 22.65 | 22.15 | 19.80 | 22.65 | 18.90 | 20.61 | 1.60 |
| ViCo(linear,100) | 13.85 | 14.95 | 12.50 | 17.25 | 15.90 | 14.15 | 13.30 | 16.65 | 14.82 | 1.56 |
| GloVe+ViCo(linear,100) | 22.75 | 24.20 | 18.15 | 20.30 | 22.80 | 24.45 | 21.45 | 22.20 | 22.04 | 1.94 |
| GloVe+ViCo(select,200) | **28.40** | **27.00** | **25.90** | **28.80** | **28.10** | **27.60** | **28.85** | **31.25** | **28.24** | 1.47 |

[Back to Contents](#contents)