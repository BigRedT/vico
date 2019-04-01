# Steps for Learning ViCo embeddings

1. Create co-occurrence matrices for different co-occurrence types:
    * Object-Attribute
    * Attribute-Attribute
    * Context
    * Object-Hypernym
    * Synonyms
2. Train ViCo's multitask log-bilinear model and save model
3. Extract embeddings from the saved model
4. Concat with GloVe

# Evaluation

We provide scripts for the following:
1. Unsupervised Clustering Analysis
2. Supervised Partitioning Analysis
3. Zero-Shot-like (visual generalization) Analysis

