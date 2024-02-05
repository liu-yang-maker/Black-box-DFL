# ZODFL

This code implements and evaluates ZODFL method described in "A General Decision-focused Framework for
Graph Learning and Optimization". 



# Files

* experiments_*.py runs the experiments which do link prediction on a given graph.
* models.py contains the definitions for both the ClusterNet model (GCNClusterNet) as well as various models used for the baselines.
* modularity.py contains helper functions and baseline optimization algorithms for the community detection task.
* kcenter.py contains helper functions and baseline optimization algorithms for the facility location task.
* loss_functions.py contains definitions of the loss function used to train ClusterNet and GCN-e2e for both tasks.
* utils.py contains helper functions to load/manipulate the datasets.

# Datasets

Included are the datasets used to run the experiments in the paper. Here are the mappings between the filenames (which can be used as the --dataset argument to the code) and the names of the datasets in the paper:

* cora: the [cora](https://relational.fit.cvut.cz/dataset/CORA) dataset (cora)
* citeseer: the [citeseer](https://linqs.soe.ucsc.edu/data) dataset (citeseer)
* moreno: the [adolescent social network](http://konect.uni-koblenz.de/networks/moreno_health) (adol)
* protein_vidal: the [protein interaction network](http://konect.uni-koblenz.de/networks/maayan-vidal) (protein)
* fb_small: the [facebook network](http://konect.uni-koblenz.de/networks/ego-facebook) (fb)
* pubmed: the [pubmed](https://linqs.soe.ucsc.edu/data) citation dataset (pubmed)
* synthetic_spa: a [synthetic distribution](https://dl.acm.org/citation.cfm?id=3237383.3237507) based on spacial preferential attachment model (synthetic)

# Examples of running the experiments

Example running the single-graph experiment for the community detection problem on the cora dataset:

```
python experiments_singlegraph_kcenter_zeroth.py --objective modularity --hidden 50 --embed_dim 50 --weight_decay 5e-4 --dropout 0.2 --train_iters 1001 --clustertemp 50 --num_cluster_iter 1 --lr 0.01 --dataset cora
```