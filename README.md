# Subsplit Bayesian Networks for Generalizing Phylogenetic Posterior Estimation
Thank you for your interest in our paper:
[Generalizing Tree Probability Estimation via Bayesian Networks](https://arxiv.org/pdf/1805.07834.pdf).

Please consider citing the paper when any of the material is used for your research.

## Dependencies

* [ete3](http://etetoolkit.org)
* [Biopython](http://biopython.org)
* [bitarray](https://pypi.org/project/bitarray/)


## Basic Usage

Load MCMC sample
```python
from utils import summary, mcmc_treeprob
# for golden runs
tree_dict_total, tree_names_total, tree_wts_total = summary(dataname, data_directory)
# for sample runs
tree_dict, tree_names, tree_wts = mcmc_treeprob(path_to_data, 'nexus')
```

Run SBN
```python
from models import SBN

# parameters to set up the model
#   @taxa is the taxa list of the dataset
#   @emp_tree_freq is the empirical frequency dictionary of the trees, can be left None if kl divergence computation is not required.
model = SBN(taxa, emp_tree_freq)

# parameters to train the model
#   @tree_dict is the unique tree dictionary
#   @tree_names is the name list of the trees
#   @tree_wts is the corresponding frequencies for the trees with names in tree_names

# run sbn-sa
model.bn_train_prob(tree_dict, tree_names, tree_wts)
# run sbn-em
logp = model.bn_em_prob(tree_dict, tree_names, tree_wts, maxiter=200, abstol=1e-05, monitor=True, MAP=False)

```

Once trained, one can compute the sbn probablities of trees
```
sbn_est_prob = model.bn_estimate(tree)
```
When `emp_tree_freq` is provided, one can evaluate the kl divergence
```
sbn_kl_div = model.kl_div(method='bn')['bn']
```

See more detailed examples in the [jupyter notebooks](https://github.com/zcrabbit/sbn/tree/master/experiments).
