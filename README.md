# sbn
Generalizing Phylogenetic Posterior Estimator from MCMC samples via [subsplit Bayesian networks](https://arxiv.org/pdf/1805.07834.pdf).

## Usage Example
```python
from models import SBN

# parameters to set up the model
#   taxa is the taxa list of the dataset
#   emp_tree_freq is the empirical frequency dictionary of the trees, can be left None if kl divergence computation is not required.
model = SBN(taxa, emp_tree_freq)

# parameters to train to the model
#   tree_dict is the unique tree dictionary
#   tree_names is the name list of the trees
#   tree_wts is the corresponding frequencies for the trees with names in tree_names

# run sbn-sa
model.bn_train_prob(tree_dict, tree_names, tree_wts)
# run sbn-em
logp = model.bn_em_prob(tree_dict, tree_names, tree_wts, maxiter=200, abstol=1e-05, monitor=True, MAP=False)

```

