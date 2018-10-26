import numpy as np
from Bio import Phylo
from cStringIO import StringIO
from ete3 import Tree
import copy
from collections import defaultdict
EPS = np.finfo(float).eps


# generate full tree space
def generate(taxa):
    if len(taxa) == 3:
        return [Tree('(' + ','.join(taxa) + ');')]
    else:
        res = []
        sister = Tree('(' + taxa[-1] + ');')
        for tree in generate(taxa[:-1]):
            for node in tree.traverse('preorder'):
                if not node.is_root():
                    node.up.add_child(sister)
                    node.detach()
                    sister.add_child(node)
                    res.append(copy.deepcopy(tree))
                    node.detach()
                    sister.up.add_child(node)
                    sister.detach()

        return res


def mcmc_treeprob(filename, data_type, truncate=None):
    mcmc_samp_tree_stats = Phylo.parse(filename, data_type)
    mcmc_samp_tree_dict = {}
    mcmc_samp_tree_name = []
    mcmc_samp_tree_wts = []
    num_hp_tree = 0
    for tree in mcmc_samp_tree_stats:
        handle = StringIO()
        Phylo.write(tree, handle, 'newick')
        mcmc_samp_tree_dict[tree.name] = Tree(handle.getvalue().strip())

        handle.close()
        mcmc_samp_tree_name.append(tree.name)
        mcmc_samp_tree_wts.append(tree.weight)
        num_hp_tree += 1

        if truncate and num_hp_tree >= truncate:
            break

    return mcmc_samp_tree_dict, mcmc_samp_tree_name, mcmc_samp_tree_wts


def summary(dataset, file_path):
    tree_dict_total = {}
    tree_dict_map_total = defaultdict(float)
    tree_names_total = []
    tree_wts_total = []
    n_samp_tree = 0
    for i in range(1, 11):
        tree_dict_rep, tree_name_rep, tree_wts_rep = mcmc_treeprob(file_path + dataset + '/rep_{}/'.format(i) + dataset + '.trprobs', 'nexus')
        tree_wts_rep = np.round(np.array(tree_wts_rep) * 750001)

        for i, name in enumerate(tree_name_rep):
            tree_id = tree_dict_rep[name].get_topology_id()
            if tree_id not in tree_dict_map_total:
                n_samp_tree += 1
                tree_names_total.append('tree_{}'.format(n_samp_tree))
                tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]

            tree_dict_map_total[tree_id] += tree_wts_rep[i]

    for key in tree_dict_map_total:
        tree_dict_map_total[key] /= 10 * 750001

    for name in tree_names_total:
        tree_wts_total.append(tree_dict_map_total[tree_dict_total[name].get_topology_id()])

    return tree_dict_total, tree_names_total, tree_wts_total
