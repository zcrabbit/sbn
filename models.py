"""
For theory and comments, see "Generalizing Tree Probability Estimation via Bayesian Networks", Zhang & Matsen

Notes:
* Assume the standard total order on bitarrays.
* Addition on bitarrays is concatenation.
* A "composite" bitarray represents a subsplit. Say we have n taxa, and a
  well-defined parent node and child node. The first n bits represent the clade
  of the child node's sister (the parent node's other child) and the second n
  bits represent the clade of the child node itself.
* To "decompose" a composite bitarray means to cut it into two.
"""


import numpy as np
from collections import defaultdict
from bitarray import bitarray
from copy import deepcopy

EPS = np.finfo(float).eps


class SBN:
    def __init__(self, taxa, emp_tree_freq=None, alpha=0.0):
        self.taxa = taxa
        self.ntaxa = len(taxa)
        self.map = {taxon: i for i, taxon in enumerate(taxa)}
        self.alpha = alpha

        if emp_tree_freq is None:
            self.emp_tree_freq = {}
        else:
            self.emp_tree_freq = emp_tree_freq

        # TODO: consider having this be calculated on the fly, in case emp_tree_freq is silently altered.
        self.negDataEnt = np.sum([wts * np.log(wts + EPS) for wts in self.emp_tree_freq.values()])

        self.samp_tree_freq = defaultdict(float)

        # Dictionary containing root split probabilities
        self.clade_dict = defaultdict(float)

        # Dictionary mostly containing joint parent clade-child subsplit data.
        # Used to calculate Pr(child subsplit | parent clade) = CCD probs.
        # It is a double dictionary, such that clade_bipart_dict[w][y] accesses
        # the relevant value for the subsplit (y, w - y).
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))

        # Dictionary mostly containing joint parent subsplit-child subsplit data
        # Used to calculate Pr(child subsplit | parent subsplit) = conditional
        # subsplit distribution (CSD) probs.
        # This is also a double dictionary, such that clade_double_bipart_dict[s][y]
        # where s is a composite bitarray representing the parent subsplit, and
        # y splits the second component of the parent subsplit.
        self.clade_double_bipart_dict = defaultdict(lambda: defaultdict(float))

        # These are normalized sample counts that are used to calculate the contribution from the prior.
        # (See equation directly after Theorem 1).
        # This one corresponds to root split probabilities (like clade_dict) and thus \tilde m_{s_1}^u in the paper.
        self.clade_freq_est = defaultdict(float)
        # This one corresponds to subsplit probabilities (like clade_bipart_dict) and thus \tilde m_{s,t}^u in the
        # paper.
        self.clade_bipart_freq_est = defaultdict(lambda: defaultdict(float))
        # The length of clade_double_bipart_freq_est, which is needed on the denominator.
        self.clade_double_bipart_len = defaultdict(int)

    def _combine_bitarr(self, arrA, arrB):
        """Concatenate two bitarrays with the lesser bitarray on the left.

        :param arrA: bitarray representing a split.
        :param arrB: bitarray representing a split.
        :return: Composite bitarray with the lesser bitarray on the left.
        """
        if arrA < arrB:
            return arrA + arrB
        else:
            return arrB + arrA

    def _merge_bitarr(self, key):
        """OR a composite bitarray, i.e. merge a subsplit into its parent clade.

        :param key: string of 0s and 1s representing a composite bitarray.
        :return: bitarray representing the OR of the two sub-bitarrays.
        """
        return bitarray(key[:self.ntaxa]) | bitarray(key[self.ntaxa:])

    def _decomp_minor_bitarr(self, key):
        """Decomposes a composite bitarray (passed as a character string) and returns the lesser of the two resulting bitarrays.

        :param key: string of 0s and 1s representing a composite bitarray.
        :return: bitarray representing the lesser of the two sub-bitarrays.
        """
        return min(bitarray(key[:self.ntaxa]), bitarray(key[self.ntaxa:]))

    def _minor_bitarr(self, arrA):
        """Symmetry collapse by returning the lesser of the bitarray and its complement.

        :param arrA: bitarray representing a split.
        :return: bitarray containing arrA or its NOT, whichever is lesser.
        """
        return min(arrA, ~arrA)

    def clade_to_bitarr(self, clade):
        """Creates an indicator bitarray from a collection of taxa.

        :param clade: collection containing elements the SBN object's taxa list.
        :return: bitarray indicating which taxa are in clade.
        """
        bit_list = ['0'] * self.ntaxa
        for taxon in clade:
            bit_list[self.map[taxon]] = '1'
        return bitarray(''.join(bit_list))

    def check_clade_dict(self):
        """Print and compare summary statistics for each clade.

        Shows the sum of the root split probability dictionary,
        which should be 1.0.  For each clade, shows the sum of
        clade_bipart_dict[clade][.] (sum of subsplit probabilities) next to
        clade_dict[clade] (clade probabilities) which should be equal.
        """
        print "clade_dict sum: {:.12f}".format(sum(self.clade_dict.values()))
        print "clade_bipart_dict tabular sum:"
        for key in self.clade_bipart_dict:
            bipart_bitarr = self._minor_bitarr(bitarray(key))
            print '{}:{:.12f}|{:.12f}'.format(bipart_bitarr.to01(), sum(self.clade_bipart_dict[key].values()), self.clade_dict[bipart_bitarr.to01()])

    def check_clade_dict_em(self):
        """Print and compare summary statistics for each subsplit.

        Shows the sum of the root split probability dictionary,
        which should be 1.0.  For each subsplit, shows the sum of
        clade_double_bipart_dict[subsplit][.] (parent-child subsplit probabilities)
        next to clade_dict[subsplit] (if it is at the root) or
        clade_bipart_dict[union(subsplit)][subsplit] (if it is not at the root)
        which should be equal.
        """
        print "clade_dict sum: {:.12f}".format(sum(self.clade_dict.values()))
        print "clade_double_bipart_dict tabular sum:"
        for key in self.clade_double_bipart_dict:
            parent_clade_bitarr = self._merge_bitarr(key)
            bipart_bitarr = self._decomp_minor_bitarr(key)
            if parent_clade_bitarr.count() != self.ntaxa:
                print '{}|{}:{:.12f}|{:.12f}'.format(parent_clade_bitarr.to01(), bipart_bitarr.to01(),
                                                     sum(self.clade_double_bipart_dict[key].values()),
                                                     self.clade_bipart_dict[parent_clade_bitarr.to01()][bipart_bitarr.to01()])
            else:
                print '{}|{}:{:.12f}|{:.12f}'.format(parent_clade_bitarr.to01(), bipart_bitarr.to01(),
                                                     sum(self.clade_double_bipart_dict[key].values()),
                                                     self.clade_dict[bipart_bitarr.to01()])

    def logprior(self):
        """Calculate the Dirichlet conjugate prior, namely the two summation
        terms in the equation directly after Theorem 1 of the paper.

        :return: float containing the value of the prior.
        """
        return self.alpha * sum([
            self.clade_freq_est[key] * np.log((self.clade_dict[key] + self.alpha * self.clade_freq_est[key]) / (1.0 + self.alpha))
            for key in self.clade_dict
        ]) + sum([
            self.alpha * self.clade_bipart_freq_est[self._merge_bitarr(key).to01()][self._decomp_minor_bitarr(key).to01()] /
            len(self.clade_double_bipart_dict[key]) * np.sum(
                np.log((np.array(self.clade_double_bipart_dict[key].values()) + self.alpha * self.clade_bipart_freq_est[self._merge_bitarr(
                    key).to01()][self._decomp_minor_bitarr(key).to01()] / len(self.clade_double_bipart_dict[key])) /
                       (self.alpha * self.clade_bipart_freq_est[self._merge_bitarr(key).to01()][self._decomp_minor_bitarr(key).to01()] +
                        self.clade_bipart_dict[self._merge_bitarr(key).to01()][self._decomp_minor_bitarr(key).to01()])))
            if self._merge_bitarr(key).to01() != '1' * self.ntaxa else self.alpha *
            self.clade_freq_est[self._minor_bitarr(bitarray(key[:self.ntaxa])).to01()] / len(self.clade_double_bipart_dict[key]) * np.sum(
                np.log((np.array(self.clade_double_bipart_dict[key].values()) + self.alpha *
                        self.clade_freq_est[self._minor_bitarr(bitarray(key[:self.ntaxa])).to01()] / len(self.clade_double_bipart_dict[key])) /
                       (self.alpha * self.clade_freq_est[self._minor_bitarr(bitarray(key[:self.ntaxa])).to01()] +
                        self.clade_dict[self._minor_bitarr(bitarray(key[:self.ntaxa])).to01()]))) for key in self.clade_double_bipart_dict
        ])

    def clade_update(self, tree, wts):
        """Updates clade distribution.

        Updates the clade dictionary, based on a single tree topology,
        weighted by the fraction of times that tree appears in the
        sample or distribution.

        :param tree: Tree (ete3) unrooted tree object providing the topology.
        :param wts: float representing the fraction of the sampled trees that
        this tree topology represents.
        :return: dictionary mapping nodes to their bitarray representation.
        """
        nodetobitMap = {}
        for node in tree.traverse('levelorder'):
            # NB: tree topology is unrooted, but is stored in a rooted tree format (with a trifurcating root).
            if not node.is_root():
                clade = node.get_leaf_names()
                node_bitarr = self.clade_to_bitarr(clade)
                self.clade_dict[self._minor_bitarr(node_bitarr).to01()] += wts / (2 * self.ntaxa - 3.0)
                nodetobitMap[node] = node_bitarr

        return nodetobitMap

    def ccd_dict_update(self, tree, wts):
        """Updates conditional clade distribution (CCD) with a single topology.

        Updates the CCD dictionary, based on a single unrooted tree topology,
        weighted by the fraction of times that tree appears in the sample or
        distribution. This function does the updating for all rootings of the
        unrooted tree simultaneously. We will call these "virtual roots."

        :param tree: Tree (ete3) unrooted tree object providing the topology.
        :param wts: float representing the fraction of the sampled trees that
        this tree topology represents.
        :return: dictionary mapping nodes to their bitarray representation.
        """
        nodetobitMap = self.clade_update(tree, wts)
        for node in tree.traverse('levelorder'):
            # Updates the conditional clade distribution (CCD) weights for all rootings of an unrooted tree.
            # Recall that the unrooted tree topology is stored in a rooted tree format (with a trifurcating root).
            # Here we traverse that data structure and set values in the CCD dictionary corresponding to all root node
            # assignments for the unrooted tree.
            # Notation: Node and the edge 'above' it are equivalent.
            if not node.is_root():
                # Orientation 1
                # Virtual root node is 'above' node, so node's subsplit splits its 'child' clades.
                if not node.is_leaf():
                    bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    self.clade_bipart_dict[nodetobitMap[node].to01()][bipart_bitarr.to01()] += wts / (2 * self.ntaxa - 3.0)

                # Orientation 2
                # Virtual root node is 'below' node, so node's subsplit splits its 'sister' and/or 'parent' clades.
                if not node.up.is_root():
                    # This is the standard case for this orientation in the middle of the ETE tree. The first term is a
                    # singleton list (the single sister) and the second is the rest of the tree that is above node.up.
                    # Thus, the "+" is list concatenation, not bitarray concatenation.
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    # This is the case that the parent is the the ETE data structure root, so the split is between the
                    # two sisters of node.
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                self.clade_bipart_dict[(~nodetobitMap[node]).to01()][bipart_bitarr.to01()] += wts / (2 * self.ntaxa - 3.0)

        return nodetobitMap

    def bn_dict_update(self, tree, wts, root_wts=None):
        """Updates conditional subsplit distribution (CSD).

        Updates the CSD dictionary, based on a single tree topology,
        weighted by the number of times that tree appears in the
        sample or distribution. This function does the updating for all
        rootings of the unrooted tree simultaneously. We will call these
        "virtual roots."

        :param tree: Tree (ete3) unrooted tree object providing the topology.
        :param wts: float representing the fraction of the sampled trees that
        this tree topology represents.
        :param root_wts: dictionary, optional, mapping node bit
        signatures to edge weights. Used for SBN-EM and SBN-EM-alpha.
        :return: None
        """
        nodetobitMap = self.ccd_dict_update(tree, wts)
        for node in tree.traverse('levelorder'):
            if not root_wts:
                # Simple averaging weighting, used in SBN-SA
                node_wts = wts / (2 * self.ntaxa - 3.0)
            else:
                # Weighted edges, used in SBN-EM and SBN-EM-alpha
                node_wts = wts * root_wts[nodetobitMap[node].to01()]

            # Below we update the conditional subsplit distribution (CSD) weights.
            # Explores the six orientations that this (node, node.up) pair can take, and updates the CSD dictionaries.
            # See the `orientations` image in the `doc/` directory to see a figure depicting these orientations.
            # Notation: Node and the edge 'above' it are equivalent.
            if not node.is_root():
                # Orientations 1,2,3: node is child subsplit, node.up is in the direction of the virtual root.
                if not node.is_leaf():
                    # Given nodetobitMap[node], this bitarray well-defines the subsplit at node.
                    bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    for sister in node.get_sisters():
                        # Orientation 1
                        # If virtual root is beyond node.up, this composite bitarray well-defines the parent subsplit
                        # at node.up.
                        comb_parent_bipart_bitarr = nodetobitMap[sister] + nodetobitMap[node]
                        self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts
                    if not node.up.is_root():
                        # Orientation 2
                        # If virtual root is linked via a sister, this composite bitarray well-defines the parent
                        # subsplit at node.up.
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[node]
                        self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

                    # Orientation 3
                    # If virtual root is on the edge between node and node.up, this composite bitarray well-defines the
                    # root split.
                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

                # Orientations 4,5,6: node.up is child subsplit, node is in the direction of the virtual root.
                # With the root split "below" node, this bitarray well-defines the subsplit at node.up.
                if not node.up.is_root():
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                if not node.is_leaf():
                    # Orientations 4 and 5
                    # If virtual root is beyond node's children, this composite bitarray well-defines the parent
                    # subsplit at node.
                    for child in node.children:
                        comb_parent_bipart_bitarr = nodetobitMap[child] + ~nodetobitMap[node]
                        self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

                # Orientation 6
                # If virtual root is on the edge between node.up and node, this composite bitarray well-defines the
                # root split.
                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

    def ccd_train_count(self, tree_count, tree_id):
        """Extracts the conditional clade distributions from sample trees and stores them in the SBN object.

        :param tree_count: dictionary mapping tree topology ID to count
        of that tree in the sample.
        :param tree_id: dictionary mapping tree topology ID to a
        singleton list containing the tree object.
        """
        # Clear the SBN model dictionaries
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        total_count = sum(tree_count.values()) * 1.0
        # Iterate through trees and update with corresponding CCD values.
        for key in tree_count:
            count = tree_count[key]
            tree = tree_id[key][0]

            wts = count / total_count
            self.ccd_dict_update(tree, wts)

    def ccd_train_prob(self, tree_dict, tree_names, tree_wts):
        """Extracts the conditional clade distributions from tree probabilities and stores them in the SBN object.

        :param tree_dict: dictionary mapping tree topology ID to count
        of that tree in the sample.
        :param tree_names: list of tree topology IDs.
        :param tree_wts: list of tree probabilities.
        """
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        for i, tree_name in enumerate(tree_names):
            tree = tree_dict[tree_name]
            wts = tree_wts[i]

            self.ccd_dict_update(tree, wts)

    def bn_train_count(self, tree_count, tree_id):
        """Extracts the conditional subsplit distributions (CSDs) from sample trees and stores them in the SBN object.

        :param tree_count: dictionary mapping tree topology ID to count
        of that tree in the sample.
        :param tree_id: dictionary mapping tree topology ID to a
        singleton list containing the tree object.
        """
        # Clear the SBN model dictionaries
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        self.clade_double_bipart_dict = defaultdict(lambda: defaultdict(float))
        total_count = sum(tree_count.values()) * 1.0

        # Iterate over the trees and update the model dictionaries, weighted by tree count.
        for key in tree_count:
            count = tree_count[key]
            tree = tree_id[key][0]

            wts = count / total_count
            # Note that by calling bn_dict_update we call clade_update and ccd_dict_update, which update clade_dict,
            # clade_bipart_dict, and clade_double_bipart_dict.
            self.bn_dict_update(tree, wts)
            self.samp_tree_freq[tree.get_topology_id()] = wts

        # We stash these count-based dictionaries for future use. The pattern in bn_em_count is to call this function
        # once and then repeatedly do EM steps using bn_dict_em_update; thus they don't get overwritten each time.
        self.clade_freq_est = deepcopy(self.clade_dict)
        self.clade_bipart_freq_est = deepcopy(self.clade_bipart_dict)
        for key in self.clade_double_bipart_dict:
            self.clade_double_bipart_len[key] = len(self.clade_double_bipart_dict[key])

    def bn_train_prob(self, tree_dict, tree_names, tree_wts):
        """Extracts the conditional subsplit distributions (CSDs) from tree probabilities and stores them in the SBN object.

        :param tree_dict: dictionary mapping tree topology ID to count
        of that tree in the sample.
        :param tree_names: list of tree topology IDs.
        :param tree_wts: list of tree probabilities.
        """
        # Clear the SBN model dictionaries
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        self.clade_double_bipart_dict = defaultdict(lambda: defaultdict(float))
        for i, tree_name in enumerate(tree_names):
            tree = tree_dict[tree_name]
            wts = tree_wts[i]

            self.bn_dict_update(tree, wts)
            self.samp_tree_freq[tree.get_topology_id()] = wts

        self.clade_freq_est = deepcopy(self.clade_dict)
        self.clade_bipart_freq_est = deepcopy(self.clade_bipart_dict)
        for key in self.clade_double_bipart_dict:
            self.clade_double_bipart_len[key] = len(self.clade_double_bipart_dict[key])

    def bn_em_root_prob(self, tree, bipart_bitarr_prob, nodetobitMap):
        """Calculate the rooting probability distribution.

        :param tree: Tree (ete3) topology and edge lengths.
        :param bipart_bitarr_prob: Dictionary mapping each node (via to01())
        to the SBN likelihood of the tree joint with rooting at that node.
        :param nodetobitMap: Cached dictionary mapping each node to
        the bitarray representation of its descendant leaves.
        :return: tuple (root_prob, cum_root_prob, normalizing_const),
        where root_prob is a dictionary mapping each node to
        the conditional subsplit distribution (CSD) of rooting at that node
        given the unrooted tree,
        cum_root_prob is a dictionary mapping each node to
        the conditional subsplit distribution (CSD) of rooting at _or below_
        that node given the unrooted tree,
        and normalizing_const is the probability/likelihood of the
        unrooted tree.
        """
        root_prob = {}
        bipart_bitarr_up = {}
        cum_root_prob = defaultdict(float)
        # NB, Abuse of comment notation: dict[node] shorthand for dict[bipart_bitarr.to01()] where bipart_bitarr is the
        # bitarray signature for node.
        for node in tree.traverse('postorder'):
            if not node.is_root():
                bipart_bitarr = self._minor_bitarr(nodetobitMap[node])
                bipart_bitarr_up[node] = bipart_bitarr
                # This line initializes cum_root_prob[node] to Pr(root @ node, T^u)
                cum_root_prob[bipart_bitarr.to01()] += bipart_bitarr_prob[bipart_bitarr.to01()]
                if not node.is_leaf():
                    # This loop adds Pr(root below node, T^u) to cum_root_prob[node] resulting in cum_root_prob[node]
                    # containing Pr(root @ or below node, T^u)
                    for child in node.children:
                        cum_root_prob[bipart_bitarr.to01()] += cum_root_prob[bipart_bitarr_up[child].to01()]

        # normalizing_const will contain the unrooted tree likelihood,
        # as the sum of the joint distribution over all tree rootings.
        normalizing_const = 0.0
        for child in tree.children:
            normalizing_const += cum_root_prob[bipart_bitarr_up[child].to01()]

        for key in bipart_bitarr_prob:
            # This line fills root_prob[node] with Pr(root @ node | T^u)
            root_prob[key] = bipart_bitarr_prob[key] / normalizing_const
            # This line converts cum_root_prob[node] from
            # Pr(root @ or below node, T^u) to Pr(root @ or below node | T^u)
            cum_root_prob[key] /= normalizing_const

        return root_prob, cum_root_prob, normalizing_const

    def bn_dict_em_update(self, tree, wts, bipart_bitarr_prob, clade_dict, clade_bipart_dict, clade_double_bipart_dict):
        """EM-algorithm update step for one tree.

        :param tree: Tree (ete3) containing topology and edge lengths.
        :param wts: Weight of tree in the tree sample or probability distribution.
        :param bipart_bitarr_prob: Dictionary mapping each node (via to01())
        to the likelihood of the tree joint with rooting at that node.
        :param clade_dict: Dictionary mapping each node
        to the probability of observing that root split.
        :param clade_bipart_dict: Dictionary mapping each node
        to the probability of observing that non-root split.
        :param clade_double_bipart_dict: Dictionary mapping each node
        to the probability of observing that subsplit, specifically the
        joint probability of the parent split and child split.
        :return: The likelihood of the unrooted tree.
        """
        nodetobitMap = {node: self.clade_to_bitarr(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        root_prob, cum_root_prob, est_prob = self.bn_em_root_prob(tree, bipart_bitarr_prob, nodetobitMap)

        for node in tree.traverse('levelorder'):
            if not node.is_root():
                bipart_bitarr = self._minor_bitarr(nodetobitMap[node])
                node_wts = wts * root_prob[bipart_bitarr.to01()]
                # Update clade_dict adding wts * Pr(root @ node | T^u)
                clade_dict[bipart_bitarr.to01()] += node_wts

                if not node.is_leaf():
                    # Orientation 1/2
                    # Root node is 'above' node, so node's subsplit splits its 'child' clades.
                    children_bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    cum_node_wts = wts * (1.0 - cum_root_prob[bipart_bitarr.to01()] + root_prob[bipart_bitarr.to01()])
                    # Update clade_bipart_dict (Orientation 1/2) adding wts * Pr(root @ or above node | T^u)
                    clade_bipart_dict[nodetobitMap[node].to01()][children_bipart_bitarr.to01()] += cum_node_wts
                    if not node.up.is_root():
                        parent_bipart_bitarr = self._minor_bitarr(nodetobitMap[node.up])
                        cum_node_wts = wts * (1.0 - cum_root_prob[parent_bipart_bitarr.to01()] + root_prob[parent_bipart_bitarr.to01()])
                        # Orientation 1/6
                        # See the `orientations` image in the `doc/` directory to see a figure depicting these orientations.
                        # If virtual root is beyond node.up, this well-defines the parent subsplit at node.up.
                        comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]
                        # Update clade_double_bipart_dict (Orientation 1/6) adding wts * Pr(root @ or above parent node | T^u)
                        clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts

                        cum_node_wts = wts * cum_root_prob[self._minor_bitarr(nodetobitMap[node.get_sisters()[0]]).to01()]
                        # Orientation 2/6
                        # If virtual root is linked via a sister, this well-defines the parent subsplit at node.up.
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[node]
                        # Update clade_double_bipart_dict (Orientation 2/6) adding wts * Pr(root @ or below sister node | T^u)
                        clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts
                    else:
                        for sister in node.get_sisters():
                            # Here ^ is bitwise XOR.
                            cum_node_wts = wts * cum_root_prob[self._minor_bitarr(nodetobitMap[sister] ^ (~nodetobitMap[node])).to01()]
                            # Orientation 2/6
                            # If virtual root is linked via a sister, this well-defines the parent subsplit at node.up.
                            comb_parent_bipart_bitarr = nodetobitMap[sister] + nodetobitMap[node]
                            # Update clade_double_bipart_dict (Orientation 2/6 again) adding wts * Pr(root @ or below sister node | T^u)
                            clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts

                    # Orientation 3/6
                    # If virtual root is on the edge between node and node.up, this well-defines the root split.
                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    # Update clade_double_bipart_dict (Orientation 3/6) adding wts * Pr(root @ node | T^u)
                    clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += node_wts

                # Orientation 2/2
                # Root node is 'below' node, so node's subsplit splits its 'sister' and/or 'parent' clades.
                # NB: tree topology is unrooted, but is stored in a rooted tree format (with a trifurcating root).
                if not node.up.is_root():
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])

                cum_node_wts = wts * cum_root_prob[bipart_bitarr.to01()]
                # Update clade_bipart_dict (Orientation 2/2) adding wts * Pr(root @ or below node | T^u)
                clade_bipart_dict[(~nodetobitMap[node]).to01()][children_bipart_bitarr.to01()] += cum_node_wts
                if not node.is_leaf():
                    for child in node.children:
                        cum_node_wts = wts * cum_root_prob[self._minor_bitarr(nodetobitMap[node] ^ nodetobitMap[child]).to01()]
                        # Orientations 4/6 and 5/6
                        # If virtual root is beyond node's children, this well-defines the parent subsplit at node.
                        comb_parent_bipart_bitarr = nodetobitMap[child] + ~nodetobitMap[node]
                        # Update clade_double_bipart_dict (Orientations 4/6 and 5/6) adding wts * Pr(root @ or below child node | T^u)
                        clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts

                # Orientation 6/6
                # If virtual root is on the edge between node.up and node, this well-defines the root split.
                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                # Update clade_double_bipart_dict (Orientation 6/6) adding wts * Pr(root @ node | T^u)
                clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += node_wts

        # Returns Pr(T^u)
        return est_prob

    def bn_em_prob(self, tree_dict, tree_names, tree_wts, maxiter=100, miniter=50, abstol=1e-04, monitor=False, MAP=False):
        """Run EM-algorithm on a set of unrooted tree probability data.

        :param tree_dict: Dictionary where keys are tree topology IDs
        and the values are integers representing how many times that
        tree appeared in the sample data.
        :param tree_names: Collection of tree topology IDs.
        :param tree_wts: List of tree probabilities.
        :param maxiter: The maximum number of EM iterations (default=100).
        :param miniter: The minimum number of EM iterations (default=50).
        :param abstol: The stepwise likelihood difference below which
        the algorithm will consider itself converged and try to
        terminate, if at lease miniter steps have passed (default=1e-4).
        :param monitor: Boolean, whether to print iteration diagnostic
        (default=False).
        :param MAP: Boolean, whether to use regularization
        (default=False).
        :return: The log-likelihood.
        """
        self.bn_train_prob(tree_dict, tree_names, tree_wts)
        logp = []

        for k in range(maxiter):
            curr_logp = 0.0
            clade_dict = defaultdict(float)
            clade_bipart_dict = defaultdict(lambda: defaultdict(float))
            clade_double_bipart_dict = defaultdict(lambda: defaultdict(float))
            for i, tree_name in enumerate(tree_names):
                tree = tree_dict[tree_name]
                wts = tree_wts[i]
                # E-step
                bipart_bitarr_prob = self._bn_estimate_fast(tree, MAP)
                # Update the weighted frequency counts.
                est_prob = self.bn_dict_em_update(tree, wts, bipart_bitarr_prob, clade_dict, clade_bipart_dict, clade_double_bipart_dict)
                curr_logp += wts * np.log(est_prob)

            if MAP:
                curr_logp += self.logprior()
            logp.append(curr_logp)
            if monitor:
                print "Iter {}: current per tree log-likelihood {:.06f}".format(k + 1, curr_logp)

            self.set_clade_bipart(clade_dict, clade_bipart_dict, clade_double_bipart_dict)

            if k > miniter and abs(logp[-1] - logp[-2]) < abstol:
                break

        return logp

    def bn_em_count(self, tree_count, tree_id, maxiter=100, miniter=50, abstol=1e-04, monitor=False, MAP=False):
        """Run EM-algorithm on a set of unrooted tree count data.

        :param tree_count: Dictionary where keys are tree topology IDs
        and the values are integers representing how many times that
        tree appeared in the sample data.
        :param tree_id: Dictionary where the keys are tree topology IDs
        and the values are singleton lists containing the tree
        corresponding to the topology ID.
        :param maxiter: The maximum number of EM iterations (default=100).
        :param miniter: The minimum number of EM iterations (default=50).
        :param abstol: The stepwise likelihood difference below which
        the algorithm will consider itself converged and try to
        terminate, if at lease miniter steps have passed (default=1e-4).
        :param monitor: Boolean, whether to print iteration diagnostic
        (default=False).
        :param MAP: Boolean, whether to use regularization
        (default=False).
        :return: The log-likelihood.
        """
        self.bn_train_count(tree_count, tree_id)
        total_count = sum(tree_count.values()) * 1.0
        logp = []

        for k in range(maxiter):
            curr_logp = 0.0
            clade_dict = defaultdict(float)
            clade_bipart_dict = defaultdict(lambda: defaultdict(float))
            clade_double_bipart_dict = defaultdict(lambda: defaultdict(float))
            for key in tree_count:
                count = tree_count[key]
                tree = tree_id[key][0]

                # E-step
                bipart_bitarr_prob = self._bn_estimate_fast(tree, MAP)
                # update the weighted frequency counts
                est_prob = self.bn_dict_em_update(tree, count / total_count, bipart_bitarr_prob, clade_dict, clade_bipart_dict,
                                                  clade_double_bipart_dict)
                curr_logp += count / total_count * np.log(est_prob)

            if MAP:
                curr_logp += self.logprior()
            logp.append(curr_logp)
            if monitor:
                print "Iter: {}: current per tree log-likelihood {:.06f}".format(k + 1, curr_logp)

            self.set_clade_bipart(clade_dict, clade_bipart_dict, clade_double_bipart_dict)

            if k > miniter and abs(logp[-1] - logp[-2]) < abstol:
                break

        return logp

    def get_clade_bipart(self):
        """Gets a copy of clade dictionaries.

        This is not symmetric with set_clade_bipart.

        :return: tuple containing a dictionary of root split
        probabilities and a dictionary of dictionaries containing clade
        subsplit probabilities.
        """
        return deepcopy(self.clade_dict), deepcopy(self.clade_bipart_dict)

    def set_clade_bipart(self, clade_dict, clade_bipart_dict, clade_double_bipart_dict):
        """Sets clade dictionaries.

        :param clade_dict: dictionary of root split probabilities.
        :param clade_bipart_dict: dictionary of dictionaries
        containing clade subsplit probabilities.
        :param clade_double_bipart_dict: dictionary of dictionaries
        containing subsplit probabilities (technically joint parent
        split, child split probabilities).
        """
        self.clade_dict, self.clade_bipart_dict, self.clade_double_bipart_dict = deepcopy(clade_dict), deepcopy(clade_bipart_dict), deepcopy(
            clade_double_bipart_dict)

    def ccd_estimate(self, tree, unrooted=True):
        """Calculate tree (rooted or unrooted) likelihood using clade conditional distibutions.

        :param tree: Tree (ete3) containing topology and edge lengths.
        :param unrooted: boolean (default False) whether tree is unrooted or not.
        :return: tree likelihood.
        """
        ccd_est = 1.0
        nodetobitMap = {}
        # One-pass Bayesian network probability calculation
        for node in tree.traverse('levelorder'):
            if node.is_root():
                if unrooted:
                    child_1 = node.children[0]
                    child_1_bitarr = self.clade_to_bitarr(child_1.get_leaf_names())
                    nodetobitMap[child_1] = child_1_bitarr

                    for sister in child_1.get_sisters():
                        sister_bitarr = self.clade_to_bitarr(sister.get_leaf_names())
                        nodetobitMap[sister] = sister_bitarr

                    bipart_bitarr = min([nodetobitMap[sister] for sister in child_1.get_sisters()])
                    ccd_est *= self.clade_bipart_dict[(~child_1_bitarr).to01()][bipart_bitarr.to01()]
                else:
                    for child in node.children:
                        child_bitarr = self.clade_to_bitarr(child.get_leaf_names())
                        nodetobitMap[child] = child_bitarr
                    bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    ccd_est *= self.clade_dict[bipart_bitarr.to01()]
            elif not node.is_leaf():
                for child in node.children:
                    child_bitarr = self.clade_to_bitarr(child.get_leaf_names())
                    nodetobitMap[child] = child_bitarr

                bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                # Notes for log-conversion:
                # np.exp(-745) == 0.0 -> False
                # np.exp(-746) == 0.0 -> True
                # Consider moving this check to the end of the loop body.
                if ccd_est == 0.0: break
                ccd_est *= self.clade_bipart_dict[nodetobitMap[node].to01()][bipart_bitarr.to01()] / self.clade_dict[self._minor_bitarr(
                    nodetobitMap[node]).to01()]

        return (2 * self.ntaxa - 3.0) * ccd_est

    def bn_estimate_rooted(self, tree, MAP=False):
        """Calculate rooted tree likelihood using subsplit distributions.

        :param tree: Tree (ete3) containing tree topology.
        :param MAP: boolean (default False) whether to regularize or not.
        :return: tree likelihood.
        """
        bn_est = 1.0
        nodetobitMap = {}
        # One-pass Bayesian network probability calculation
        for node in tree.traverse('levelorder'):
            if node.is_root():
                for child in node.children:
                    child_bitarr = self.clade_to_bitarr(child.get_leaf_names())
                    nodetobitMap[child] = child_bitarr
                normalizing_const_est = self.clade_freq_est[self._minor_bitarr(child_bitarr).to01()]
                # Root split probability with optional regularization
                bn_est *= (self.clade_dict[self._minor_bitarr(child_bitarr).to01()] + MAP * self.alpha * normalizing_const_est) / (
                    1.0 + MAP * self.alpha)
            elif not node.is_leaf():
                for child in node.children:
                    child_bitarr = self.clade_to_bitarr(child.get_leaf_names())
                    nodetobitMap[child] = child_bitarr

                bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                if bn_est == 0.0:
                    break
                parent_bipart_bitarr = min([nodetobitMap[node.get_sisters()[0]], nodetobitMap[node]])
                comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]
                if node.up.is_root():
                    normalizing_const = self.clade_dict[self._minor_bitarr(nodetobitMap[node]).to01()]
                    normalizing_const_est = self.clade_freq_est[self._minor_bitarr(nodetobitMap[node]).to01()]
                else:
                    normalizing_const = self.clade_bipart_dict[nodetobitMap[node.up].to01()][parent_bipart_bitarr.to01()]
                    normalizing_const_est = self.clade_bipart_freq_est[nodetobitMap[node.up].to01()][parent_bipart_bitarr.to01()]

                if (normalizing_const + MAP * self.alpha * normalizing_const_est) == 0.0:
                    bn_est = 0.0
                    break

                if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] > 0:
                    # Subsplit conditional probability with optional regularization
                    bn_est *= (self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] +
                               MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                   normalizing_const + MAP * self.alpha * normalizing_const_est)
                else:
                    bn_est = 0.0
                    break

        return bn_est

    def _bn_estimate_fast(self, tree, MAP=False):
        """Two-pass algorithm for calculating rooted SBN likelihoods for all
        rootings of a given unrooted tree.

        :param tree: Tree (ete3) containing tree topology.
        :param MAP: boolean (default False) whether to regularize or not.
        :return: dictionary mapping nodes' clade-01 representation to
        the joint Pr(root @ node, T^u)
        """

        # cbn_est_up[node] contains the probability of the node subsplit and all descendant subsplits,
        # given the node's parent subsplit.
        cbn_est_up = {node: 1.0 for node in tree.traverse('postorder') if not node.is_root()}

        # Up[node] contains the probability of all descendant subsplits, given the node subsplit.
        # Calling this Up is motivated by the message-passing algorithm, such that Up is the aggregation of all of the
        # messages going up the tree. We need an up and a down pass so that we can calculate the rooted SBN likelihoods
        # for every rooting of the tree.
        Up = {node: 1.0 for node in tree.traverse('postorder') if not node.is_root()}

        nodetobitMap = {node: self.clade_to_bitarr(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        bipart_bitarr_up = {}

        # bipart_bitarr_prob[node] contains the joint probability of the tree and the rooting above node.
        bipart_bitarr_prob = {}

        # Upward (leaf-to-root) pass
        # Tree likelihood should be available at the end of this pass, and cbn_est_up and Up will be filled out.
        for node in tree.traverse('postorder'):
            if not node.is_leaf() and not node.is_root():
                # Collecting the child conditional probabilities.
                # Up[node] is complete now.
                for child in node.children:
                    Up[node] *= cbn_est_up[child]

                bipart_bitarr = min(nodetobitMap[child] for child in node.children)
                bipart_bitarr_up[node] = bipart_bitarr
                if not node.up.is_root():
                    # cbn_est_up[node] is a product of Up[node] and the conditional probability of the node subsplit,
                    # given the parent subsplit.
                    # %EM Couldn't this be an = rather than *=?
                    cbn_est_up[node] *= Up[node]
                    parent_bipart_bitarr = min([nodetobitMap[node.get_sisters()[0]], nodetobitMap[node]])
                    comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]

                    # normalizing_const holds the probability of the parent subsplit, and so by dividing by it turns the
                    # joint probability of parent and child into child conditional on parent.
                    normalizing_const = self.clade_bipart_dict[nodetobitMap[node.up].to01()][parent_bipart_bitarr.to01()]
                    normalizing_const_est = self.clade_bipart_freq_est[nodetobitMap[node.up].to01()][parent_bipart_bitarr.to01()]
                    if (normalizing_const + MAP * self.alpha * normalizing_const_est
                        ) == 0.0 or self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] == 0.0:
                        cbn_est_up[node] = 0.0
                    else:
                        # Here cbn_est_up[node] is complete, multiplying by the conditional probability
                        # of the node subsplit, given the parent subsplit.
                        cbn_est_up[node] *= (self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] +
                                             MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]
                                             ) / (normalizing_const + MAP * self.alpha * normalizing_const_est)

        # Downward (root-to-leaf) pass
        # cbn_est_down[node] contains the conditional probability of all subsplits above the parent, given the parent
        # subsplit.
        cbn_est_down = {node: 1.0 for node in tree.traverse('preorder') if not node.is_root()}
        bipart_bitarr_down = {}
        for node in tree.traverse('preorder'):
            if not node.is_root():
                if node.up.is_root():
                    parent_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                    bipart_bitarr_down[node] = parent_bipart_bitarr

                    normalizing_const = self.clade_bipart_dict[(~nodetobitMap[node]).to01()][parent_bipart_bitarr.to01()]
                    normalizing_const_est = self.clade_bipart_freq_est[(~nodetobitMap[node]).to01()][parent_bipart_bitarr.to01()]

                    if (normalizing_const + MAP * self.alpha * normalizing_const_est) == 0.0:
                        cbn_est_down[node] = 0.0
                    else:
                        for sister in node.get_sisters():
                            if not sister.is_leaf():
                                # For each sister node, cbn_est_down[node] factors in the contribution from Up[sister]:
                                # the probability of all sister-descendant subsplits, given the sister subsplit.
                                cbn_est_down[node] *= Up[sister]
                                bipart_bitarr = min(nodetobitMap[child] for child in sister.children)
                                comb_parent_bipart_bitarr = ((~nodetobitMap[node]) ^ nodetobitMap[sister]) + nodetobitMap[sister]
                                if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] > 0.0:
                                    # Multiply by the conditional probability of the node subsplit, given the parent
                                    # subsplit.
                                    cbn_est_down[node] *= (
                                        self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] +
                                        MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                            normalizing_const + MAP * self.alpha * normalizing_const_est)
                                else:
                                    cbn_est_down[node] = 0.0
                else:
                    sister = node.get_sisters()[0]
                    parent_bipart_bitarr = min([nodetobitMap[sister], ~nodetobitMap[node.up]])
                    bipart_bitarr_down[node] = parent_bipart_bitarr

                    normalizing_const = self.clade_bipart_dict[(~nodetobitMap[node]).to01()][parent_bipart_bitarr.to01()]
                    normalizing_const_est = self.clade_bipart_freq_est[(~nodetobitMap[node]).to01()][parent_bipart_bitarr.to01()]

                    if (normalizing_const + MAP * self.alpha * normalizing_const_est) == 0.0:
                        cbn_est_down[node] = 0.0
                    else:
                        # cbn_est_down[node] factors in the contribution from cbn_est_down[node.up]: the conditional
                        # probability of all subsplits above the parent's parent, given the parent's parent subsplit.
                        cbn_est_down[node] *= cbn_est_down[node.up]
                        comb_parent_bipart_bitarr = nodetobitMap[sister] + ~nodetobitMap[node.up]
                        if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node.up].to01()] > 0.0:
                            # Multiply by the conditional probability of the parent's parent subsplit, given the parent
                            # subsplit.
                            cbn_est_down[node] *= (
                                self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node.up].to01()] +
                                MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                    normalizing_const + MAP * self.alpha * normalizing_const_est)
                        else:
                            cbn_est_down[node] = 0.0

                        if not sister.is_leaf():
                            # Multiply by the probability of all sister-descendant subsplits, given the sister subsplit.
                            cbn_est_down[node] *= Up[sister]
                            comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[sister]
                            if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[sister].to01()] > 0.0:
                                # Multiply by by the conditional probability of the sister subsplit,
                                # given the parent subsplit.
                                cbn_est_down[node] *= (self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[sister].to01(
                                )] + MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                    normalizing_const + MAP * self.alpha * normalizing_const_est)
                            else:
                                cbn_est_down[node] = 0.0

                parent_bipart_bitarr = self._minor_bitarr(nodetobitMap[node])
                normalizing_const = self.clade_dict[parent_bipart_bitarr.to01()]
                normalizing_const_est = self.clade_freq_est[parent_bipart_bitarr.to01()]
                # bipart_bitarr_prob[node] (abuse of notation) factors in the probability of the node-vs-parent split
                # as a root split.
                bipart_bitarr_prob[parent_bipart_bitarr.to01()] = (
                    self.clade_dict[parent_bipart_bitarr.to01()] + MAP * self.alpha * normalizing_const_est) / (1.0 + MAP * self.alpha)

                if (normalizing_const + MAP * self.alpha * normalizing_const_est) == 0.0:
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0
                else:
                    if not node.is_leaf():
                        # Multiply by the probability of all descendant subsplits, given the node subsplit.
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= Up[node]
                        comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                        if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[node].to01()] > 0.0:
                            # Multiply by the probability of the node subsplit given the rooting.
                            bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= (
                                self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[node].to01()] +
                                MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                    normalizing_const + MAP * self.alpha * normalizing_const_est)
                        else:
                            bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0

                    # Multiply by the conditional probability of all subsplits above the parent,
                    # given the parent subsplit.
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= cbn_est_down[node]
                    comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                    if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node].to01()] > 0.0:
                        # Multiply by the probability of the parent subsplit given the rooting.
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= (
                            self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node].to01()] +
                            MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                normalizing_const + MAP * self.alpha * normalizing_const_est)
                    else:
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0

        # bipart_bitarr_prob is complete now.
        # If you sum over all values in the dict, you get the likelihood of the unrooted tree.
        return bipart_bitarr_prob

    def bn_estimate(self, tree, MAP=False):
        """Summarizes the joint Pr(root @ node, T^u) into Pr(T^u).

        :param tree: Tree (ete3) containing tree topology.
        :param MAP: boolean (default False) whether to regularize or not.
        :return: the unrooted tree likelihood Pr(T^u).
        """
        return np.sum(self._bn_estimate_fast(tree, MAP).values())

    def kl_div(self, method='all', MAP=False):
        """Calculates the KL-Divergence of the different SBN backends relative to truth.

        :param method: string denoting which method to train the SBN.
        Options are 'ccd': conditional clade distribution;
        'bn': conditional subsplit distributions, i.e. child subsplit conditional probabilities given parent subsplit;
        'freq': empirical distribution;
        'all': all of the above.
        :param MAP: boolean (default False) whether to regularize or not.
        :return: dictionary mapping method to value of KL-divergence.
        """
        kl_div = defaultdict(float)
        for tree, wts in self.emp_tree_freq.iteritems():
            if method in ['ccd', 'all']:
                kl_div['ccd'] += wts * np.log(max(self.ccd_estimate(tree), EPS))
            if method in ['bn', 'all']:
                kl_div['bn'] += wts * np.log(max(self.bn_estimate(tree, MAP), EPS))
            if method in ['freq', 'all']:
                kl_div['freq'] += wts * np.log(max(self.samp_tree_freq[tree.get_topology_id()], EPS))

        for key in kl_div:
            kl_div[key] = self.negDataEnt - kl_div[key]

        return kl_div
