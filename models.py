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
        self.emp_tree_freq = emp_tree_freq
        self.negDataEnt = np.sum([wts * np.log(wts + EPS) for wts in emp_tree_freq.values()])
        self.samp_tree_freq = defaultdict(float)
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        self.clade_double_bipart_dict = defaultdict(lambda: defaultdict(float))

        self.clade_freq_est = defaultdict(float)
        self.clade_bipart_freq_est = defaultdict(lambda: defaultdict(float))
        self.clade_double_bipart_len = defaultdict(int)

    def _combine_bitarr(self, arrA, arrB):
        if arrA < arrB:
            return arrA + arrB
        else:
            return arrB + arrA

    def _merge_bitarr(self, key):
        return bitarray(key[:self.ntaxa]) | bitarray(key[self.ntaxa:])

    def _decomp_minor_bitarr(self, key):
        return min(bitarray(key[:self.ntaxa]), bitarray(key[self.ntaxa:]))

    def _minor_bitarr(self, arrA):
        return min(arrA, ~arrA)

    def clade_to_bitarr(self, clade):
        bit_list = ['0'] * self.ntaxa
        for i in clade:
            bit_list[self.map[i]] = '1'
        return bitarray(''.join(bit_list))

    def check_clade_dict(self):
        print "clade_dict sum: {:.12f}".format(sum(self.clade_dict.values()))
        print "clade_bipart_dict tabular sum:"
        for key in self.clade_bipart_dict:
            bipart_bitarr = self._minor_bitarr(bitarray(key))
            print '{}:{:.12f}|{:.12f}'.format(bipart_bitarr.to01(), sum(self.clade_bipart_dict[key].values()), self.clade_dict[bipart_bitarr.to01()])

    # check the validity of weighted frequency tables used in EM
    def check_clade_dict_em(self):
        print "clade_dict sum: {:.12f}".format(sum(self.clade_dict.values()))
        print "clade_double_bipart_dict tabular sum:"
        for key in self.clade_double_bipart_dict:
            parent_clade_bitarr = self._merge_bitarr(key)
            bipart_bitarr = self._decomp_minor_bitarr(key)
            if parent_clade_bitarr.count() != self.ntaxa:
                print '{}|{}:{:.12f}|{:.12f}'.format(parent_clade_bitarr.to01(), bipart_bitarr.to01(), sum(
                    self.clade_double_bipart_dict[key].values()), self.clade_bipart_dict[parent_clade_bitarr.to01()][bipart_bitarr.to01()])
            else:
                print '{}|{}:{:.12f}|{:.12f}'.format(parent_clade_bitarr.to01(), bipart_bitarr.to01(),
                                                     sum(self.clade_double_bipart_dict[key].values()), self.clade_dict[bipart_bitarr.to01()])

    # Dirichlet conjugate prior
    def logprior(self):
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
        nodetobitMap = {}
        for node in tree.traverse('levelorder'):
            if not node.is_root():
                clade = node.get_leaf_names()
                node_bitarr = self.clade_to_bitarr(clade)
                self.clade_dict[self._minor_bitarr(node_bitarr).to01()] += wts / (2 * self.ntaxa - 3.0)
                nodetobitMap[node] = node_bitarr

        return nodetobitMap

    def ccd_dict_update(self, tree, wts):
        nodetobitMap = self.clade_update(tree, wts)
        for node in tree.traverse('levelorder'):
            if not node.is_root():
                if not node.is_leaf():
                    bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    self.clade_bipart_dict[nodetobitMap[node].to01()][bipart_bitarr.to01()] += wts / (2 * self.ntaxa - 3.0)
                if not node.up.is_root():
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])

                self.clade_bipart_dict[(~nodetobitMap[node]).to01()][bipart_bitarr.to01()] += wts / (2 * self.ntaxa - 3.0)

        return nodetobitMap

    def bn_dict_update(self, tree, wts, root_wts=None):
        nodetobitMap = self.ccd_dict_update(tree, wts)
        for node in tree.traverse('levelorder'):
            if not root_wts:
                node_wts = wts / (2 * self.ntaxa - 3.0)
            else:
                node_wts = wts * root_wts[nodetobitMap[node].to01()]

            if not node.is_root():
                if not node.is_leaf():
                    bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    for sister in node.get_sisters():
                        comb_parent_bipart_bitarr = nodetobitMap[sister] + nodetobitMap[node]
                        self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts
                    if not node.up.is_root():
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[node]
                        self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

                if not node.up.is_root():
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                if not node.is_leaf():
                    for child in node.children:
                        comb_parent_bipart_bitarr = nodetobitMap[child] + ~nodetobitMap[node]
                        self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] += node_wts

    def ccd_train_count(self, tree_count, tree_id):
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        total_count = sum(tree_count.values()) * 1.0
        for key in tree_count:
            count = tree_count[key]
            tree = tree_id[key][0]

            wts = count / total_count
            self.ccd_dict_update(tree, wts)

    def ccd_train_prob(self, tree_dict, tree_names, tree_wts):
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        for i, tree_name in enumerate(tree_names):
            tree = tree_dict[tree_name]
            wts = tree_wts[i]

            self.ccd_dict_update(tree, wts)

    def bn_train_count(self, tree_count, tree_id):
        self.clade_dict = defaultdict(float)
        self.clade_bipart_dict = defaultdict(lambda: defaultdict(float))
        self.clade_double_bipart_dict = defaultdict(lambda: defaultdict(float))
        total_count = sum(tree_count.values()) * 1.0
        for key in tree_count:
            count = tree_count[key]
            tree = tree_id[key][0]

            wts = count / total_count
            self.bn_dict_update(tree, wts)
            self.samp_tree_freq[tree.get_topology_id()] = wts

        self.clade_freq_est = deepcopy(self.clade_dict)
        self.clade_bipart_freq_est = deepcopy(self.clade_bipart_dict)
        for key in self.clade_double_bipart_dict:
            self.clade_double_bipart_len[key] = len(self.clade_double_bipart_dict[key])

    def bn_train_prob(self, tree_dict, tree_names, tree_wts):
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
        root_prob = {}
        bipart_bitarr_up = {}
        cum_root_prob = defaultdict(float)
        for node in tree.traverse('postorder'):
            if not node.is_root():
                bipart_bitarr = self._minor_bitarr(nodetobitMap[node])
                bipart_bitarr_up[node] = bipart_bitarr
                cum_root_prob[bipart_bitarr.to01()] += bipart_bitarr_prob[bipart_bitarr.to01()]
                if not node.is_leaf():
                    for child in node.children:
                        cum_root_prob[bipart_bitarr.to01()] += cum_root_prob[bipart_bitarr_up[child].to01()]

        normalizing_const = 0.0
        for child in tree.children:
            normalizing_const += cum_root_prob[bipart_bitarr_up[child].to01()]

        for key in bipart_bitarr_prob:
            root_prob[key] = bipart_bitarr_prob[key] / normalizing_const
            cum_root_prob[key] /= normalizing_const

        return root_prob, cum_root_prob, normalizing_const

    def bn_dict_em_update(self, tree, wts, bipart_bitarr_prob, clade_dict, clade_bipart_dict, clade_double_bipart_dict):
        nodetobitMap = {node: self.clade_to_bitarr(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        root_prob, cum_root_prob, est_prob = self.bn_em_root_prob(tree, bipart_bitarr_prob, nodetobitMap)

        for node in tree.traverse('levelorder'):
            if not node.is_root():
                bipart_bitarr = self._minor_bitarr(nodetobitMap[node])
                node_wts = wts * root_prob[bipart_bitarr.to01()]
                clade_dict[bipart_bitarr.to01()] += node_wts

                if not node.is_leaf():
                    children_bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    cum_node_wts = wts * (1.0 - cum_root_prob[bipart_bitarr.to01()] + root_prob[bipart_bitarr.to01()])
                    clade_bipart_dict[nodetobitMap[node].to01()][children_bipart_bitarr.to01()] += cum_node_wts
                    if not node.up.is_root():
                        parent_bipart_bitarr = self._minor_bitarr(nodetobitMap[node.up])
                        cum_node_wts = wts * (1.0 - cum_root_prob[parent_bipart_bitarr.to01()] + root_prob[parent_bipart_bitarr.to01()])
                        comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]
                        clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts

                        cum_node_wts = wts * cum_root_prob[self._minor_bitarr(nodetobitMap[node.get_sisters()[0]]).to01()]
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[node]
                        clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts
                    else:
                        for sister in node.get_sisters():
                            cum_node_wts = wts * cum_root_prob[self._minor_bitarr(nodetobitMap[sister] ^ (~nodetobitMap[node])).to01()]
                            comb_parent_bipart_bitarr = nodetobitMap[sister] + nodetobitMap[node]
                            clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts

                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += node_wts

                if not node.up.is_root():
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])

                cum_node_wts = wts * cum_root_prob[bipart_bitarr.to01()]
                clade_bipart_dict[(~nodetobitMap[node]).to01()][children_bipart_bitarr.to01()] += cum_node_wts
                if not node.is_leaf():
                    for child in node.children:
                        cum_node_wts = wts * cum_root_prob[self._minor_bitarr(nodetobitMap[node] ^ nodetobitMap[child]).to01()]
                        comb_parent_bipart_bitarr = nodetobitMap[child] + ~nodetobitMap[node]
                        clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += cum_node_wts

                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][children_bipart_bitarr.to01()] += node_wts

        return est_prob

    def bn_em_prob(self, tree_dict, tree_names, tree_wts, maxiter=100, miniter=50, abstol=1e-04, monitor=False, MAP=False):
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
                # update the weigted frequency counts
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
        return deepcopy(self.clade_dict), deepcopy(self.clade_bipart_dict)

    def set_clade_bipart(self, clade_dict, clade_bipart_dict, clade_double_bipart_dict):
        self.clade_dict, self.clade_bipart_dict, self.clade_double_bipart_dict = deepcopy(clade_dict), deepcopy(clade_bipart_dict), deepcopy(
            clade_double_bipart_dict)

    def ccd_estimate(self, tree, unrooted=True):
        ccd_est = 1.0
        nodetobitMap = {}
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
                    ccd_est *= self.clade_dict[bitpart_bitarr.to01()]
            elif not node.is_leaf():
                for child in node.children:
                    child_bitarr = self.clade_to_bitarr(child.get_leaf_names())
                    nodetobitMap[child] = child_bitarr

                bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                if ccd_est == 0.0: break
                ccd_est *= self.clade_bipart_dict[nodetobitMap[node].to01()][bipart_bitarr.to01()] / self.clade_dict[self._minor_bitarr(
                    nodetobitMap[node]).to01()]

        return (2 * self.ntaxa - 3.0) * ccd_est

    def bn_estimate_rooted(self, tree, MAP=False):
        bn_est = 1.0
        nodetobitMap = {}
        for node in tree.traverse('levelorder'):
            if node.is_root():
                for child in node.children:
                    child_bitarr = self.clade_to_bitarr(child.get_leaf_names())
                    nodetobitMap[child] = child_bitarr
                normalizing_const_est = self.clade_freq_est[self._minor_bitarr(child_bitarr).to01()]
                bn_est *= (self.clade_dict[self._minor_bitarr(child_bitarr).to01()] + MAP * self.alpha * normalizing_const_est) / (
                    1.0 + MAP * self.alpha)
            elif not node.is_leaf():
                for child in node.children:
                    child_bitarr = self.clade_to_bitarr(child.get_leaf_names())
                    nodetobitMap[child] = child_bitarr

                bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                if bn_est == 0.0: break
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
                    bn_est *= (self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] +
                               MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                   normalizing_const + MAP * self.alpha * normalizing_const_est)
                else:
                    bn_est = 0.0
                    break

        return bn_est

    def _bn_estimate_fast(self, tree, MAP=False):
        cbn_est_up = {node: 1.0 for node in tree.traverse('postorder') if not node.is_root()}
        Up = {node: 1.0 for node in tree.traverse('postorder') if not node.is_root()}
        nodetobitMap = {node: self.clade_to_bitarr(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        bipart_bitarr_up = {}
        bipart_bitarr_prob = {}

        for node in tree.traverse('postorder'):
            if not node.is_leaf() and not node.is_root():
                for child in node.children:
                    Up[node] *= cbn_est_up[child]

                bipart_bitarr = min(nodetobitMap[child] for child in node.children)
                bipart_bitarr_up[node] = bipart_bitarr
                if not node.up.is_root():
                    cbn_est_up[node] *= Up[node]
                    parent_bipart_bitarr = min([nodetobitMap[node.get_sisters()[0]], nodetobitMap[node]])
                    comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]

                    normalizing_const = self.clade_bipart_dict[nodetobitMap[node.up].to01()][parent_bipart_bitarr.to01()]
                    normalizing_const_est = self.clade_bipart_freq_est[nodetobitMap[node.up].to01()][parent_bipart_bitarr.to01()]
                    if (normalizing_const + MAP * self.alpha * normalizing_const_est
                        ) == 0.0 or self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] == 0.0:
                        cbn_est_up[node] = 0.0
                    else:
                        cbn_est_up[node] *= (self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] +
                                             MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]
                                             ) / (normalizing_const + MAP * self.alpha * normalizing_const_est)

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
                                cbn_est_down[node] *= Up[sister]
                                bipart_bitarr = min(nodetobitMap[child] for child in sister.children)
                                comb_parent_bipart_bitarr = ((~nodetobitMap[node]) ^ nodetobitMap[sister]) + nodetobitMap[sister]
                                if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr.to01()] > 0.0:
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
                        cbn_est_down[node] *= cbn_est_down[node.up]
                        comb_parent_bipart_bitarr = nodetobitMap[sister] + ~nodetobitMap[node.up]
                        if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node.up].to01()] > 0.0:
                            cbn_est_down[node] *= (
                                self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node.up].to01()] +
                                MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                    normalizing_const + MAP * self.alpha * normalizing_const_est)
                        else:
                            cbn_est_down[node] = 0.0

                        if not sister.is_leaf():
                            cbn_est_down[node] *= Up[sister]
                            comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[sister]
                            if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[sister].to01()] > 0.0:
                                cbn_est_down[node] *= (self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[sister].to01(
                                )] + MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                    normalizing_const + MAP * self.alpha * normalizing_const_est)
                            else:
                                cbn_est_down[node] = 0.0

                parent_bipart_bitarr = self._minor_bitarr(nodetobitMap[node])
                normalizing_const = self.clade_dict[parent_bipart_bitarr.to01()]
                normalizing_const_est = self.clade_freq_est[parent_bipart_bitarr.to01()]
                bipart_bitarr_prob[parent_bipart_bitarr.to01()] = (
                    self.clade_dict[parent_bipart_bitarr.to01()] + MAP * self.alpha * normalizing_const_est) / (1.0 + MAP * self.alpha)

                if (normalizing_const + MAP * self.alpha * normalizing_const_est) == 0.0:
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0
                else:
                    if not node.is_leaf():
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= Up[node]
                        comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                        if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[node].to01()] > 0.0:
                            bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= (
                                self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_up[node].to01()] +
                                MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                    normalizing_const + MAP * self.alpha * normalizing_const_est)
                        else:
                            bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0

                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= cbn_est_down[node]
                    comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                    if self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node].to01()] > 0.0:
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] *= (
                            self.clade_double_bipart_dict[comb_parent_bipart_bitarr.to01()][bipart_bitarr_down[node].to01()] +
                            MAP * self.alpha * normalizing_const_est / self.clade_double_bipart_len[comb_parent_bipart_bitarr.to01()]) / (
                                normalizing_const + MAP * self.alpha * normalizing_const_est)
                    else:
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0

        return bipart_bitarr_prob

    def bn_estimate(self, tree, MAP=False):
        return np.sum(self._bn_estimate_fast(tree, MAP).values())

    def kl_div(self, method='all', MAP=False):
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
