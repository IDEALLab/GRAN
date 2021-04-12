from __future__ import (division, print_function)
import os
import time
import networkx as nx
from networkx.algorithms import bipartite

import numpy as np
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.vis_helper import draw_graph_list, draw_graph_list_separate
from utils.data_parallel import DataParallel

from utils.eval_helper import mean_degree_stats, max_degree_stats, mean_centrality_stats, assortativity_stats, mean_degree_connectivity_stats, draw_hists

import seaborn as sns
import matplotlib.pyplot as plt

try:
    ###
    # workaround for solving the issue of multi-worker
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
    ###
except:
    pass

logger = get_logger('exp_logger')
__all__ = ['GranRunner', 'compute_edge_ratio', 'get_graph', 'evaluate']

NPR = np.random.RandomState(seed=1234)

# type_to_label = {'R': 0, 'L': 1, 'C': 2, 'D': 3, 'V': 4, 'S': 5, 'joint': 6, 'VP': 7, 'nonjoint': 8}
label_to_type = {0: 'R', 1: 'L', 2: 'C', 3: 'D', 4: 'V', 5: 'S', 6: 'joint', 7: 'VP', 8: 'nonjoint'}


def compute_edge_ratio(G_list):
    num_edges_max, num_edges = .0, .0
    for gg in G_list:
        num_nodes = gg.number_of_nodes()
        num_edges += gg.number_of_edges()
        num_edges_max += num_nodes**2

    ratio = (num_edges_max - num_edges) / num_edges
    return ratio


def get_graph(adj):
    """ get a graph from zero-padded adj """
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G


def get_graph_with_labels(adj, labels):
    '''
    Used for GRAN generated graphs.
    '''
    G = nx.from_numpy_matrix(adj)

    # add bipartite property
    for i, n in enumerate(G.nodes()):
        if labels[i] == 7:
            G.nodes[n]['bipartite'] = 1
        else:
            G.nodes[n]['bipartite'] = 0

    # rename nodes from 0, 1, ... to 'R0', 'L3', ...
    types = [label_to_type[l] for l in labels]  # e.g. ['R', 'L', 'R']
    types_unique = []  # target: ['R0', 'L0', 'R1']
    e_count = defaultdict(int)  # element count
    for t in types:
        t_unique = t + str(e_count[t])
        e_count[t] += 1
        types_unique.append(t_unique)
    print(types, '->', types_unique)
    print(e_count)

    mapping = dict([(i, types_unique[i]) for i in range(len(types_unique))])
    H = nx.relabel_nodes(G, mapping)

    # remove isolates and take the largest component
    H.remove_nodes_from(list(nx.isolates(H)))
    CGs = [H.subgraph(c) for c in nx.connected_components(H)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)

    return CGs[0]


def get_graph_from_nx(G):
    '''
    Used for training graphs.
    Could have combined this function with the previous one. But for neatness...
    G: a labelled networkx graph. Each node has an attribute 'label'.
        E.g. G.nodes[3]['label'] = 4 -> node 3 is type V.
    Returns:
    H: relabelled graph. E.g. node 3 is converted into node 'V1'.
    '''
    types = [label_to_type[G.nodes[n]['label']] for n in G.nodes]
    e_count = defaultdict(int)  # element count
    types_unique = []  # target: ['R0', 'L0', 'R1']
    for t in types:
        t_unique = t + str(e_count[t])
        e_count[t] += 1
        types_unique.append(t_unique)

    mapping = dict(zip(G.nodes, types_unique))
    H = nx.relabel_nodes(G, mapping)
    return H


def calculate_validity(graph_list):
    '''
    Prerequisite: Already removed isolates for each graph. 
    for each graph, returns:
        1. number of invalid nodes
        2. valid or not;
    Return:
        #valid / len(graph_list)
        save/return? a pmf figure
    '''
    n_invalid_nodes = []
    valid_count = 0
    bipartite_count = 0

    for G in graph_list:
        deg = [d for n, d in list(G.degree)]
        nn_inv = deg.count(1)
        n_invalid_nodes.append(nn_inv)
        if nn_inv == 0:
            valid_count += 1
            G.graph['valid'] = True
        else:
            G.graph['valid'] = False

        if nx.is_bipartite(G):
            bipartite_count += 1
            G.graph['bipartite'] = True
        else:
            G.graph['bipartite'] = False

    plt.figure()
    # sns.kdeplot(n_invalid_nodes, legend=False, color='magenta')
    plt.hist(n_invalid_nodes)
    plt.title('validity')
    plt.xlabel('number of invalid nodes/G')
    plt.ylabel('density')
    plt.savefig('validity')

    return valid_count / len(graph_list), bipartite_count / len(graph_list)


# def check_bipartite


def evaluate(graph_gt, graph_pred, degree_only=True):
    mmd_degree = degree_stats(graph_gt, graph_pred)
    '''=====XD:these five metrics are computed in accordance with RTRC====='''
    mmd_mean_degree = mean_degree_stats(graph_gt, graph_pred)
    mmd_max_degree = max_degree_stats(graph_gt, graph_pred)
    mmd_mean_centrality = mean_centrality_stats(graph_gt, graph_pred)
    mmd_assortativity = assortativity_stats(graph_gt, graph_pred)
    mmd_mean_degree_connectivity = mean_degree_connectivity_stats(graph_gt, graph_pred)
    '''=====XD====='''

    if degree_only:
        mmd_4orbits = 0.0
        mmd_clustering = 0.0
        mmd_spectral = 0.0
    else:
        mmd_4orbits = orbit_stats_all(graph_gt, graph_pred)
        mmd_clustering = clustering_stats(graph_gt, graph_pred)
        mmd_spectral = spectral_stats(graph_gt, graph_pred)

    return mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral, mmd_mean_degree, mmd_max_degree, mmd_mean_centrality, mmd_assortativity, mmd_mean_degree_connectivity


'''=====XD====='''


def generate_random_baseline_single(n, p=0.5):
    ''' 
  Generate a random graph with binomial degree distribution Binomial(n, p).
  '''
    adj = np.zeros(shape=(n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            adj[i, j] = np.random.binomial(n=1, p=p)  # n=1 equivalent to Bernoulli(p)

    # make adj symmetric
    adj = (adj + adj.T) / 2

    G = nx.from_numpy_matrix(adj)

    return G


'''=====XD====='''


class GranRunner(object):
    def __init__(self, config):
        self.config = config
        self.seed = config.seed
        self.dataset_conf = config.dataset
        self.model_conf = config.model
        self.train_conf = config.train
        self.test_conf = config.test
        self.use_gpu = config.use_gpu
        self.gpus = config.gpus
        self.device = config.device
        self.writer = SummaryWriter(config.save_dir)
        self.is_vis = config.test.is_vis
        self.better_vis = config.test.better_vis
        self.num_vis = config.test.num_vis
        self.vis_num_row = config.test.vis_num_row
        self.is_single_plot = config.test.is_single_plot
        self.num_gpus = len(self.gpus)
        self.is_shuffle = False

        assert self.use_gpu == True

        if self.train_conf.is_resume:
            self.config.save_dir = self.train_conf.resume_dir

        ### load graphs
        self.graphs = create_graphs(config.dataset.name, data_dir=config.dataset.data_path)

        self.train_ratio = config.dataset.train_ratio
        self.dev_ratio = config.dataset.dev_ratio
        self.block_size = config.model.block_size
        self.stride = config.model.sample_stride
        self.num_graphs = len(self.graphs)
        self.num_train = int(float(self.num_graphs) * self.train_ratio)
        self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
        self.num_test_gt = self.num_graphs - self.num_train
        self.num_test_gen = config.test.num_test_gen

        logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev, self.num_test_gt))

        ### shuffle all graphs
        if self.is_shuffle:
            self.npr = np.random.RandomState(self.seed)
            self.npr.shuffle(self.graphs)

        self.graphs_train = self.graphs[:self.num_train]
        self.graphs_dev = self.graphs[:self.num_dev]
        self.graphs_test = self.graphs[self.num_train:]

        self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
        logger.info('No Edges vs. Edges in training set = {}'.format(self.config.dataset.sparse_ratio))

        self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])
        self.max_num_nodes = len(self.num_nodes_pmf_train)
        self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()

        ### save split for benchmarking
        if config.dataset.is_save_split:
            base_path = os.path.join(config.dataset.data_path, 'save_split')
            if not os.path.exists(base_path):
                os.makedirs(base_path)

            save_graph_list(self.graphs_train, os.path.join(base_path, '{}_train.p'.format(config.dataset.name)))
            save_graph_list(self.graphs_dev, os.path.join(base_path, '{}_dev.p'.format(config.dataset.name)))
            save_graph_list(self.graphs_test, os.path.join(base_path, '{}_test.p'.format(config.dataset.name)))

    def train(self):
        ### create data loader
        train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, tag='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_conf.batch_size,
            shuffle=self.train_conf.shuffle,
            num_workers=self.train_conf.num_workers,
            collate_fn=train_dataset.collate_fn,
            drop_last=False)

        # create models
        model = eval(self.model_conf.name)(self.config)

        if self.use_gpu:
            model = DataParallel(model, device_ids=self.gpus).to(self.device)

        # create optimizer
        params = filter(lambda p: p.requires_grad, model.parameters())
        if self.train_conf.optimizer == 'SGD':
            optimizer = optim.SGD(
                params, lr=self.train_conf.lr, momentum=self.train_conf.momentum, weight_decay=self.train_conf.wd)
        elif self.train_conf.optimizer == 'Adam':
            optimizer = optim.Adam(params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
        else:
            raise ValueError("Non-supported optimizer!")

        early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.train_conf.lr_decay_epoch, gamma=self.train_conf.lr_decay)

        # reset gradient
        optimizer.zero_grad()

        # resume training
        resume_epoch = 0
        if self.train_conf.is_resume:
            model_file = os.path.join(self.train_conf.resume_dir, self.train_conf.resume_model)
            load_model(
                model.module if self.use_gpu else model,
                model_file,
                self.device,
                optimizer=optimizer,
                scheduler=lr_scheduler)
            resume_epoch = self.train_conf.resume_epoch

        # Training Loop
        iter_count = 0
        results = defaultdict(list)
        for epoch in range(resume_epoch, self.train_conf.max_epoch):
            model.train()
            lr_scheduler.step()
            train_iterator = train_loader.__iter__()

            for inner_iter in range(len(train_loader) // self.num_gpus):
                optimizer.zero_grad()

                batch_data = []
                if self.use_gpu:
                    for _ in self.gpus:
                        data = train_iterator.next()
                        batch_data.append(data)
                        iter_count += 1

                avg_train_loss = .0
                for ff in range(self.dataset_conf.num_fwd_pass):
                    batch_fwd = []

                    if self.use_gpu:
                        for dd, gpu_id in enumerate(self.gpus):
                            data = {}
                            data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)

                            data['node_label'] = batch_data[dd][ff]['node_label'].pin_memory().to(
                                gpu_id, non_blocking=True)
                            data['node_label_gt'] = batch_data[dd][ff]['node_label_gt'].pin_memory().to(
                                gpu_id, non_blocking=True)

                            data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
                            data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(
                                gpu_id, non_blocking=True)
                            data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(
                                gpu_id, non_blocking=True)
                            data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
                            data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
                            data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(
                                gpu_id, non_blocking=True)
                            data['subgraph_idx_base'] = batch_data[dd][ff]['subgraph_idx_base'].pin_memory().to(
                                gpu_id, non_blocking=True)
                            batch_fwd.append((data, ))

                    if batch_fwd:
                        train_loss = model(*batch_fwd).mean()
                        # return
                        avg_train_loss += train_loss

                        # assign gradient
                        train_loss.backward()

                # clip_grad_norm_(model.parameters(), 5.0e-0)
                optimizer.step()
                avg_train_loss /= float(self.dataset_conf.num_fwd_pass)

                # reduce
                train_loss = float(avg_train_loss.data.cpu().numpy())

                self.writer.add_scalar('train_loss', train_loss, iter_count)
                results['train_loss'] += [train_loss]
                results['train_step'] += [iter_count]

                if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
                    logger.info("NLL Loss @ epoch {:04d} iteration {:08d} = {}".format(
                        epoch + 1, iter_count, train_loss))

            # snapshot model
            if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
                logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
                snapshot(
                    model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)

        pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
        self.writer.close()

        return 1

    def test(self):
        self.config.save_dir = self.test_conf.test_model_dir

        ### Compute Erdos-Renyi baseline
        # if self.config.test.is_test_ER:
        p_ER = sum([aa.number_of_edges()
                    for aa in self.graphs_train]) / sum([aa.number_of_nodes()**2 for aa in self.graphs_train])
        # graphs_baseline = [nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii) for ii in range(self.num_test_gen)]
        graphs_gen = [nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii) for ii in range(self.num_test_gen)]
        temp = []
        for G in graphs_gen:
            G.remove_nodes_from(list(nx.isolates(G)))
            if G is not None:
                #  take the largest connected component
                CGs = [G.subgraph(c) for c in nx.connected_components(G)]
                CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
                temp.append(CGs[0])
        # graphs_gen = temp
        graphs_baseline = temp

        # else:
        ### load model
        model = eval(self.model_conf.name)(self.config)
        model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)
        load_model(model, model_file, self.device)

        if self.use_gpu:
            model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

        model.eval()

        ### Generate Graphs
        A_pred = []
        node_label_pred = []
        num_nodes_pred = []
        num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

        gen_run_time = []
        for ii in tqdm(range(num_test_batch)):
            with torch.no_grad():
                start_time = time.time()
                input_dict = {}
                input_dict['is_sampling'] = True
                input_dict['batch_size'] = self.test_conf.batch_size
                input_dict['num_nodes_pmf'] = self.num_nodes_pmf_train
                A_tmp, node_label_tmp = model(input_dict)
                gen_run_time += [time.time() - start_time]
                A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
                node_label_pred += [ll.data.cpu().numpy() for ll in node_label_tmp]
                num_nodes_pred += [aa.shape[0] for aa in A_tmp]
        # print(len(A_pred), type(A_pred[0]))

        logger.info('Average test time per mini-batch = {}'.format(np.mean(gen_run_time)))

        # print(A_pred[0].shape,
        #       get_graph(A_pred[0]).number_of_nodes(),
        #       get_graph_with_labels(A_pred[0], node_label_pred[0]).number_of_nodes())
        # print(A_pred[0])
        # return
        # graphs_gen = [get_graph(aa) for aa in A_pred]
        graphs_gen = [get_graph_with_labels(aa, ll) for aa, ll in zip(A_pred, node_label_pred)]
        valid_pctg, bipartite_pctg = calculate_validity(graphs_gen)  # for adding bipartite graph attribute

        # return

        ### Visualize Generated Graphs
        if self.is_vis:
            num_col = self.vis_num_row
            num_row = int(np.ceil(self.num_vis / num_col))
            test_epoch = self.test_conf.test_model_name
            test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
            save_name = os.path.join(self.config.save_dir, '{}_gen_graphs_epoch_{}_block_{}_stride_{}.png'.format(
                self.config.test.test_model_name[:-4], test_epoch, self.block_size, self.stride))

            # remove isolated nodes for better visulization
            # graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[:self.num_vis]]
            graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen if not gg.graph['bipartite']]
            logger.info('Number of not bipartite graphs: {} / {}'.format(len(graphs_pred_vis), len(graphs_gen)))
            # if self.better_vis:
            #     for gg in graphs_pred_vis:
            #         gg.remove_nodes_from(list(nx.isolates(gg)))

            # # display the largest connected component for better visualization
            # vis_graphs = []
            # for gg in graphs_pred_vis:
            #     CGs = [gg.subgraph(c) for c in nx.connected_components(gg)] # nx.subgraph makes a graph frozen!
            #     CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            #     vis_graphs += [CGs[0]]
            vis_graphs = graphs_pred_vis

            if self.is_single_plot:
                draw_graph_list(vis_graphs, num_row, num_col, fname=save_name, layout='spring')
            else:  #XD: using this for now
                draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='spring')

            save_name = os.path.join(self.config.save_dir, 'train_graphs.png')
            if self.is_single_plot:
                draw_graph_list(self.graphs_train[:self.num_vis], num_row, num_col, fname=save_name, layout='spring')
                print('training single plot saved at:', save_name)
            else:  #XD: using this for now
                graph_list_train = [get_graph_from_nx(G) for G in self.graphs_train[:self.num_vis]]
                draw_graph_list_separate(graph_list_train, fname=save_name[:-4], is_single=True, layout='spring')
                print('training plots saved individually at:', save_name[:-4])
        return

        ### Evaluation
        if self.config.dataset.name in ['lobster']:
            acc = eval_acc_lobster_graph(graphs_gen)
            logger.info('Validity accuracy of generated graphs = {}'.format(acc))
        '''=====XD====='''
        ## graphs_gen = [generate_random_baseline_single(len(aa)) for aa in graphs_gen]  # use this line for random baseline MMD scores. Remember to comment it later!
        # draw_hists(self.graphs_test, graphs_baseline, graphs_gen)
        valid_pctg, bipartite_pctg = calculate_validity(graphs_gen)
        # logger.info('Generated {} graphs, valid percentage = {:.2f}, bipartite percentage = {:.2f}'.format(
        #     len(graphs_gen), valid_pctg, bipartite_pctg))
        # # return
        '''=====XD====='''

        num_nodes_gen = [len(aa) for aa in graphs_gen]

        # # Compared with Validation Set
        # num_nodes_dev = [len(gg.nodes) for gg in self.graphs_dev]  # shape B X 1
        # mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_mean_degree_dev, mmd_max_degree_dev, mmd_mean_centrality_dev, mmd_assortativity_dev, mmd_mean_degree_connectivity_dev = evaluate(self.graphs_dev, graphs_gen, degree_only=False)
        # mmd_num_nodes_dev = compute_mmd([np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)
        # logger.info("Validation MMD scores of #nodes/degree/clustering/4orbits/spectral/... are = {:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}".format(mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_mean_degree_dev, mmd_max_degree_dev, mmd_mean_centrality_dev, mmd_assortativity_dev, mmd_mean_degree_connectivity_dev))

        # Compared with Test Set
        num_nodes_test = [len(gg.nodes) for gg in self.graphs_test]  # shape B X 1
        mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test, mmd_mean_degree_test, mmd_max_degree_test, mmd_mean_centrality_test, mmd_assortativity_test, mmd_mean_degree_connectivity_test = evaluate(
            self.graphs_test, graphs_gen, degree_only=False)
        mmd_num_nodes_test = compute_mmd(
            [np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

        logger.info(
            "Test MMD scores of #nodes/degree/clustering/4orbits/spectral/... are = {:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}/{:.5f}".
            format(mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test,
                   mmd_mean_degree_test, mmd_max_degree_test, mmd_mean_centrality_test, mmd_assortativity_test,
                   mmd_mean_degree_connectivity_test))

        # if self.config.dataset.name in ['lobster']:
        #   return mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test, acc
        # else:
        #   return mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test
