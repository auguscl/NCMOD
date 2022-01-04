import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import sys
from PIL import Image
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from sv_net import Autoencoder
from sv_trainer import AETrainer
import logging
from my_dataset import MyDataset
import utils

torch.manual_seed(1)    # reproducible
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run(data_path, dataset_name, batch_size, learning_rate, k_neibs, module_weight, albation_set):
    logger = logging.getLogger()

    print('#########  LocalAE conducted on ' + dataset_name)
    # print(torch.cuda.get_device_name(0))
    num_rounds = 10
    num_epochs = 16

    net_name = 'AE_KNN'

    logging.basicConfig(level=logging.INFO, filename='running_log')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'

    device = 'cpu'
    num_views = utils.NUM_VIEWS
    num_neibs = k_neibs
    pre_train = False
    pre_epochs = 100

    s, e = utils.get_view_fea()
    # print(s)
    # print(e)

    all_nets = [None for a in range(num_views)]
    datasets = [None for a in range(num_views)]
    view_encoded = [None for a in range(num_views)]
    recon_error = [None for a in range(num_views)]
    labels = None

    for i in range(num_views):
        datasets[i] = MyDataset(root=data_path, dataset_name=dataset_name, num_neibs=num_neibs, fea_start=s[i], fea_end=e[i])
    if labels is None:
        labels = datasets[0].get_labels()

    for v in range(num_views):
        # all_nets[v].set_network(net_name, e[v] - s[v])
        all_nets[v] = Autoencoder(e[v] - s[v])

    Trainer = AETrainer(optimizer_name='adam',
                        lr=learning_rate,
                        n_epochs=num_epochs,
                        lr_milestones=tuple(()),
                        batch_size=batch_size,
                        weight_decay=1e-6,
                        device=device,
                        n_jobs_dataloader=0)

    for eround in range(num_rounds):
        for id_view in range(num_views):
            all_nets[id_view], view_encoded[id_view], recon_error[id_view] =\
                Trainer.train(eround, id_view, datasets[id_view], all_nets[id_view], pre_train, pre_epochs,
                              module_weight, albation_set)
        neibs_global, neibs_local, weights_global = get_new_knn2(view_encoded, num_neibs)

        print('round %d: ' % eround)
        for id_view in range(num_views):
            tmp1 = list(neibs_global[id_view][0])
            tmp2 = list(neibs_local[id_view][0])
            list.sort(tmp1)
            list.sort(tmp2)
            print('View %d: global neighbors is %s, local neighbors is %s.' % (id_view + 1, tmp1, tmp2))
            datasets[id_view].set_knn(neibs_global[id_view], neibs_local[id_view], weights_global[id_view])

    recon_scores, knn_scores = cal_scores(view_encoded, num_neibs, recon_error)

    recon_auc = utils.compute_auc(labels, recon_scores)
    recon_fs = utils.compute_f1_score(labels, recon_scores)
    # auc, precision, recall, f1-score
    print('recon performance: %.4f  %.4f  %.4f  %.4f' % (recon_auc, recon_fs[0], recon_fs[1], recon_fs[2]))

    knn_auc = utils.compute_auc(labels, knn_scores)
    knn_fs = utils.compute_f1_score(labels, knn_scores)
    # auc, precision, recall, f1-score
    print('knn performance: %.4f  %.4f  %.4f  %.4f' % (knn_auc, knn_fs[0], knn_fs[1], knn_fs[2]))

    # Scale to range [0,1]
    total_scores = cal_final_scores(recon_scores, knn_scores)
    total_auc = utils.compute_auc(labels, total_scores)
    total_fs = utils.compute_f1_score(labels, total_scores)
    # auc, precision, recall, f1-score
    print('total performance: %.4f  %.4f  %.4f  %.4f' % (total_auc, total_fs[0], total_fs[1], total_fs[2]))

    test_auc = albation_set[0] * recon_auc + albation_set[1] * knn_auc
    test_f1 = albation_set[0] * recon_fs[0] + albation_set[1] * knn_fs[0]

    if albation_set == (1, 1):
        test_auc = knn_auc
        test_f1 = knn_fs[0]
    return test_auc, test_f1
    # return knn_auc, knn_fs[0]


def cal_final_scores(recon_scores, knn_scores):
    # total_scores = recon_scores + knn_scores
    s1 = (recon_scores - np.min(recon_scores)) / (np.max(recon_scores) - np.min(recon_scores))
    s2 = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores))
    total_scores = 0.5 * s1 + s2
    # total_scores = np.maximum(s1, s2)
    # rank-based
    return total_scores


def get_new_knn2(view_encoded, num_neibs):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]

    neibs_local = [[] for a in range(num_views)]
    kth_neib_local = [[] for a in range(num_views)]
    for i in range(num_views):
        nbrs = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(view_encoded[i])
        distances, indices = nbrs.kneighbors(view_encoded[i])
        # print(distances[332])
        # print(indices[332])
        # tmp = list(indices[0])
        # list.sort(tmp)
        # print('View %d: %s' % (i + 1, tmp))
        for j in range(num_obj):
            indice_j = list(indices[j])
            if j in indice_j:
                indice_j.remove(j)
            else:
                indice_j = indice_j[1: ]
            neibs_local[i].append(indice_j)
            kth_neib_local[i].append(indices[j, num_neibs])

    weights_global = [[] for a in range(num_views)]
    neibs_global = [[] for a in range(num_views)]
    for i in range(num_views):
        for j in range(num_obj):
            tmp = []
            tmp_weights = []
            for k in range(num_views):
                if k != i:
                    tmp = tmp + neibs_local[k][j]
            for k in range(len(tmp)):
                if cal_dis(view_encoded[i], j, tmp[k]) > 1e-8:
                    w = cal_dis(view_encoded[i], j, kth_neib_local[i][j]) / cal_dis(view_encoded[i], j, tmp[k])
                else:
                    w = 2
                tmp_weights.append(w)
            neibs_global[i].append(tmp)
            weights_global[i].append(tmp_weights)
    # print(weights_global[0][0])
    return neibs_global, neibs_local, weights_global


def get_new_knn(view_encoded, num_neibs):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]
    dim_concen = 0
    for i in range(num_views):
        dim_concen += view_encoded[i].shape[1]

    encoded_concen = np.zeros((num_obj, dim_concen))
    tmp_dim = 0
    for i in range(num_views):
        encoded_concen[:, tmp_dim: tmp_dim + view_encoded[i].shape[1]] = view_encoded[i]
        tmp_dim += view_encoded[i].shape[1]

    neibs_view = [[] for a in range(num_views)]
    for i in range(num_views):
        nbrs = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree').fit(view_encoded[i])
        distances, indices = nbrs.kneighbors(view_encoded[i])
        # print(indices)
        for j in range(num_obj):
            neibs_view[i].append((list(indices[j, 1:])))

    neibs = []
    nbrs = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree').fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)
    # print(indices)
    for i in range(num_obj):
        neibs.append((list(indices[i, 1: ])))
    return neibs

def cal_dis(X, a, b):
    return np.sum((X[a] - X[b]) ** 2)

def cal_scores(view_encoded, num_neibs, recon_error):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]

    recon_scores = np.zeros(num_obj)
    for i in range(num_views):
        s1 = (recon_error[i] - np.min(recon_error[i])) / (np.max(recon_error[i]) - np.min(recon_error[i]))
        recon_scores += s1

    dim_concen = 0
    for i in range(num_views):
        dim_concen += view_encoded[i].shape[1]
    encoded_concen = np.zeros((num_obj, dim_concen))
    tmp_dim = 0
    for i in range(num_views):
        encoded_concen[:, tmp_dim: tmp_dim + view_encoded[i].shape[1]] = view_encoded[i]
        tmp_dim += view_encoded[i].shape[1]
    neibs = []
    NNK = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree')

    # concentrate neighbors
    nbrs = NNK.fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)
    knn_scores = np.sum(distances, axis=1)
    tmp = list(indices[0][1:])
    list.sort(tmp)
    print('last neighbors: ')
    print(tmp)

    NNK = NearestNeighbors(n_neighbors=num_views * num_neibs + 1, algorithm='ball_tree')
    knn_scores2 = np.zeros(num_obj)
    for i in range(num_views):
        view_nbrs = NNK.fit(view_encoded[i])
        _, view_indice = view_nbrs.kneighbors(view_encoded[i])
        # print('view %d: ' % (i+1))
        # tmp = list(view_indice[0][1:])
        # list.sort(tmp)
        # print(tmp)
        for j in range(num_obj):
            for k in range(num_views):
                if i != k:
                    for l in range(1, num_neibs+1):
                        knn_scores2[j] += np.sum((view_encoded[k][j] - view_encoded[k][view_indice[j][l]]) ** 2)
    # print(np.average(knn_scores))
    return recon_scores, knn_scores


if __name__ == '__main__':
    run(utils.data_root_path, utils.FILE_NAME, 10, 0.001)
