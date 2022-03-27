import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import random
import utils


class MyDataset(Dataset):

    def __init__(self, root: str, dataset_name: str, num_neibs: int = 8, fea_start: int = -1, fea_end: int = -1):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name

        od_data = np.loadtxt(root + dataset_name + '.csv', delimiter=',')
        xs = None
        if fea_start == -1 and fea_end == -1:
            xs = od_data[:, 0:-1]
        else:
            xs = od_data[:, fea_start: fea_end]
        self.X_train = xs.astype(np.float32)
        labels = od_data[:, -1]
        labels = labels.astype(np.float32)

        self.labels = labels
        self.size = labels.shape[0]
        self.data = torch.tensor(self.X_train, dtype=torch.float32)
        # self.data = torch.tensor(self.X_train)
        self.targets = torch.tensor(labels, dtype=torch.float32)
        # self.targets = torch.tensor(labels)

        # a neighbor list for each view
        self.num_views = utils.NUM_VIEWS
        self.num_neibs = num_neibs
        self.neibs_global = []
        self.noneibs_global = []
        self.weights_global = []
        self.neibs_local = []
        self.noneibs_local = []

        self.init_knn()

    def get_name(self):
        return self.dataset_name

    def get_all(self):
        return self.data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index)
        """

        sample, target = self.data[index], (self.targets[index])
        # if index == 0:
        #     tmp = self.neibs[index]
        #     list.sort(tmp)
        #     print(tmp)
        if len(self.neibs_global) > 0:
            tmp = random.randint(0, len(self.neibs_global[index]) - 1)
            id_global_neighbor = self.neibs_global[index][tmp]
            weight_global = self.weights_global[index][tmp]
            id_local_neighbor = random.choice(self.neibs_local[index])
        else:
            id_global_neighbor = index
            id_local_neighbor = index
            weight_global = 0.0
        weight_global = np.float32(weight_global)
        neighbor_global, neighbor_local = self.data[id_global_neighbor], self.data[id_local_neighbor]
        return sample, neighbor_global, neighbor_local, weight_global, target, index, id_global_neighbor, id_local_neighbor

    def init_knn(self):
        X = self.X_train
        nbrs = NearestNeighbors(n_neighbors=self.num_neibs+1, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)
        # print(indices)
        for i in range(self.size):
            self.neibs.append((list(indices[i, 1: ])))
        temp_n = set(range(self.size))
        for i in range(self.size):
            self.noneibs.append(list(temp_n - set(self.neibs[i])))
        print(self.neibs[0], self.noneibs[0])
        return indices

    def update_knn(self):
        return

    def get_knn(self):
        return self.neibs

    def set_knn(self, neibs_global, neibs_local, weights_global):
        self.neibs_global = neibs_global
        self.noneibs_global = []
        temp_n = set(range(self.size))
        for i in range(self.size):
            self.noneibs_global.append(list(temp_n - set(self.neibs_global[i])))
        self.weights_global = weights_global
        self.neibs_local = neibs_local
        self.noneibs_local = []
        temp_n = set(range(self.size))
        for i in range(self.size):
            self.noneibs_local.append(list(temp_n - set(self.neibs_local[i])))
        return

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)

