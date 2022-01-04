import logging
import time
import torch
import torch.optim as optim
import numpy as np
import utils
import matplotlib.pyplot as plt
import itertools
from torch.utils.data import DataLoader, Subset


class AETrainer:

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                n_jobs_dataloader: int = 0):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader


    def train(self, id_round, id_view, dataset, view_net, pre_train, pre_epochs, module_weight, albation_set):
        logger = logging.getLogger()

        # Set device for network
        view_net = view_net.to(self.device)

        # Get train data loader
        train_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.n_jobs_dataloader, drop_last=False)

        # Set optimizer (Adam optimizer for now)
        all_parameters = view_net.parameters()
        optimizer = optim.Adam(all_parameters, lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        start_time = time.time()
        view_net.train()

        num_epochs = self.n_epochs
        if pre_train and id_round == 0:
            num_epochs = pre_epochs

        for epoch in range(num_epochs):
            # time.sleep(1)

            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))
            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, neighbor_global, neighbor_local, weights_global, labels, index, index_g, index_l = data
                # print(inputs.shape, neighbor.shape, no_neighbor.shape)
                inputs = inputs.to(self.device)
                neighbor_global = neighbor_global.to(self.device)
                neighbor_local = neighbor_local.to(self.device)
                labels = labels.to(self.device)
                weights_global = weights_global.to(self.device)
                # print(weights_global.dtype)
                # print(weights_global)
                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                objectsize = inputs.shape[0]
                encoded, outputs = view_net(inputs)
                encoded_global_neighbor, _ = view_net(neighbor_global)
                encoded_local_neighbor, _ = view_net(neighbor_local)

                recon_error = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))

                dis_global = torch.sum((encoded - encoded_global_neighbor) ** 2, dim=tuple(range(1, outputs.dim())))
                dis_local = torch.sum((encoded - encoded_local_neighbor) ** 2, dim=tuple(range(1, outputs.dim())))


                if pre_train and id_round == 0:
                    scores = recon_error
                else:

                    scores = albation_set[0] * recon_error + albation_set[1] * module_weight * weights_global * dis_global

                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  View {} Round {} Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f} '
                        .format(id_view + 1, id_round, epoch + 1, num_epochs, epoch_train_time, loss_epoch / n_batches))
            # if epoch in np.array([10 - 1, 50 - 1, 100 - 1, 150 - 1, 200 - 1, 250 - 1, 300 - 1,
            #                       350 - 1, 400 - 1, 450 - 1, 500 - 1, 550 - 1, 600 - 1]):

            # logger.info('  Intermediate Test at Epoch {}'.format(epoch + 1))

        view_encoded, recon_error = self.test(id_view, dataset, view_net)

        train_time = time.time() - start_time
        logger.info('View {} Training time: {:.3f}'.format(id_view + 1, train_time))
        # logger.info('Finished training.')

        return view_net, view_encoded, recon_error

    def test(self, id_view, dataset, ae_net):
        logger = logging.getLogger()

        # Set device for network
        ae_net = ae_net.to(self.device)

        # Get test data loader
        test_loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.n_jobs_dataloader, drop_last=False)

        # Testing
        logger.info('View {} Testing lg_ae...'.format(id_view + 1))
        loss_epoch = 0.0

        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        encoded_data = None

        loss_recon_epoch = 0.0

        with torch.no_grad():
            for data in test_loader:
                inputs, _, _, _, labels, idx, _, _ = data
                inputs = inputs.to(self.device)

                encoded, outputs = ae_net(inputs)
                recon_error = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                # print(recon_error.size())
                scores = recon_error

                loss = torch.mean(scores)
                loss_recon = torch.mean(recon_error)

                if n_batches == 0:
                    encoded_data = encoded.cpu().data.numpy()
                else:
                    encoded_data = np.concatenate((encoded_data, encoded.cpu().numpy()), axis=0)
                # print(encoded_data.shape)

                # Save triple of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                loss_recon_epoch += loss_recon.item()
                n_batches += 1

        logger.info('View {}: Test set score: {:.8f}, Recon_score: {:.8f}'
                    .format(id_view, loss_epoch / n_batches, loss_recon_epoch / n_batches))

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)


        # if ae_net.rep_dim == 2:
        #     self.circle_plot(labels, encoded_data, epoch, dataset.get_name())

        # self.dist_plot(labels, scores, epoch, dataset.get_name())

        return encoded_data, scores

    def circle_plot(self, labels, encoded_data, epoch, dataset_name):
        # embedd_dim = 2
        plot_x = encoded_data
        plot_center = np.mean(encoded_data, 0)
        outlier_num = np.sum(labels)
        inlier_num = encoded_data.shape[0] - outlier_num
        print(encoded_data.shape, inlier_num, outlier_num)
        print(plot_center)

        plt.rcParams['figure.figsize'] = (8.0, 8.0)
        plt.scatter(encoded_data[0: inlier_num, 0], encoded_data[0: inlier_num, 1], c='g', marker='*')
        plt.scatter(encoded_data[inlier_num: encoded_data.shape[0], 0],
                    encoded_data[inlier_num: encoded_data.shape[0], 1], c='r', marker='x')
        plt.scatter(plot_center[0], plot_center[1], c='b')
        plt.savefig('./circles_changes/' + dataset_name + '-' + 'lgae' + str(epoch) + '.jpg')
        # plt.show()
        plt.close()


    def dist_plot(self, lables, scores, epoch, dataset_name):
        out_scores = []
        in_scores = []
        for i in range(len(lables)):
            if lables[i] == 1:
                out_scores.append(scores[i])
            else:
                in_scores.append(scores[i])

        plt.figure(figsize=(7, 6))

        font1 = {'family': 'Times New Roman',
                 'style': 'normal',
                 'weight': 'normal',
                 'size': 35,
                 }
        plt.grid(axis='both', linestyle='--')
        plt.hist(in_scores, bins=30, normed=True, facecolor='green', alpha=0.75, label='Inlier')
        plt.hist(out_scores, bins=30, normed=True, facecolor='red', alpha=0.75, label='Outlier')
        # plt.xlabel('Smarts', font1)
        # plt.ylabel('Probability', font1)
        # plt.xticks([])
        # plt.yticks([])
        plt.legend(prop=font1)
        plt.savefig('./loss_changes/' + dataset_name + '-' + 'good' + str(epoch) + '.jpg')
        # plt.close()
        # plt.show()
        return


