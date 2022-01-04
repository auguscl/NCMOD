import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import utils

class Autoencoder(nn.Module):

    def __init__(self, data_dim):
        super().__init__()
        self.rep_dim = 32
        self.data_dim = data_dim

        en_layers_num = None
        if utils.DATA_DIM == 784:
            en_layers_num = [self.data_dim, 128, self.rep_dim]  # MNIST
        elif utils.DATA_DIM == 2000:
            en_layers_num = [self.data_dim, 128, self.rep_dim]  # REUTERS
        elif utils.DATA_DIM == 7507:
            en_layers_num = [self.data_dim, 256, self.rep_dim]  # TTC
        else:
            self.rep_dim = 16
            en_layers_num = [self.data_dim, 128, self.rep_dim]  # syn
            # print(en_layers_num)

        self.encoder = self.encode(en_layers_num)
        de_layers_num = list(reversed(en_layers_num))
        self.decoder = self.decode(de_layers_num)

    def encode(self, layers_num):
        if len(layers_num) > 2:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
            for i in range(1, len(layers_num) - 2):
                encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[i], layers_num[i+1]), nn.Tanh())
            encoded_output = nn.Sequential(encoded_output, nn.Linear(layers_num[len(layers_num)-2],
                                                                     layers_num[len(layers_num)-1]))
        else:
            encoded_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())

        return encoded_output

    def decode(self, layers_num):
        if len(layers_num) > 2:
            decode_output = nn.Sequential(nn.Linear(layers_num[0], layers_num[1]), nn.Tanh())
            for i in range(1, len(layers_num) - 2):
                decode_output = nn.Sequential(decode_output, nn.Linear(layers_num[i], layers_num[i+1]), nn.Tanh())
            decode_output = nn.Sequential(decode_output, nn.Linear(layers_num[len(layers_num)-2],
                                                                   layers_num[len(layers_num)-1]), nn.Sigmoid())
        return decode_output

    def forward(self, x):
        # print(x.size())
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # print(decoded.size())
        return encoded, decoded
