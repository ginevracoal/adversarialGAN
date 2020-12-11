import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import architecture.default
from architecture.default import Trainer, Defender

DEBUG=False
BATCH_SIZE=32
FIXED_POLICY=False
NORMALIZE=False
K=10
PENALTY=False
GAMMA=0.2

torch.set_default_tensor_type(torch.DoubleTensor)


class Attacker(architecture.default.Attacker):
    """ NN architecture for the attacker """
    def __init__(self, model, n_hidden_layers, layer_size, n_coeff, noise_size):
        super(Attacker, self).__init__(model, n_hidden_layers, layer_size, n_coeff, noise_size)
        
    def forward(self, x):
        output = self.nn(x)
        dot_eps = output[0]
        mu = torch.clamp(output[1], 0, 1000)
        return dot_eps, mu