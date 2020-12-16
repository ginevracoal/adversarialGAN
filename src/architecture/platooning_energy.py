import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import architecture.default
from architecture.default import Trainer


torch.set_default_tensor_type(torch.DoubleTensor)


class Attacker(architecture.default.Attacker):
    """ NN architecture for the attacker """
    def __init__(self, model, n_hidden_layers, layer_size, n_coeff, noise_size):
        super(Attacker, self).__init__(model, n_hidden_layers, layer_size, n_coeff, noise_size)
        
    def forward(self, x):
        output = self.nn(x)
        e_torque = torch.tanh(output[0])
        br_torque = torch.sigmoid(output[1])
        return e_torque, br_torque

class Defender(architecture.default.Defender):
    """ NN architecture for the defender """
    def __init__(self, model, n_hidden_layers, layer_size, n_coeff):
        super(Defender, self).__init__(model, n_hidden_layers, layer_size, n_coeff)

    def forward(self, x):
        output = self.nn(x)
        e_torque = torch.tanh(output[0])
        br_torque = torch.sigmoid(output[1])        
        return e_torque, br_torque
