import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import architecture.default
from architecture.default import Trainer

K=10
PENALTY=True
FIXED_POLICY=False
THS_POWER=100

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

# class Trainer(architecture.default.Trainer):
#     """ The class contains the training logic """

#     def __init__(self, world_model, robustness_computer, \
#                 attacker_nn, defender_nn, lr, logging_dir=None):
#         super(Trainer, self).__init__(world_model, robustness_computer, attacker_nn, defender_nn, lr, logging_dir)

#     def train_defender_step(self, timesteps, dt, atk_static):

#         self.defender_optimizer.zero_grad()

#         if FIXED_POLICY is True:

#             z = torch.rand(self.attacker.noise_size)
#             oe = torch.tensor(self.model.environment.status)
#             oa = torch.tensor(self.model.agent.status)

#             with torch.no_grad():
#                 atk_policy = self.attacker(torch.cat((z, oe)))

#             def_policy = self.defender(oa)

#         cumloss = 0.
        
#         for t in range(timesteps):

#             if FIXED_POLICY is False:

#                 z = torch.rand(self.attacker.noise_size)
#                 oe = torch.tensor(self.model.environment.status)
#                 oa = torch.tensor(self.model.agent.status)

#                 with torch.no_grad():
#                     atk_policy = self.attacker(torch.cat((z, oe)))

#                 def_policy = self.defender(oa)

#             self.model.step(atk_policy, def_policy, dt)
        
#             if t>K:
#                 rho = self.robustness_computer.compute(self.model)
#                 cumloss += self.defender_loss_fn(rho)

#                 if PENALTY:
#                     timestep_power = self.model.agent._car.timestep_power.item()
#                     if timestep_power > THS_POWER:
#                         cumloss += 100

#         cumloss.backward()
#         self.defender_optimizer.step()  
#         return cumloss.detach() / timesteps