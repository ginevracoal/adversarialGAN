import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

import architecture.default
from architecture.default import Defender

DEBUG=False
BATCH_SIZE=32
FIXED_POLICY=False
NORMALIZE=False
K=10
PENALTY=10
MAX_TARGET_POS=10

torch.set_default_tensor_type(torch.DoubleTensor)


class Attacker(architecture.default.Attacker):
    """ NN architecture for the attacker """
    def __init__(self, model, n_hidden_layers, layer_size, n_coeff, noise_size):
        super(Attacker, self).__init__(model, n_hidden_layers, layer_size, n_coeff, noise_size)
        
    def forward(self, x):
        output = self.nn(x)
        dot_eps = output[0]
        mu = torch.sigmoid(output[1])
        return dot_eps, mu

class Trainer(architecture.default.Trainer):
    """ The class contains the training logic """

    def __init__(self, world_model, robustness_computer, \
                attacker_nn, defender_nn, lr, logging_dir=None):
        super(Trainer, self).__init__(world_model, robustness_computer, attacker_nn, defender_nn, lr, logging_dir)

    def train_attacker_step(self, timesteps, dt, atk_static):

        self.attacker_optimizer.zero_grad()

        if FIXED_POLICY is True:
            z = torch.rand(self.attacker.noise_size)
            oe = torch.tensor(self.model.environment.status)
            oa = torch.tensor(self.model.agent.status)
            x_target = oe[3]

            atk_policy = self.attacker(torch.cat((z, oe)))

            with torch.no_grad():
                def_policy = self.defender(oa)

        cumloss = 0.

        if PENALTY:
            previous_def_policy = torch.zeros_like(self.defender(torch.tensor(self.model.agent.status)))

        for t in range(timesteps):

            if FIXED_POLICY is False:
                z = torch.rand(self.attacker.noise_size)
                oe = torch.tensor(self.model.environment.status)
                oa = torch.tensor(self.model.agent.status)
                x_target = oe[3]
                
                atk_policy = self.attacker(torch.cat((z, oe)))

                with torch.no_grad():
                    def_policy = self.defender(oa)

            self.model.step(atk_policy, def_policy, dt)

            if t>K:
                rho = self.robustness_computer.compute(self.model)
                cumloss += self.attacker_loss_fn(rho) 

                if torch.abs(x_target) >= MAX_TARGET_POS:
                    cumloss += PENALTY

        cumloss.backward()
        self.attacker_optimizer.step()  

        if DEBUG:
            print(self.attacker.state_dict()["nn.0.bias"])

        return cumloss.detach() / timesteps

    def train_defender_step(self, timesteps, dt, atk_static):

        self.defender_optimizer.zero_grad()

        if FIXED_POLICY is True:

            z = torch.rand(self.attacker.noise_size)
            oe = torch.tensor(self.model.environment.status)
            oa = torch.tensor(self.model.agent.status)
            x_target = oa[3]

            with torch.no_grad():
                atk_policy = self.attacker(torch.cat((z, oe)))

            def_policy = self.defender(oa)

        cumloss = 0.
        
        if PENALTY:
            previous_def_policy = torch.zeros_like(self.defender(torch.tensor(self.model.agent.status)))

        for t in range(timesteps):

            if FIXED_POLICY is False:

                z = torch.rand(self.attacker.noise_size)
                oe = torch.tensor(self.model.environment.status)
                oa = torch.tensor(self.model.agent.status)
                x_target = oa[3]

                with torch.no_grad():
                    atk_policy = self.attacker(torch.cat((z, oe)))

                def_policy = self.defender(oa)

            self.model.step(atk_policy, def_policy, dt)
        
            if t>K:
                rho = self.robustness_computer.compute(self.model)

                cumloss += self.defender_loss_fn(rho)

                if torch.abs(x_target) >= MAX_TARGET_POS:
                    cumloss += PENALTY

        cumloss.backward()
        self.defender_optimizer.step()  

        if DEBUG:
            print(self.defender.state_dict()["nn.0.bias"])
            # make_dot(def_input, self.defender.named_parameters(), path=self.logging_dir)

        return cumloss.detach() / timesteps