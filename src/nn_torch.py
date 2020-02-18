import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def policy(coeff):
    return lambda x: coeff.dot(torch.tensor([1.0, x, x*x]))

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        self.input_size = layers[0].in_features
        self.output_size = layers[-1].out_features
        
        self.nn = nn.Sequential(*layers)
        self.nn.double()

    def forward(self, x):
        return self.nn(x)


class Trainer:
    def __init__(self, world_model, robustness_computer, \
                attacker_nn, defender_nn, logging_dir=None):

        self.model = world_model
        self.robustness_computer = robustness_computer

        self.attacker = attacker_nn
        self.defender = defender_nn

        self.attacker_loss_fn = lambda x: x
        self.defender_loss_fn = lambda x: -x

        atk_optimizer = optim.SGD(attacker_nn.parameters(), lr=0.01, momentum=0.9)
        def_optimizer = optim.SGD(defender_nn.parameters(), lr=0.01, momentum=0.9)
        self.attacker_optimizer = atk_optimizer
        self.defender_optimizer = def_optimizer

        self.logging = True if logging_dir else False

        if self.logging:
            self.log = SummaryWriter(logging_dir)

    def train_attacker_step(self, time_horizon, dt):
        z = torch.rand(5)
        o = torch.tensor(self.model.get_status())

        atk_coeff = self.attacker(torch.cat((z, o)))

        with torch.no_grad():
            def_coeff = self.defender(o)

        atk_p = policy(atk_coeff)
        def_p = policy(def_coeff)

        t = 0
        for i in range(time_horizon):
            atk_input = atk_p(t)
            def_input = def_p(t)

            self.model.step([atk_input], [def_input], dt)

            t += dt

        rho = self.robustness_computer.compute(self.model)

        self.defender_optimizer.zero_grad()

        loss = self.attacker_loss_fn(rho)
        loss.backward()

        self.defender_optimizer.step()

        return float(loss.detach())

    def train_defender_step(self, time_horizon, dt):
        z = torch.rand(5)
        o = torch.tensor(self.model.get_status())

        with torch.no_grad():
            atk_coeff = self.attacker(torch.cat((z, o)))

        def_coeff = self.defender(o)

        atk_p = policy(atk_coeff)
        def_p = policy(def_coeff)

        t = 0
        for i in range(time_horizon):
            atk_input = atk_p(t)
            def_input = def_p(t)

            self.model.step([atk_input], [def_input], dt)

            t += dt

        rho = self.robustness_computer.compute(self.model)

        self.defender_optimizer.zero_grad()

        loss = self.defender_loss_fn(rho)
        loss.backward()

        self.defender_optimizer.step()

        return float(loss.detach())

    def train(self, iteration, time_horizon, dt):
        self.model.initialize_random()
        atk_loss = self.train_attacker_step(time_horizon, dt)

        self.model.initialize_random()
        def_loss = self.train_defender_step(time_horizon, dt)

        return (atk_loss, def_loss)

    def run(self, n_steps, time_horizon=100, dt=0.05):

        if self.logging:
            hist_every = int(n_steps / 10)
            hist_counter = 0

            atk_loss_vals = torch.zeros(n_steps)
            def_loss_vals = torch.zeros(n_steps)

        for i in tqdm(range(n_steps)):
            atk_loss, def_loss = self.train(i, time_horizon, dt)

            if self.logging:
                atk_loss_vals[i] = atk_loss
                def_loss_vals[i] = def_loss

                self.log.add_scalar('attacker loss', atk_loss, i)
                self.log.add_scalar('defender loss', def_loss, i)

                if (i + 1) % hist_every == 0:
                    a = hist_counter * hist_every
                    b = (hist_counter + 1) * hist_every
                    hist_counter += 1

                    self.log.add_histogram('attacker loss hist', atk_loss_vals[a:b], i)
                    self.log.add_histogram('defender loss hist', def_loss_vals[a:b], i)

        if self.logging:
            self.log.close()
