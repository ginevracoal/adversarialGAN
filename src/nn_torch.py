import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)


class Attacker(nn.Module):
    def __init__(self, model, n_hidden_layers, layer_size, noise_size, n_coeff=3):
        super().__init__()

        assert n_hidden_layers > 0

        self.noise_size = noise_size

        input_layer_size = model.environment.sensors + noise_size
        output_layer_size = model.environment.actuators * n_coeff

        layers = []
        layers.append(nn.Linear(input_layer_size, layer_size))
        layers.append(nn.LeakyReLU())

        for i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(layer_size, output_layer_size))

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        coefficients = self.nn(x)

        def policy_generator(t):
            basis = [t**i for i in range(len(coefficients))]
            basis = torch.tensor(basis, dtype=torch.get_default_dtype())
            return coefficients.dot(basis)

        return policy_generator


class Defender(nn.Module):
    def __init__(self, model, n_hidden_layers, layer_size, n_coeff=3):
        super().__init__()

        assert n_hidden_layers > 0

        input_layer_size = model.agent.sensors
        output_layer_size = model.agent.actuators * n_coeff

        layers = []
        layers.append(nn.Linear(input_layer_size, layer_size))
        layers.append(nn.LeakyReLU())

        for i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(layer_size, output_layer_size))

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        coefficients = self.nn(x)

        def policy_generator(t):
            basis = [t**i for i in range(len(coefficients))]
            basis = torch.tensor(basis, dtype=torch.get_default_dtype())
            return coefficients.dot(basis)

        return policy_generator


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
        z = torch.rand(self.attacker.noise_size)
        o = torch.tensor(self.model.environment.status)

        atk_policy = self.attacker(torch.cat((z, o)))

        with torch.no_grad():
            def_policy = self.defender(o)

        t = 0
        for i in range(time_horizon):
            atk_input = atk_policy(t)
            def_input = def_policy(t)

            self.model.step([atk_input], [def_input], dt)

            t += dt

        rho = self.robustness_computer.compute(self.model)

        self.attacker_optimizer.zero_grad()

        loss = self.attacker_loss_fn(rho)
        loss.backward()

        self.attacker_optimizer.step()

        return float(loss.detach())

    def train_defender_step(self, time_horizon, dt):
        z = torch.rand(self.attacker.noise_size)
        o = torch.tensor(self.model.agent.status)

        with torch.no_grad():
            atk_policy = self.attacker(torch.cat((z, o)))

        def_policy = self.defender(o)

        t = 0
        for i in range(time_horizon):
            atk_input = atk_policy(t)
            def_input = def_policy(t)

            self.model.step([atk_input], [def_input], dt)

            t += dt

        rho = self.robustness_computer.compute(self.model)

        self.defender_optimizer.zero_grad()

        loss = self.defender_loss_fn(rho)
        loss.backward()

        self.defender_optimizer.step()

        return float(loss.detach())

    def train(self, atk_steps, def_steps, time_horizon, dt):
        for i in range(atk_steps):
            self.model.initialize_random()
            atk_loss = self.train_attacker_step(time_horizon, dt)

        for i in range(def_steps):
            self.model.initialize_random()
            def_loss = self.train_defender_step(time_horizon, dt)

        return (atk_loss, def_loss)

    def run(self, n_steps, time_horizon=100, dt=0.05, *, atk_steps=1, def_steps=1):

        if self.logging:
            hist_every = int(n_steps / 10)
            hist_counter = 0

            atk_loss_vals = torch.zeros(n_steps)
            def_loss_vals = torch.zeros(n_steps)

        for i in tqdm(range(n_steps)):
            atk_loss, def_loss = self.train(atk_steps, def_steps, time_horizon, dt)

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


class Tester:
    def __init__(self, world_model, robustness_computer, \
                attacker_nn, defender_nn, logging_dir=None):

        self.model = world_model
        self.robustness_computer = robustness_computer

        self.attacker = attacker_nn
        self.defender = defender_nn

        self.logging = True if logging_dir else False

        if self.logging:
            self.log = SummaryWriter(logging_dir)

    def test(self, time_horizon, dt):
        self.model.initialize_random()

        for t in range(time_horizon):
            oa = torch.tensor(self.model.agent.status)
            oe = torch.tensor(self.model.environment.status)
            z = torch.rand(self.attacker.noise_size)

            with torch.no_grad():
                atk_policy = self.attacker(torch.cat((z, oe)))
                def_policy = self.defender(oa)

            atk_input = atk_policy(dt)
            def_input = def_policy(dt)

            self.model.step([atk_input], [def_input], dt)

        rho = self.robustness_computer.compute(self.model)

        return rho

    def run(self, times, time_horizon=1000, dt=0.05):

        if self.logging:
            def_rho_vals = torch.zeros(times)

        for i in tqdm(range(times)):
            def_rho = self.test(time_horizon, dt)

            if self.logging:
                def_rho_vals[i] = def_rho

        if self.logging:
            self.log.add_histogram('defender robustness', def_rho_vals, i)
            self.log.close()
