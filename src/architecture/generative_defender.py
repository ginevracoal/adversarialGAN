import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.print_pytorch_autograd import make_dot

DEBUG=False
BATCH_SIZE=16
NORMALIZE=False

torch.set_default_tensor_type(torch.DoubleTensor)


class Attacker(nn.Module):
    """ NN architecture for the attacker """
    def __init__(self, model, n_hidden_layers, layer_size, n_coeff, noise_size):
        super().__init__()

        assert n_hidden_layers > 0

        self.hid = n_hidden_layers
        self.ls = layer_size
        self.n_coeff = n_coeff
        self.noise_size = noise_size

        input_layer_size = model.environment.sensors + noise_size
        output_layer_size = model.environment.actuators * n_coeff

        layers = []
        if NORMALIZE:
            layers.append(nn.LayerNorm(input_layer_size))
        layers.append(nn.Linear(input_layer_size, layer_size))
        layers.append(nn.LeakyReLU())

        for i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(layer_size, output_layer_size))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        """ Uses the NN's output to compute the coefficients of the policy function """
        output = self.nn(x)
        return output


class Defender(nn.Module):
    """ NN architecture for the defender """

    def __init__(self, model, n_hidden_layers, layer_size, n_coeff, noise_size):
        super().__init__()

        assert n_hidden_layers > 0

        self.hid = n_hidden_layers
        self.ls = layer_size
        self.n_coeff = n_coeff
        self.noise_size = noise_size

        input_layer_size = model.agent.sensors + noise_size
        output_layer_size = model.agent.actuators * n_coeff

        layers = []
        if NORMALIZE:
            layers.append(nn.LayerNorm(input_layer_size))
        layers.append(nn.Linear(input_layer_size, layer_size))
        layers.append(nn.LeakyReLU())

        for i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(layer_size, output_layer_size))
        self.nn = nn.Sequential(*layers)


    def forward(self, x):
        """ Uses the NN's output to compute the coefficients of the policy function """
        output = self.nn(x)
        return output


class Trainer:
    """ The class contains the training logic """

    def __init__(self, world_model, robustness_computer, \
                attacker_nn, defender_nn, lr, logging_dir=None):

        self.model = world_model
        self.robustness_computer = robustness_computer

        self.attacker = attacker_nn
        self.defender = defender_nn

        self.attacker_loss_fn = lambda x: x
        self.defender_loss_fn = lambda x: -x

        self.attacker_optimizer = optim.Adam(self.attacker.parameters(), lr=lr)
        self.defender_optimizer = optim.Adam(self.defender.parameters(), lr=lr)

        self.logging = True if logging_dir else False
        if self.logging:
            self.logging_dir = logging_dir

    def train_attacker_step(self, timesteps, dt):
        
        self.attacker_optimizer.zero_grad()

        loss = 0
        for _ in range(timesteps):

            ze = torch.rand(self.attacker.noise_size)
            oe = torch.tensor(self.model.environment.status)
            za = torch.rand(self.defender.noise_size)
            oa = torch.tensor(self.model.agent.status)

            atk_policy = self.attacker(torch.cat((ze, oe)))

            with torch.no_grad():
                def_policy = self.defender(torch.cat((za, oa)))

            self.model.step(atk_policy, def_policy, dt)
            rho = self.robustness_computer.compute(self.model)
            loss += self.attacker_loss_fn(rho)

        loss.backward()
        self.attacker_optimizer.step()  

        if DEBUG:
            print(self.attacker.state_dict()["nn.0.bias"])

        return loss.detach() / timesteps

    def train_defender_step(self, timesteps, dt, iter_idx):

        self.defender_optimizer.zero_grad()

        loss = 0.
        for _ in range(timesteps):

            ze = torch.rand(self.attacker.noise_size)
            oe = torch.tensor(self.model.environment.status)
            distr = MultivariateNormal(torch.zeros(self.defender.noise_size),
                                       torch.eye(self.defender.noise_size)/(iter_idx+1))
            za = distr.sample()
            oa = torch.tensor(self.model.agent.status)

            with torch.no_grad():
                atk_policy = self.attacker(torch.cat((ze, oe)))

            def_policy = self.defender(torch.cat((za, oa)))

            self.model.step(atk_policy, def_policy, dt)
        
            rho = self.robustness_computer.compute(self.model)
            loss += self.defender_loss_fn(rho)

        loss.backward()
        self.defender_optimizer.step()  

        if DEBUG:
            print(self.defender.state_dict()["nn.0.bias"])
            # make_dot(def_policy, self.defender.named_parameters(), path=self.logging_dir)

        return loss.detach() / timesteps

    def initialize_random_batch(self, batch_size=BATCH_SIZE):
        return [next(self.model._param_generator) for _ in range(batch_size)]

    def train(self, atk_steps, def_steps, timesteps, dt):
        """ Trains both the attacker and the defender
        """
        random_batch = self.initialize_random_batch() 

        for init_state in random_batch:      

            for _ in range(atk_steps):
                self.model.reinitialize(*init_state)
                atk_loss = self.train_attacker_step(timesteps, dt)
            
            for i in range(def_steps):
                self.model.reinitialize(*init_state)
                def_loss = self.train_defender_step(timesteps, dt, i)

        return (atk_loss, def_loss)

    def run(self, n_steps, time_horizon=100, dt=0.05, *, atk_steps=1, def_steps=1, 
            atk_static=False):
        """ Trains the architecture and provides logging and visual feedback """
        if self.logging:
            hist_every = int(n_steps / 10)
            hist_counter = 0

            atk_loss_vals = torch.zeros(n_steps)
            def_loss_vals = torch.zeros(n_steps)

        for i in tqdm(range(n_steps)):
            atk_loss, def_loss = self.train(atk_steps, def_steps, time_horizon, dt)
            print(f"def_rob = {-def_loss:.4f}\tatk_rob = {atk_loss:.4f}")

            if self.logging:
                atk_loss_vals[i] = atk_loss
                def_loss_vals[i] = def_loss

        def plot_loss(atk_loss, def_loss, path):
            fig, ax = plt.subplots(1)
            ax.plot(atk_loss, label="attacker loss")
            ax.plot(def_loss, label="defender loss")
            ax.legend()
            os.makedirs(os.path.dirname(path+"/"), exist_ok=True)
            fig.savefig(path+"/loss.png")

        if self.logging:
            plot_loss(atk_loss_vals.detach().cpu(), def_loss_vals.detach().cpu(), self.logging_dir)
