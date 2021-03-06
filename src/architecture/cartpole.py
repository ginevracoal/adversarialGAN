import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from utils.print_pytorch_autograd import make_dot

DEBUG=False
BATCH_SIZE=32
NORMALIZE=False
K=10
PENALTY=False
GAMMA=0.2

torch.set_default_tensor_type(torch.DoubleTensor)


class PolicyNetwork(nn.Module):

    def __init__(self, model, n_hidden_layers, layer_size):
        super().__init__()

        assert n_hidden_layers > 0

        self.hid = n_hidden_layers
        self.ls = layer_size
        self.noise_size = None

        input_layer_size = model.cartpole.sensors
        output_layer_size = model.cartpole.actuators

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
                policy_network, lr, logging_dir=None):

        self.model = world_model
        self.robustness_computer = robustness_computer

        self.policy_network = policy_network
        
        self.loss_fn = lambda x: -x
        self.optimizer = optim.Adam(policy_network.parameters(), lr=lr)

        self.logging = True if logging_dir else False
        if self.logging:
            self.logging_dir = logging_dir

    def train_step(self, timesteps, dt):
        """ Training step for the attacker. The defender's passive. """
        self.optimizer.zero_grad()
        cumloss = 0        
        if PENALTY:    
            previous_policy = torch.zeros_like(self.policy_network(torch.tensor(self.model.cartpole.status)))

        for t in range(timesteps):
            status = torch.tensor(self.model.cartpole.status)
            action = self.policy_network(status)
            self.model.step(action, dt)

            if t>K:
                rho = self.robustness_computer.compute(self.model)

                if PENALTY:
                    diff_policy = torch.sum(torch.abs(action-previous_policy))
                    rho += GAMMA*diff_policy
                    previous_policy = action

                cumloss += self.loss_fn(rho)

        cumloss.backward()
        self.optimizer.step()
        
        if DEBUG:
            print(self.policy_network.state_dict()['nn.0.bias'])
            # make_dot(action, self.policy_network.named_parameters(), path=self.logging_dir)
            
        return cumloss.detach() / timesteps

    def initialize_random_batch(self, batch_size=BATCH_SIZE):
        random_batch = [next(self.model._param_generator) for _ in range(batch_size)]
        return random_batch

    def train(self, timesteps, dt):
        """ Trains both the attacker and the defender
        """
        random_batch = self.initialize_random_batch() 

        for init_state in random_batch:
            self.model.reinitialize(*init_state)
            loss = self.train_step(timesteps, dt)

        return loss

    def run(self, n_steps, time_horizon=100, dt=0.05):
        """ Trains the architecture and provides logging and visual feedback """

        if self.logging:
            hist_every = int(n_steps/10)
            hist_counter = 0
            loss_vals = torch.zeros(n_steps)

        for i in tqdm(range(n_steps)):
            loss = self.train(time_horizon, dt)
            print(f"rob = {-loss:.4f}")

            if self.logging:
                loss_vals[i] = loss

        def plot_loss(loss, path):
            fig, ax = plt.subplots(1)
            ax.plot(loss, label="loss")
            ax.legend()
            os.makedirs(os.path.dirname(path+"/"), exist_ok=True)
            fig.savefig(path+"/loss.png")

        if self.logging:
            plot_loss(loss_vals, self.logging_dir)

