import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt


torch.set_default_tensor_type(torch.DoubleTensor)


class Attacker(nn.Module):
    """ NN architecture for the attacker """
    def __init__(self, model, n_hidden_layers, layer_size, n_coeff, noise_size):
        super().__init__()

        assert n_hidden_layers > 0

        self.hid = n_hidden_layers
        self.ls = layer_size
        self.noise_size = noise_size
        self.n_coeff = n_coeff

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
        """ Uses the NN's output to compute the coefficients of the policy function """
        coefficients = self.nn(x)
        coefficients = torch.reshape(coefficients, (-1, self.n_coeff))

        def policy_generator(t):
            """ The policy function is defined as polynomial """
            basis = [t**i for i in range(self.n_coeff)]
            basis = torch.tensor(basis, dtype=torch.get_default_dtype())
            basis = torch.reshape(basis, (self.n_coeff, -1))
            return coefficients.mm(basis).squeeze()

        return policy_generator


class Defender(nn.Module):
    """ NN architecture for the defender """

    def __init__(self, model, n_hidden_layers, layer_size, n_coeff):
        super().__init__()

        assert n_hidden_layers > 0

        self.hid = n_hidden_layers
        self.ls = layer_size
        self.n_coeff = n_coeff

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
        """ Uses the NN's output to compute the coefficients of the policy function """
        coefficients = self.nn(x)
        coefficients = torch.reshape(coefficients, (-1, self.n_coeff))

        def policy_generator(t):
            """ The policy function is defined as polynomial """
            basis = [t**i for i in range(self.n_coeff)]
            basis = torch.tensor(basis, dtype=torch.get_default_dtype())
            basis = torch.reshape(basis, (self.n_coeff, -1))
            return coefficients.mm(basis).squeeze()

        return policy_generator



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

        atk_optimizer = optim.Adam(attacker_nn.parameters(), lr=lr)
        def_optimizer = optim.Adam(defender_nn.parameters(), lr=lr)
        self.attacker_optimizer = atk_optimizer
        self.defender_optimizer = def_optimizer

        self.logging = True if logging_dir else False

        if self.logging:
            # self.log = SummaryWriter(logging_dir)
            self.logging_dir = logging_dir


    def train_attacker_step(self, time_horizon, dt, atk_static):
        """ Training step for the attacker. The defender's passive. """
        z = torch.rand(self.attacker.noise_size)
        oa = torch.tensor(self.model.agent.status)
        oe = torch.tensor(self.model.environment.status)

        atk_policy = self.attacker(torch.cat((z, oe)))

        with torch.no_grad():
            def_policy = self.defender(oa)

        t = 0
        for i in range(time_horizon):
            # if the attacker is static (e.g. in the case it does not vary over time)
            # the policy function is always sampled in the same point since the
            # attacker do not vary policy over time
            atk_input = atk_policy(0 if atk_static else t)
            def_input = def_policy(t)

            self.model.step(atk_input, def_input, dt)

            t += dt

        rho = self.robustness_computer.compute(self.model)

        self.attacker_optimizer.zero_grad()

        loss = self.attacker_loss_fn(rho)
        loss.backward()

        self.attacker_optimizer.step()

        return float(loss.detach())


    def train_defender_step(self, time_horizon, dt, atk_static):
        """ Training step for the defender. The attacker's passive. """
        z = torch.rand(self.attacker.noise_size)
        oa = torch.tensor(self.model.agent.status)
        oe = torch.tensor(self.model.environment.status)

        with torch.no_grad():
            atk_policy = self.attacker(torch.cat((z, oe)))

        def_policy = self.defender(oa)

        t = 0
        for i in range(time_horizon):
            # if the attacker is static, see the comments above
            atk_input = atk_policy(0 if atk_static else t)
            def_input = def_policy(t)

            self.model.step(atk_input, def_input, dt)

            t += dt

        rho = self.robustness_computer.compute(self.model)

        self.defender_optimizer.zero_grad()

        loss = self.defender_loss_fn(rho)
        loss.backward()

        self.defender_optimizer.step()

        return float(loss.detach())

    # def initialize_random_batch(self, batch_size=128):
    #     return [next(self.model._param_generator) for _ in range(batch_size)]

    def train(self, atk_steps, def_steps, time_horizon, dt, atk_static):
        """ Trains both the attacker and the defender on the same
            initial senario (different for each)
        """

        atk_loss, def_loss = 0, 0

        self.model.initialize_random() # samples a random initial state
        for i in range(atk_steps):
            atk_loss = self.train_attacker_step(time_horizon, dt, atk_static)
            self.model.initialize_rewind() # restores the initial state

        self.model.initialize_random() # samples a random initial state
        for i in range(def_steps):
            def_loss = self.train_defender_step(time_horizon, dt, atk_static)
            self.model.initialize_rewind() # restores the initial state

        return (atk_loss, def_loss)


    def run(self, n_steps, time_horizon=100, dt=0.05, *, atk_steps=1, def_steps=1, atk_static=False):
        """ Trains the architecture and provides logging and visual feedback """
        if self.logging:
            hist_every = int(n_steps / 10)
            hist_counter = 0

            atk_loss_vals = torch.zeros(n_steps)
            def_loss_vals = torch.zeros(n_steps)

        for i in tqdm(range(n_steps)):
            atk_loss, def_loss = self.train(atk_steps, def_steps, time_horizon, dt, atk_static)
            print(f"def_rob = {-def_loss:.4f}\tatk_rob = {atk_loss:.4f}")

            if self.logging:
                atk_loss_vals[i] = atk_loss
                def_loss_vals[i] = def_loss

                # self.log.add_scalar('attacker loss', atk_loss, i)
                # self.log.add_scalar('defender loss', def_loss, i)

                # if (i + 1) % hist_every == 0:
                #     a = hist_counter * hist_every
                #     b = (hist_counter + 1) * hist_every
                #     hist_counter += 1

                #     self.log.add_histogram('attacker loss hist', atk_loss_vals[a:b], i)
                #     self.log.add_histogram('defender loss hist', def_loss_vals[a:b], i)

        def plot_loss(atk_loss, def_loss, path):
            fig, ax = plt.subplots(1)
            ax.plot(atk_loss, label="attacker loss")
            ax.plot(def_loss, label="defender loss")
            ax.legend()
            os.makedirs(os.path.dirname(path+"/"), exist_ok=True)
            fig.savefig(path+"/loss.png")

        if self.logging:
            # self.log.close()
            plot_loss(atk_loss_vals.detach().cpu(), def_loss_vals.detach().cpu(), self.logging_dir)



class Tester:
    """ The class contains the testing logic """

    def __init__(self, world_model, robustness_computer, \
                attacker_nn, defender_nn, logging_dir=None):

        self.model = world_model
        self.robustness_computer = robustness_computer

        self.attacker = attacker_nn
        self.defender = defender_nn

        self.logging = True if logging_dir else False

        # if self.logging:
        #     self.log = SummaryWriter(logging_dir)

    def test(self, time_horizon, dt):
        """ Tests a whole episode """
        self.model.initialize_random()

        for t in range(time_horizon):
            z = torch.rand(self.attacker.noise_size)
            oa = torch.tensor(self.model.agent.status)
            oe = torch.tensor(self.model.environment.status)

            with torch.no_grad():
                atk_policy = self.attacker(torch.cat((z, oe)))
                def_policy = self.defender(oa)

            atk_input = atk_policy(dt)
            def_input = def_policy(dt)

            self.model.step(atk_input, def_input, dt)

        rho = self.robustness_computer.compute(self.model)

        return rho


    def run(self, times, time_horizon=1000, dt=0.05):
        """ Test the architecture and provides logging """
        if self.logging:
            def_rho_vals = torch.zeros(times)

        for i in tqdm(range(times)):
            def_rho = self.test(time_horizon, dt)

            if self.logging:
                def_rho_vals[i] = def_rho

        # if self.logging:
        #     self.log.add_histogram('defender robustness', def_rho_vals, i)
        #     self.log.close()

        print(f"avg robustness = {def_rho_vals.mean().item():.2f}")
