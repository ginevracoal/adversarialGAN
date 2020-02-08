import os
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        self.input_size = layers[0].in_features
        self.output_size = layers[-1].out_features
        
        self.nn = nn.Sequential(*layers)
        self.nn.double()

    def forward(self, x):
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        return self.nn(x)


class Trainer:
    def __init__(self, world_model, robustness_computer, \
                attacker_nn, defender_nn, \
                attacker_loss_fn, defender_loss_fn, \
                attacker_optimizer, defender_optimizer):

        self.model = world_model
        self.robustness_computer = robustness_computer

        self.attacker = attacker_nn
        self.defender = defender_nn
        self.attacker_loss_fn = attacker_loss_fn
        self.defender_loss_fn = defender_loss_fn
        self.attacker_optimizer = attacker_optimizer
        self.defender_optimizer = defender_optimizer

    def train_step(self, dataset, simulation_horizon, dt):
        atk_y, def_x, def_y = dataset

        atk_x = torch.rand(self.attacker.input_size)
        atk_y = torch.from_numpy(atk_y)
        def_x = torch.from_numpy(def_x)
        def_y = torch.from_numpy(def_y)

        self.attacker_optimizer.zero_grad()
        self.defender_optimizer.zero_grad()

        atk_output = self.attacker(atk_x)
        def_output = self.defender(def_x)

        self.attacker_loss_fn(atk_output, atk_y)
        self.defender_loss_fn(def_output, def_y)

        self.attacker_optimizer.step()
        self.defender_optimizer.step()

    def simulate(self, atk_output, def_output, simulation_horizon, dt):
        # project into the future (consider derivative in future)
        atk_constant = np.ones((simulation_horizon, len(atk_output)))
        atk_commands = atk_output * atk_constant
        def_constant = np.ones((simulation_horizon, len(def_output)))
        def_commands = def_output * def_constant

        _ = [self.model.step(atk_move, def_move, dt)
            for atk_move, def_move in zip(atk_commands, def_commands)]

        return self.robustness_computer.compute(self.model, -simulation_horizon)

    def generate_dataset(self, n_episodes, p_best, simulation_horizon, dt):
        atk_y = np.zeros((n_episodes, self.attacker.output_size))
        def_x = np.zeros((n_episodes, self.defender.input_size))
        def_y = np.zeros((n_episodes, self.defender.output_size))
        rho = np.zeros(n_episodes)

        base_config = self.model.save()

        for i in range(n_episodes):
            with torch.no_grad():
                atk_input = torch.rand(self.attacker.input_size)
                def_input = torch.from_numpy(self.model.agent.get_status())

                atk_output = self.attacker(atk_input)
                def_output = self.defender(def_input)

                atk_y[i] = atk_output
                def_x[i] = def_input
                def_y[i] = def_output
                rho[i] = self.simulate(atk_output, def_output, simulation_horizon, dt)

            self.model.restore(base_config)

        worst_idx = np.argsort(rho)[:p_best]
        best_idx = np.argsort(rho)[-p_best:]
        return (atk_y[worst_idx], def_x[best_idx], def_y[best_idx])

    def train(self, n_steps, n_episodes, p_best, simulation_horizon, dt):
        # generazione ambiente iniziale
        for i in range(n_steps):

            dataset = self.generate_dataset(n_episodes, p_best, simulation_horizon, dt)

            # Train for possible futures
            for d in zip(*dataset):
                self.train_step(d, simulation_horizon, dt)

            atk_input = self.model.environment.get_status()
            def_input = self.model.agent.get_status()

            with torch.no_grad():
                atk_output = self.attacker(torch.from_numpy(atk_input))
                def_output = self.defender(torch.from_numpy(def_input))

                # Applies the choice on the physical model
                self.model.step(atk_output, def_output, dt)

    def test_step(self, dt):
        with torch.no_grad():
            atk_input = torch.rand(self.attacker.input_size)
            def_input = torch.from_numpy(self.model.agent.get_status())

            atk_output = self.attacker(atk_input)
            def_output = self.defender(def_input)

            self.model.step(atk_output, def_output, dt)

    def test(self, n_steps, dt):
        for _ in range(n_steps):
            self.test_step(dt)

    def run(self, n_epochs, n_steps, n_episodes, p_best, simulation_horizon=100, dt=0.05):
        initial_config = self.model.save()

        for i in tqdm(range(n_epochs)):
            self.train(n_steps, n_episodes, p_best, simulation_horizon, dt)

            self.model.restore(initial_config)

        self.test(n_steps, dt)
        test_rho = self.robustness_computer.compute(self.model)
        print('Robustness during test:', test_rho)
