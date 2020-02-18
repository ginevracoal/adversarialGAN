import model
import nn_torch

import torch
import torch.nn as nn

physical_model = model.Model()

robustness_formula = 'G(dist <= 100 & dist >= 3)'
robustness_computer = model.RobustnessComputer(robustness_formula)

atk_layers = [
    nn.Linear(3 + 5, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 3),
]

def_layers = [
    nn.Linear(3, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 10),
    nn.LeakyReLU(),
    nn.Linear(10, 3),
]

attacker = nn_torch.NeuralNetwork(atk_layers)
defender = nn_torch.NeuralNetwork(def_layers)

working_dir = '/tmp/experiment'

trainer = nn_torch.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

dt = 0.05
n_steps = 5
simulation_horizon = int(1 / dt) # 5 seconds

trainer.run(n_steps, simulation_horizon, dt)
