import os

import misc
import architecture
import model_platooning_energy

import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

# Specifies the initial conditions of the setup
parser = ArgumentParser()
parser.add_argument("--dir", default="../experiments/platooning_energy", help="model's directory")
parser.add_argument("--training_steps", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

# Sets the device
if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Specifies the initial conditions of the setup
agent_position = 0
agent_velocity = np.linspace(0, 20, 40)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 20, 40)
dt = 0.05 # timestep
simulation_horizon = int(5. / dt) # 5 seconds

# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity, leader_position, leader_velocity)

# Instantiates the world's model
physical_model = model_platooning_energy.Model(pg.sample(sigma=0.05), device=args.device)

# Specifies the STL formula to compute the robustness
robustness_formula = 'G(dist <= 10 & dist >= 2)'
robustness_computer = model_platooning_energy.RobustnessComputer(robustness_formula)

# Instantiates the NN architectures
attacker = architecture.Attacker(physical_model, 2, 10, 2)
defender = architecture.Defender(physical_model, 2, 10)

# Instantiates the traning and test environments
trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, args.dir)
tester = architecture.Tester(physical_model, robustness_computer, \
                            attacker, defender, args.dir)


# Starts the training
training_steps = args.training_steps # number of episodes for training
trainer.run(training_steps, tester, simulation_horizon, dt, atk_steps=1, def_steps=5)

# Saves the trained models
misc.save_models(attacker, defender, args.dir)

# Starts the testing
# test_steps = 10 # number of episodes for testing
# simulation_horizon = int(60 / dt) # 60 seconds
# tester.run(test_steps, simulation_horizon, dt)
