import os

import misc
import architecture_cartpole as architecture
import model_cartpole_target

import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

# Specifies the initial conditions of the setup
parser = ArgumentParser()
parser.add_argument("--dir", default="../experiments/cartpole_target", help="model's directory")
parser.add_argument("--training_steps", type=int, default=30)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

safe_theta = 0.392
safe_dist = 0.5
cart_position = np.linspace(-.1, .1, 10)
cart_velocity = np.linspace(-.5, .5, 40)
pole_angle = np.linspace(-0.1, 0.1, 40)
pole_ang_velocity = np.linspace(-.5, .5, 40)
x_target = np.linspace(-.2, .2, 40)
dt = 0.05
simulation_horizon = int(2./dt) 

# Sets the device
if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, 
                                        pole_ang_velocity, x_target)

# Instantiates the world's model
physical_model = model_cartpole_target.Model(pg.sample(sigma=0.05), device=args.device)

# Specifies the STL formula to compute the robustness
robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta} & dist <= {safe_dist})'
robustness_computer = model_cartpole_target.RobustnessComputer(robustness_formula)

# Instantiates the NN architectures
attacker = architecture.Attacker(physical_model, n_hidden_layers=2, layer_size=10, noise_size=3)
defender = architecture.Defender(physical_model, n_hidden_layers=2, layer_size=10)

working_dir = args.dir

# Instantiates the traning and test environments
trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)
tester = architecture.Tester(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

# Starts the training
training_steps = args.training_steps # number of episodes for training
trainer.run(training_steps, tester, simulation_horizon, dt=dt, atk_steps=1, def_steps=1)

# Saves the trained models
misc.save_models(attacker, defender, working_dir)

# # Starts the testing
# test_steps = 10 # number of episodes for testing
# simulation_horizon = int(60 / dt) # 60 seconds
# tester.run(test_steps, simulation_horizon, dt)

