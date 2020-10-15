import os

import misc
import architecture
import model_cartpole

import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

# Specifies the initial conditions of the setup
parser = ArgumentParser()
parser.add_argument("--dir", default="../experiments/cartpole", help="model's directory")
parser.add_argument("--training_steps", type=int, default=30)
parser.add_argument("--ode_idx", type=int, default=2)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

safe_theta = 0.392
safe_x = 2.
cart_position = np.linspace(-1., 1., 10)
cart_velocity = np.linspace(-.5, .5, 20)
pole_angle = np.linspace(-0.1, 0.1, 20)
pole_ang_velocity = np.linspace(-1., 1., 30)
dt = 0.05
simulation_horizon = int(2./dt)

# Sets the device
if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, pole_ang_velocity)

# Instantiates the world's model
physical_model = model_cartpole.Model(pg.sample(sigma=0.05), device=args.device, ode_idx=args.ode_idx)

# Specifies the STL formula to compute the robustness
# robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta})'
robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta} & x >= -{safe_x} & x <= {safe_x})'
robustness_computer = model_cartpole.RobustnessComputer(robustness_formula)

# Instantiates the NN architectures
attacker = architecture.Attacker(physical_model, 2, 10, 5)
defender = architecture.Defender(physical_model, 3, 10, 5)

working_dir = args.dir+str(args.ode_idx)

# Instantiates the traning and test environments
trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)
tester = architecture.Tester(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

# Starts the training
training_steps = args.training_steps # number of episodes for training
trainer.run(training_steps, tester, simulation_horizon, dt=dt, atk_steps=1, def_steps=5)

# Saves the trained models
misc.save_models(attacker, defender, working_dir)

# # Starts the testing
# test_steps = 10 # number of episodes for testing
# simulation_horizon = int(60 / dt) # 60 seconds
# tester.run(test_steps, simulation_horizon, dt)

