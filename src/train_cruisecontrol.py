import os

import misc
import architecture
import model_cruisecontrol

import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser


# Specifies the initial conditions of the setup
parser = ArgumentParser()
parser.add_argument("--dir", default="../experiments/cruisecontrol", help="model's directory")
parser.add_argument("--training_steps", type=int, default=10)
parser.add_argument("--ode_idx", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

# Sets the device
if args.device=="cuda":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# Specifies the initial conditions of the setup
agent_position = np.arange(model_cruisecontrol.ROAD_LENGTH)
agent_velocity = np.linspace(-12, 12, 25)
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity)

# Instantiates the world's model
physical_model = model_cruisecontrol.Model(pg.sample(sigma=0.05), device=args.device)

# Specifies the STL formula to compute the robustness
robustness_formula = 'G(v >= 4.75 & v <= 5.25)'
robustness_computer = model_cruisecontrol.RobustnessComputer(robustness_formula)

# Instantiates the NN architectures
attacker = architecture.Attacker(physical_model, 1, 10, 5, n_coeff=1)
defender = architecture.Defender(physical_model, 2, 10)

# Instantiates the traning environment
trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, args.dir)

dt = 0.05 # timestep
training_steps = 300000 # number of episodes for training
simulation_horizon = int(0.5 / dt) # 0.5 second

# Starts the training
trainer.run(training_steps, simulation_horizon, dt, atk_steps=1, def_steps=10, atk_static=True)

# Saves the trained models
misc.save_models(attacker, defender, args.dir)
