import os

import misc
import architecture
import model_platooning

import torch
import torch.nn as nn
import numpy as np

# Specifies the initial conditions of the setup
agent_position = 0
agent_velocity = np.linspace(0, 20, 40)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 20, 40)
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity, leader_position, leader_velocity)

# Instantiates the world's model
physical_model = model_platooning.Model(pg.sample(sigma=0.05))

# Specifies the STL formula to compute the robustness
robustness_formula = 'G(dist <= 10 & dist >= 2)'
robustness_computer = model_platooning.RobustnessComputer(robustness_formula)

# Instantiates the NN architectures
attacker = architecture.Attacker(physical_model, 2, 10, 2)
defender = architecture.Defender(physical_model, 2, 10)

working_dir = '/tmp/experiment'

# Instantiates the traning and test environments
trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)
tester = architecture.Tester(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

dt = 0.05 # timestep
training_steps = 10000 # number of episodes for training
simulation_horizon = int(5 / dt) # 5 seconds

# Starts the training
trainer.run(training_steps, simulation_horizon, dt, atk_steps=3, def_steps=5)

test_steps = 10 # number of episodes for testing
simulation_horizon = int(60 / dt) # 60 seconds

# Starts the testing
tester.run(test_steps, simulation_horizon, dt)

# Saves the trained models
misc.save_models(attacker, defender, working_dir)
