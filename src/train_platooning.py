import os

import misc
import architecture
import model_platooning

import torch
import torch.nn as nn
import numpy as np

agent_position = 0
agent_velocity = np.linspace(0, 5, 10)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 5, 10)
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity, leader_position, leader_velocity)

physical_model = model_platooning.Model(pg.sample(sigma=0.05))

robustness_formula = 'G(dist <= 10 & dist >= 2)'
robustness_computer = model_platooning.RobustnessComputer(robustness_formula)

attacker = architecture.Attacker(physical_model, 2, 10, 5)
defender = architecture.Defender(physical_model, 2, 10)

working_dir = '/tmp/experiment'

trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)
tester = architecture.Tester(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

dt = 0.05
training_steps = 2000
simulation_horizon = int(5 / dt) # 5 seconds

trainer.run(training_steps, simulation_horizon, dt, atk_steps=0, def_steps=10)

test_steps = 10
simulation_horizon = int(60 / dt) # 60 seconds
tester.run(test_steps, simulation_horizon, dt)

misc.save_models(attacker, defender, working_dir)
