import os

import misc
import architecture
import model_cruisecontrol

import torch
import torch.nn as nn
import numpy as np

agent_position = np.arange(50)
agent_velocity = np.linspace(-12, 12, 25)
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity)

physical_model = model_cruisecontrol.Model(pg.sample(sigma=0.05))

robustness_formula = 'G(v <= 4.90 & v >= 5.10)'
robustness_computer = model_cruisecontrol.RobustnessComputer(robustness_formula)

attacker = architecture.Attacker(physical_model, 1, 10, 5, n_coeff=1)
defender = architecture.Defender(physical_model, 2, 10)

working_dir = '/tmp/experiment_cruise'

trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

dt = 0.05
training_steps = 10000
simulation_horizon = int(5 / dt) # 5 second

trainer.run(training_steps, simulation_horizon, dt, atk_steps=1, def_steps=5, atk_static=True)

misc.save_models(attacker, defender, working_dir)
