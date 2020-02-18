import model
import nn_torch

import torch
import torch.nn as nn

physical_model = model.Model()

robustness_formula = 'G(dist <= 100 & dist >= 3)'
robustness_computer = model.RobustnessComputer(robustness_formula)

attacker = nn_torch.Attacker(physical_model, 1, 10, 5)
defender = nn_torch.Defender(physical_model, 1, 10)

working_dir = '/tmp/experiment'

trainer = nn_torch.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

dt = 0.05
n_steps = 10
simulation_horizon = int(1 / dt) # 5 seconds

trainer.run(n_steps, simulation_horizon, dt)
