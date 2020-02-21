import model
import nn_torch

import torch
import torch.nn as nn

physical_model = model.Model()

robustness_formula = 'G(dist <= 100 & dist >= 3)'
robustness_computer = model.RobustnessComputer(robustness_formula)

attacker = nn_torch.Attacker(physical_model, 2, 10, 5)
defender = nn_torch.Defender(physical_model, 2, 10)

working_dir = '/tmp/experiment'

trainer = nn_torch.Trainer(physical_model, robustness_computer, \
                            attacker, defender, working_dir)
tester = nn_torch.Tester(physical_model, robustness_computer, \
                            attacker, defender, working_dir)

dt = 0.05
training_steps = 100
simulation_horizon = int(5 / dt) # 5 seconds

trainer.run(training_steps, simulation_horizon, dt)

test_steps = 10
simulation_horizon = int(60 / dt) # 60 seconds
tester.run(test_steps, simulation_horizon, dt)
