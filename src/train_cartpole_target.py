import os
from misc import *
import torch
import numpy as np
import torch.nn as nn
from argparse import ArgumentParser
import architecture
import model_cartpole_target
from settings_cartpole_target import get_settings

parser = ArgumentParser()
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
        atk_arch, def_arch, train_par, test_par, \
        robustness_formula = get_settings(args.architecture, mode="train")

pg = ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, 
                                        pole_ang_velocity, x_target)
physical_model = model_cartpole_target.Model(pg.sample(sigma=0.05))
robustness_computer = model_cartpole_target.RobustnessComputer(robustness_formula)

relpath = get_relpath(main_dir="cartpole_target_"+args.architecture, train_params=train_par)

attacker = architecture.Attacker(physical_model, *atk_arch.values())
defender = architecture.Defender(physical_model, *def_arch.values())
trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, train_par["lr"], EXP+relpath)
tester = architecture.Tester(physical_model, robustness_computer, \
                            attacker, defender, EXP+relpath)

simulation_horizon = int(train_par["horizon"] / train_par["dt"])
trainer.run(train_par["train_steps"], simulation_horizon, train_par["dt"], 
            atk_steps=train_par["atk_steps"], def_steps=train_par["def_steps"])

save_models(attacker, defender, EXP+relpath)