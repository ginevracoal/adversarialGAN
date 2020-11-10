import os
from misc import *
import torch
import torch.nn as nn
from argparse import ArgumentParser
# import architecture_dqn_cartpole as architecture
# import model_dqn_cartpole
# from settings_cartpole import get_settings
from model.dqn_cartpole import *
from settings.dqn_cartpole import *
from architecture.dqn_cartpole import *

parser = ArgumentParser()
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()


cart_position, cart_velocity, pole_angle, pole_ang_velocity, \
    arch, train_par, test_par, robustness_formula = get_settings(args.architecture, mode="train")
relpath = get_relpath(main_dir="cartpole_dqn_"+args.architecture, train_params=train_par)
net_filename = get_net_filename(arch["hidden"], arch["size"])

pg = ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, pole_ang_velocity)
physical_model = Model(pg.sample(sigma=0.05))
robustness_computer = RobustnessComputer(robustness_formula)

policynetwork = PolicyNetwork(physical_model, *arch.values())
trainer = Trainer(physical_model, robustness_computer, policynetwork, train_par["lr"], EXP+relpath)

simulation_horizon = int(train_par["horizon"] / train_par["dt"])
trainer.run(train_par["train_steps"], simulation_horizon, train_par["dt"])

os.makedirs(EXP+relpath, exist_ok=True)
torch.save(policynetwork.state_dict(), EXP+relpath+net_filename)