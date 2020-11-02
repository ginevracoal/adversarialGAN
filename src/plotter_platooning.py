import os

from misc import *
import architecture
import model_platooning

import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

################
### SETTINGS ###
################

agent_position = 0
agent_velocity = np.linspace(0, 20, 100) 
leader_position = np.linspace(1, 12, 25)
leader_velocity = np.linspace(0, 20, 100)

atk_arch = {'hidden':2, 'size':10, 'coef':3, 'noise':2}
def_arch = {'hidden':2, 'size':10, 'coef':3}
train_par = {'train_steps':20000, 'atk_steps':3, 'def_steps':5, 'horizon':5., 'dt': 0.05, 'lr':1.}

robustness_formula = 'G(dist <= 10 & dist >= 2)'

################

parser = ArgumentParser()
parser.add_argument("--dir", default="platooning", help="model's directory")
args = parser.parse_args()

pg = ParametersHyperparallelepiped(agent_position, agent_velocity, leader_position, leader_velocity)
physical_model = model_platooning.Model(pg.sample(sigma=0.05))
robustness_computer = model_platooning.RobustnessComputer(robustness_formula)

relpath = args.dir+"_lr="+str(train_par["lr"])+"_dt="+str(train_par["dt"])+\
          "_horizon="+str(train_par["horizon"])+"_train_steps="+str(train_par["train_steps"])+\
          "_atk="+str(train_par["atk_steps"])+"_def="+str(train_par["def_steps"])

attacker = architecture.Attacker(physical_model, *atk_arch.values())
defender = architecture.Defender(physical_model, *def_arch.values())
trainer = architecture.Trainer(physical_model, robustness_computer, \
                            attacker, defender, train_par["lr"], EXP+relpath)
tester = architecture.Tester(physical_model, robustness_computer, \
                            attacker, defender, EXP+args.dir)

simulation_horizon = int(train_par["horizon"] / train_par["dt"])
trainer.run(train_par["train_steps"], simulation_horizon, train_par["dt"], 
            atk_steps=train_par["atk_steps"], def_steps=train_par["def_steps"])

save_models(attacker, defender, EXP+relpath)