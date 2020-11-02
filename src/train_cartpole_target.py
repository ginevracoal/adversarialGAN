import os
from misc import *
import torch
import numpy as np
import torch.nn as nn
from argparse import ArgumentParser
import architecture
import model_cartpole_target

################
### SETTINGS ###
################

cart_position = np.linspace(-.1, .1, 100)
cart_velocity = np.linspace(-.5, .5, 100)
pole_angle = np.linspace(-0.2, 0.2, 100)
pole_ang_velocity = np.linspace(-.5, .5, 100)
x_target = np.linspace(-.2, .2, 100)

atk_arch = {'hidden':2, 'size':10, 'coef':1, 'noise':2}
def_arch = {'hidden':2, 'size':10, 'coef':5}
train_par = {'train_steps':1000, 'atk_steps':3, 'def_steps':5, 'horizon':5., 'dt': 0.05, 'lr':.001}

safe_theta = 0.392
safe_dist = 0.5
robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta})'# & dist <= {safe_dist})'

################

parser = ArgumentParser()
parser.add_argument("--dir", default="../experiments/cartpole_target", help="model's directory")
args = parser.parse_args()

pg = ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, 
                                        pole_ang_velocity, x_target)
physical_model = model_cartpole_target.Model(pg.sample(sigma=0.05))
robustness_computer = model_cartpole_target.RobustnessComputer(robustness_formula)

relpath = args.dir+"_lr="+str(train_par["lr"])+"_dt="+str(train_par["dt"])+\
          "_horizon="+str(train_par["horizon"])+"_train_steps="+str(train_par["train_steps"])

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