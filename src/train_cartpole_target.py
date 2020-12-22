import os
from argparse import ArgumentParser

from utils.misc import *
from model.cartpole_target import *
from settings.cartpole_target import *
from architecture.cartpole_target import *

parser = ArgumentParser()
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
        atk_arch, def_arch, train_par, test_par, \
        robustness_theta, robustness_dist, \
        alpha, safe_theta, safe_dist = get_settings(args.architecture, mode="train")
relpath = get_relpath(main_dir="cartpole_target_"+args.architecture, train_params=train_par)

pg = ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, 
                                        pole_ang_velocity, x_target)
physical_model = Model(pg.sample())
robustness_computer = RobustnessComputer(robustness_theta, robustness_dist, alpha)

attacker = Attacker(physical_model, *atk_arch.values())
defender = Defender(physical_model, *def_arch.values())
trainer = Trainer(physical_model, robustness_computer, attacker, defender, train_par["lr"], EXP+relpath)

simulation_horizon = int(train_par["horizon"] / train_par["dt"])
trainer.run(train_par["train_steps"], simulation_horizon, train_par["dt"], 
            atk_steps=train_par["atk_steps"], def_steps=train_par["def_steps"])

save_models(attacker, defender, EXP+relpath)