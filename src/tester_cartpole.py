import os
import pickle
import torch
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm

from utils.misc import *
from model.cartpole import *
from settings.cartpole import *
from architecture.cartpole import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

cart_position, cart_velocity, pole_angle, pole_ang_velocity, \
    arch, train_par, test_par, robustness_formula = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="cartpole_"+args.architecture, train_params=train_par)
net_filename = get_net_filename(arch["hidden"], arch["size"])
sims_filename = get_sims_filename(args.repetitions, test_par["dt"], test_par["test_steps"])

pg = ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, pole_ang_velocity)
physical_model = Model(pg.sample())
robustness_computer = RobustnessComputer(robustness_formula)
policynetwork = PolicyNetwork(physical_model, *arch.values())

policynetwork.load_state_dict(torch.load(EXP+relpath+net_filename))

def run(mode=None):
    physical_model.initialize_random()
    conf_init = {
        'x': physical_model.cartpole.x,
        'dot_x': physical_model.cartpole.dot_x,
        'theta': physical_model.cartpole.theta,                     
        'dot_theta': physical_model.cartpole.dot_theta,
    }

    sim_t = []
    sim_x = []
    sim_theta = []
    sim_dot_x = []
    sim_ddot_x = []
    sim_dot_theta = []
    sim_action = []

    t = 0
    dt = test_par["dt"]
    for i in range(test_par["test_steps"]):
        with torch.no_grad():

            status = torch.tensor(physical_model.cartpole.status)            
            action = policynetwork(status)
            physical_model.step(action=action, dt=dt)

        sim_t.append(t)
        sim_x.append(physical_model.cartpole.x.item())
        sim_theta.append(physical_model.cartpole.theta.item())
        sim_dot_x.append(physical_model.cartpole.dot_x.item())
        sim_ddot_x.append(physical_model.cartpole.ddot_x.item())
        sim_dot_theta.append(physical_model.cartpole.dot_theta.item())
        sim_action.append(action.item())

        t += dt
        
    return {'init': conf_init,
            'sim_t': np.array(sim_t),
            'sim_x': np.array(sim_x),
            'sim_theta': np.array(sim_theta),
            'sim_dot_x': np.array(sim_dot_x),
            'sim_ddot_x': np.array(sim_dot_x),
            'sim_dot_theta': np.array(sim_dot_theta),
            'sim_action': np.array(sim_action),
    }

records = []
for i in tqdm(range(args.repetitions)):
    sim = {}
    sim['const'] = run(0)
    records.append(sim)
               
with open(EXP+relpath+sims_filename, 'wb') as f:
    pickle.dump(records, f)
