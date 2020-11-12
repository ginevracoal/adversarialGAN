import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

from misc import *
from architecture.default import *
from model.cartpole_target import *
from settings.cartpole_target import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=10, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
        atk_arch, def_arch, train_par, test_par, \
        robustness_theta, robustness_dist = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="cartpole_target_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

pg = ParametersHyperparallelepiped(cart_position, cart_velocity, 
                                        pole_angle, pole_ang_velocity, x_target)

physical_model = Model(pg.sample(sigma=0.05))
attacker = Attacker(physical_model, *atk_arch.values())
defender = Defender(physical_model, *def_arch.values())

load_models(attacker, defender, EXP+relpath)

def run(mode=None):
    physical_model.initialize_random()
    conf_init = {
        'x': physical_model.agent.x,
        'dot_x': physical_model.agent.dot_x,
        'theta': physical_model.agent.theta,                     
        'dot_theta': physical_model.agent.dot_theta,
        'dist': physical_model.agent.dist
    }

    sim_t = []
    sim_x = []
    sim_theta = []
    sim_dot_x = []
    sim_ddot_x = []
    sim_dot_theta = []
    sim_x_target = []
    sim_attack_mu = []
    sim_def_acc = []
    sim_dist = []

    t = 0
    dt = test_par["dt"]
    for i in range(test_par["test_steps"]):
        with torch.no_grad():

            oa = torch.tensor(physical_model.agent.status)
            oe = torch.tensor(physical_model.environment.status)
            z = torch.rand(attacker.noise_size)
            
            if mode == 0:
                atk_policy = lambda x: (torch.tensor(0.0), torch.tensor(0.0))

            elif mode == 1:
                atk_policy = lambda x: (-torch.tensor(0.5), -torch.tensor(0.5)) \
                            if i > test_par["test_steps"]*1/3 and i < test_par["test_steps"]*2/3 \
                            else (torch.tensor(0.5), torch.tensor(0.5))
            else:
                atk_policy = attacker(torch.cat((z, oe)))

            def_policy = defender(oa)

        atk_input = atk_policy(dt)
        def_input = def_policy(dt)

        physical_model.step(env_input=atk_input, agent_input=def_input, dt=dt)

        sim_t.append(t)
        sim_x.append(physical_model.agent.x.item())
        sim_theta.append(physical_model.agent.theta.item())
        sim_dot_x.append(physical_model.agent.dot_x.item())
        sim_ddot_x.append(physical_model.agent.ddot_x.item())
        sim_dot_theta.append(physical_model.agent.dot_theta.item())
        sim_x_target.append(physical_model.agent.x_target.item())
        sim_dist.append(physical_model.agent.dist.item())
        sim_attack_mu.append(atk_input[1].item())
        sim_def_acc.append(def_input.item())

        t += dt
        
    return {'init': conf_init,
            'sim_t': np.array(sim_t),
            'sim_x': np.array(sim_x),
            'sim_theta': np.array(sim_theta),
            'sim_dot_x': np.array(sim_dot_x),
            'sim_ddot_x': np.array(sim_dot_x),
            'sim_dot_theta': np.array(sim_dot_theta),
            'sim_x_target': np.array(sim_x_target),
            'sim_dist': np.array(sim_dist),
            'sim_attack_mu': np.array(sim_attack_mu),
            'sim_def_acc': np.array(sim_def_acc),
    }

records = []
for i in tqdm(range(args.repetitions)):
    sim = {}
    sim['const'] = run(0)
    sim['pulse'] = run(1)
    sim['atk'] = run()
    records.append(sim)
               
with open(os.path.join(EXP+relpath, sims_filename), 'wb') as f:
    pickle.dump(records, f)
