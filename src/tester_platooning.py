import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

from misc import *
from model.platooning import *
from settings.platooning import *
from architecture.default import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=10, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

agent_position, agent_velocity, leader_position, leader_velocity, \
        atk_arch, def_arch, train_par, test_par, \
        robustness_formula = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="platooning_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

pg = ParametersHyperparallelepiped(agent_position, agent_velocity, 
                                    leader_position, leader_velocity)

physical_model = Model(pg.sample(sigma=0.05))

attacker = Attacker(physical_model, *atk_arch.values())
defender = Defender(physical_model, *def_arch.values())

load_models(attacker, defender, EXP+relpath)

def run(mode=None):
    physical_model.initialize_random()
    conf_init = {
        'ag_pos': physical_model.agent.position,
        'ag_vel': physical_model.agent.velocity,
        'env_pos': physical_model.environment.l_position,                     
        'env_vel': physical_model.environment.l_velocity,
    }

    sim_t = []
    sim_ag_pos = []
    sim_ag_dist = []
    sim_ag_acc = []
    sim_env_pos = []
    sim_env_acc = []

    t = 0
    dt = test_par["dt"]
    for i in range(test_par["test_steps"]):
        with torch.no_grad():
            oa = torch.tensor(physical_model.agent.status)
            oe = torch.tensor(physical_model.environment.status)
            z = torch.rand(attacker.noise_size)

            if mode == 0:
                atk_policy = torch.tensor(1.) if i > int(test_par["test_steps"]*1/3) \
                             and i < int(test_par["test_steps"]*2/3) else torch.tensor(-1.)
            elif mode == 1:
                atk_policy = torch.tensor(1.) if i > int(test_par["test_steps"]/2) else torch.tensor(-1.)
            elif mode == 2:
                atk_policy = torch.tensor(1.) if i < int(test_par["test_steps"]/2) else torch.tensor(-1.)
            else:
                atk_policy = attacker(torch.cat((z, oe)))
                
            def_policy = defender(oa)

        atk_input = atk_policy
        def_input = def_policy

        physical_model.step(atk_input, def_input, dt)
        sim_ag_acc.append(def_input)
        sim_env_acc.append(atk_input)
        sim_t.append(t)
        sim_ag_pos.append(physical_model.agent.position)
        sim_env_pos.append(physical_model.environment.l_position)
        sim_ag_dist.append(physical_model.agent.distance)

        t += dt
        
    return {'init': conf_init,
            'sim_t': np.array(sim_t),
            'sim_ag_pos': np.array(sim_ag_pos),
            'sim_ag_dist': np.array(sim_ag_dist),
            'sim_ag_acc': np.array(sim_ag_acc),
            'sim_env_pos': np.array(sim_env_pos),
            'sim_env_acc': np.array(sim_env_acc),
    }

records = []
for i in tqdm(range(args.repetitions)):
    sim = {}
    sim['pulse'] = run(0)
    sim['step_up'] = run(1)
    sim['step_down'] = run(2)
    sim['atk'] = run()
    records.append(sim)
               
with open(os.path.join(EXP+relpath, sims_filename), 'wb') as f:
    pickle.dump(records, f)
