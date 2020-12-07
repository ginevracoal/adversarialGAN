import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

from utils.misc import *
from model.platooning_energy import *
from settings.platooning_energy import *
from architecture.default import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

agent_position, agent_velocity, leader_position, leader_velocity, \
        atk_arch, def_arch, train_par, test_par, \
        robustness_dist, robustness_power = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="platooning_energy_"+args.architecture, train_params=train_par)
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
    sim_ag_power = []
    sim_ag_pos = []
    sim_ag_dist = []
    sim_env_pos = []
    sim_ag_e_torque = []
    sim_ag_br_torque = []
    sim_env_e_torque = []
    sim_env_br_torque = []

    t = 0
    dt = test_par["dt"]
    for i in range(test_par["test_steps"]):
        with torch.no_grad():
            oa = torch.tensor(physical_model.agent.status)
            oe = torch.tensor(physical_model.environment.status)
            z = torch.rand(attacker.noise_size)

            if mode == 0:
                atk_policy = torch.tensor(.5) if i > int(test_par["test_steps"]*1/3) \
                             and i < int(test_par["test_steps"]*2/3) else torch.tensor(-.5)
            elif mode == 1:
                atk_policy = torch.tensor(.5) if i > int(test_par["test_steps"]/2) else torch.tensor(-.5)
            elif mode == 2:
                atk_policy = torch.tensor(.5) if i < int(test_par["test_steps"]/2) else torch.tensor(-.5)
            else:
                atk_policy = attacker(torch.cat((z, oe)))
                
            def_policy = defender(oa)

        physical_model.step(atk_policy, def_policy, dt)
        sim_t.append(t)
        sim_ag_pos.append(physical_model.agent.position)
        sim_env_pos.append(physical_model.environment.l_position)
        sim_ag_dist.append(physical_model.agent.distance)
        sim_ag_power.append(physical_model.agent.e_power)

        sim_ag_e_torque.append(def_policy[0])
        sim_ag_br_torque.append(def_policy[1])
        sim_env_e_torque.append(atk_policy[0])
        sim_env_br_torque.append(atk_policy[1])

        t += dt
        
    return {'init': conf_init,
            'sim_t': np.array(sim_t),
            'sim_ag_pos': np.array(sim_ag_pos),
            'sim_ag_dist': np.array(sim_ag_dist),
            'sim_ag_power': np.array(sim_ag_power),
            'sim_env_pos': np.array(sim_env_pos),
            'sim_ag_e_torque': np.array(sim_ag_e_torque),
            'sim_ag_br_torque': np.array(sim_ag_br_torque),
            'sim_env_e_torque': np.array(sim_env_e_torque),
            'sim_env_br_torque': np.array(sim_env_br_torque),
    }

records = []
for i in tqdm(range(args.repetitions)):
    sim = {}
    # sim['pulse'] = run(0)
    # sim['step_up'] = run(1)
    # sim['step_down'] = run(2)
    sim['atk'] = run()
    records.append(sim)
               
with open(os.path.join(EXP+relpath, sims_filename), 'wb') as f:
    pickle.dump(records, f)
