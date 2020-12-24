import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

from utils.misc import *
from model.platooning_energy import *
from model.platooning_energy_classic import *
from settings.platooning_energy import *
from architecture.platooning_energy import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

agent_position, agent_velocity, leader_position, leader_velocity, \
            atk_arch, def_arch, train_par, test_par, robustness_dist, robustness_power, \
            safe_dist_lower, safe_dist_upper, safe_power, alpha = get_settings(args.architecture, mode="test")
            
relpath = get_relpath(main_dir="platooning_energy_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

pg = ParametersHyperparallelepiped(agent_position, agent_velocity, 
                                    leader_position, leader_velocity)

model = Model(pg.sample())
model_classic = Model_classic(pg.sample())

attacker = Attacker(model, *atk_arch.values())
defender = Defender(model, *def_arch.values())
load_models(attacker, defender, EXP+relpath)


def fixed_leader(t):
    norm_e_torque = torch.tanh(t-4)*0.2
    norm_br_torque = torch.sigmoid(-4*torch.tanh(t/2))
    return norm_e_torque, norm_br_torque

def run(random_init, mode=None, classic_control=False):
    
    if classic_control is True:
        physical_model = model_classic
    else:
        physical_model = model

    physical_model.reinitialize(*random_init)

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
                atk_policy = (torch.tensor(0.0), torch.tensor(0.0))
            elif mode == 1:
                atk_policy = fixed_leader(torch.tensor(t).double())
            else:
                atk_policy = attacker(torch.cat((z, oe)))

            if classic_control is True:
                def_policy = physical_model.agent._car.get_controller_input(dt, physical_model.agent.distance)
            else:
                def_policy = defender(oa)             
        
        physical_model.step(atk_policy, def_policy, dt)

        ag_e_torque, ag_br_torque, _ = physical_model.agent._car.calculate_wheels_torque(*def_policy)
        env_e_torque, env_br_torque, _ = physical_model.environment._leader_car.calculate_wheels_torque(*atk_policy)
        
        sim_t.append(t)
        sim_ag_pos.append(physical_model.agent.position)
        sim_env_pos.append(physical_model.environment.l_position)
        sim_ag_dist.append(physical_model.agent.distance)
        sim_ag_power.append(physical_model.agent.e_power)
        sim_ag_e_torque.append(ag_e_torque)
        sim_ag_br_torque.append(ag_br_torque)
        sim_env_e_torque.append(env_e_torque)
        sim_env_br_torque.append(env_br_torque)

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

    random_init = next(model._param_generator)
    sim['const'] = run(random_init, mode=0)
    sim['classic_const'] = run(random_init, mode=0, classic_control=True)

    random_init = next(model._param_generator)
    sim['pulse'] = run(random_init, mode=1)
    sim['classic_pulse'] = run(random_init, mode=1, classic_control=True)

    random_init = next(model._param_generator)
    sim['atk'] = run(random_init)
    sim['classic_atk'] = run(random_init, classic_control=True)

    records.append(sim)
               
with open(os.path.join(EXP+relpath, sims_filename), 'wb') as f:
    pickle.dump(records, f)
