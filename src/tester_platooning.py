import os
import pickle

import model_platooning
import misc
import architecture

import torch
import torch.nn as nn
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="dirname",
                    help="model's directory")
parser.add_argument("-r", "--repetitions", dest="repetitions", type=int, default=1,
                    help="simulation repetions")
args = parser.parse_args()

agent_position = 0
agent_velocity = np.linspace(0, 20, 10)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 20, 10)
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity, leader_position, leader_velocity)

physical_model = model_platooning.Model(pg.sample(sigma=0.05))

robustness_formula = 'G(dist <= 10 & dist >= 2)'
robustness_computer = model_platooning.RobustnessComputer(robustness_formula)

attacker = architecture.Attacker(physical_model, 2, 10, 2)
defender = architecture.Defender(physical_model, 2, 10)

misc.load_models(attacker, defender, args.dirname)

dt = 0.05
steps = 300

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
    for i in range(steps):
        with torch.no_grad():
            oa = torch.tensor(physical_model.agent.status)
            oe = torch.tensor(physical_model.environment.status)
            z = torch.rand(attacker.noise_size)
            if mode == 0:
                atk_policy = lambda x: torch.tensor(2.) if i > 200 and i < 250 else torch.tensor(-2.)
            elif mode == 1:
                atk_policy = lambda x: torch.tensor(2.) if i > 150 else torch.tensor(-2.)
            elif mode == 2:
                atk_policy = lambda x: torch.tensor(2.) if i < 150 else torch.tensor(-2.)
            else:
                atk_policy = attacker(torch.cat((z, oe)))
            def_policy = defender(oa)

        atk_input = atk_policy(dt)
        def_input = def_policy(dt)

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
for i in range(args.repetitions):
    sim = {}
    sim['pulse'] = run(0)
    sim['step_up'] = run(1)
    sim['step_down'] = run(2)
    sim['atk'] = run()
    
    records.append(sim)
    
with open(os.path.join(args.dirname, 'sims.pkl'), 'wb') as f:
    pickle.dump(records, f)
