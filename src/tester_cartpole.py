import os
import pickle

import model_cartpole
import misc
import architecture

import torch
import torch.nn as nn
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", default="../experiments/cartpole", dest="dirname",
                    help="model's directory")
parser.add_argument("-r", "--repetitions", dest="repetitions", type=int, default=1,
                    help="simulation repetions")
args = parser.parse_args()

cart_position = 0
cart_velocity = np.linspace(0, 10, 20)
pole_position = np.linspace(1, 12, 15)
pole_velocity = np.linspace(0, 20, 40)
pg = misc.ParametersHyperparallelepiped(cart_position, cart_velocity, pole_position, pole_velocity)

physical_model = model_cartpole.Model(pg.sample(sigma=0.05))

attacker = architecture.Attacker(physical_model, 2, 10, 2)
defender = architecture.Defender(physical_model, 2, 10)

misc.load_models(attacker, defender, args.dirname)

dt = 0.05
steps = 300

def run(mode=None):
    physical_model.initialize_random()
    conf_init = {
        'c_pos': physical_model.agent.c_position,
        'c_vel': physical_model.agent.c_velocity,
        'p_pos': physical_model.agent.p_angle,                     
        'p_vel': physical_model.agent.p_velocity,
    }

    sim_t = []
    sim_c_pos = []
    sim_p_ang = []
    sim_c_acc = []
    sim_p_acc = []

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

        sim_t.append(t)
        sim_c_pos.append(physical_model.agent.c_position)
        sim_p_ang.append(physical_model.agent.p_angle)
        sim_c_acc.append(def_input)
        sim_p_acc.append(atk_input)

        t += dt
        
    return {'init': conf_init,
            'sim_t': np.array(sim_t),
            'sim_c_pos': np.array(sim_c_pos),
            'sim_p_ang': np.array(sim_p_ang),
            'sim_c_acc': np.array(sim_c_acc),
            'sim_p_acc': np.array(sim_p_acc),
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
