import os
import pickle

import model_cartpole
import misc
import architecture
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

torch.set_default_tensor_type('torch.FloatTensor')

parser = ArgumentParser()
parser.add_argument("-d", "--dir", default="../experiments/cartpole", dest="dirname",
                    help="model's directory")
parser.add_argument("-r", "--repetitions", dest="repetitions", type=int, default=1,
                    help="simulation repetions")
args = parser.parse_args()

cart_position = np.linspace(0., 10., 40)
cart_velocity = np.linspace(0., 0.5, 40)
pole_angle = np.linspace(-3.1415/4, 3.1415/4, 15)
pole_ang_velocity = np.linspace(0., 0.5, 40)
pg = misc.ParametersHyperparallelepiped(cart_position, cart_velocity, pole_angle, pole_ang_velocity)

physical_model = model_cartpole.Model(pg.sample(sigma=0.05))

attacker = architecture.Attacker(physical_model, 2, 10, 2)
defender = architecture.Defender(physical_model, 2, 10)

misc.load_models(attacker, defender, args.dirname)

dt = 0.05 # timestep
steps = 300

def run(mode=None):
    physical_model.initialize_random()
    conf_init = {
        'x': physical_model.agent.x,
        'dot_x': physical_model.agent.dot_x,
        'theta': physical_model.agent.theta,                     
        'dot_theta': physical_model.agent.dot_theta,
    }

    sim_t = []
    sim_x = []
    sim_theta = []
    sim_dot_x = []
    sim_dot_theta = []

    t = 0
    for i in range(steps):
        with torch.no_grad():

            oa = torch.tensor(physical_model.agent.status).float()
            oe = torch.tensor(physical_model.environment.status).float()
            z = torch.rand(attacker.noise_size).float()
            
            if mode == 0:
                atk_policy = lambda x: torch.tensor(2.) if i > 100 and i < 200 else torch.tensor(-2.)
            elif mode == 1:
                atk_policy = lambda x: torch.tensor(2.) if i > 100 else torch.tensor(-2.)
            elif mode == 2:
                atk_policy = lambda x: torch.tensor(2.) if i < 100 else torch.tensor(-2.)
            else:
                atk_policy = attacker(torch.cat((z, oe)))

            def_policy = defender(oa)

        atk_input = atk_policy(dt).float()
        def_input = def_policy(dt).float()

        physical_model.step(atk_input, def_input, dt)

        sim_t.append(t)
        sim_x.append(physical_model.agent.x)
        sim_theta.append(physical_model.agent.theta)
        sim_dot_x.append(def_input)
        sim_dot_theta.append(atk_input)

        t += dt
        
    return {'init': conf_init,
            'sim_t': np.array(sim_t),
            'sim_x': np.array(sim_x),
            'sim_theta': np.array(sim_theta),
            'sim_dot_x': np.array(sim_dot_x),
            'sim_dot_theta': np.array(sim_dot_theta),
    }

records = []
for i in range(args.repetitions):
    sim = {}
    sim['pulse'] = run(0)
    sim['step_up'] = run(1)
    sim['step_down'] = run(2)
    # sim['atk'] = run()
    
    records.append(sim)
    
with open(os.path.join(args.dirname, 'sims.pkl'), 'wb') as f:
    pickle.dump(records, f)
