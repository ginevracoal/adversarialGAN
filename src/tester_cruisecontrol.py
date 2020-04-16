import os
import pickle

import model_cruisecontrol
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
agent_velocity = np.linspace(-12, 12, 25)
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity)

physical_model = model_cruisecontrol.Model(pg.sample(sigma=0.05))

attacker = architecture.Attacker(physical_model, 1, 10, 5, n_coeff=1)
defender = architecture.Defender(physical_model, 2, 10)

misc.load_models(attacker, defender, args.dirname)

dt = 0.05
steps = 300

def run(mode=None):
    physical_model.initialize_random()
    conf_init = {
        'ag_pos': physical_model.agent.position,
        'ag_vel': physical_model.agent.velocity,
    }

    sim_t = []
    sim_ag_pos = []
    sim_ag_vel = []
    sim_ag_acc = []

    def rbf(x):
        x = x.reshape(1) if x.dim() == 0 else x
        w = np.array([5]) if mode == 0 else np.array([-5])
        phi = lambda x: np.exp(-(x * 0.2)**2)
        d = np.arange(len(w)) +25
        r = np.abs(x[:, np.newaxis] - d)
        return w.dot(phi(r).T)

    t = 0
    with torch.no_grad():
        z = torch.rand(attacker.noise_size)
        atk_policy = attacker(z)
        
    if mode is not None:
        physical_model.environment._fn = rbf
    
    for i in range(steps):
        oa = torch.tensor(physical_model.agent.status)
        
        with torch.no_grad():
            def_policy = defender(oa)

        atk_input = atk_policy(0) if mode is None else None
        def_input = def_policy(dt)

        physical_model.step(atk_input, def_input, dt)

        sim_ag_acc.append(def_input)
        sim_t.append(t)
        sim_ag_pos.append(physical_model.agent.position)
        sim_ag_vel.append(physical_model.agent.velocity)

        t += dt
        
    x = np.arange(0, 100, physical_model.environment._dx)
    y = physical_model.environment.get_fn(torch.tensor(x))

    return {'init': conf_init,
            'space': {'x': x, 'y': y},
            'sim_t': np.array(sim_t),
            'sim_ag_pos': np.array(sim_ag_pos),
            'sim_ag_vel': np.array(sim_ag_vel),
            'sim_ag_acc': np.array(sim_ag_acc),
    }

records = []
for i in range(args.repetitions):
    sim = {}
    sim['up'] = run(0)
    sim['down'] = run(1)
    sim['atk'] = run()
    
    records.append(sim)
    
with open(os.path.join(args.dirname, 'sims.pkl'), 'wb') as f:
    pickle.dump(records, f)
