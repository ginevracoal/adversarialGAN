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
agent_velocity = np.linspace(0, 12, 10)
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity)

physical_model = model_cruisecontrol.Model(pg.sample(sigma=0.05))

robustness_formula = 'G(v <= 4.90 & v >= 5.10)'
robustness_computer = model_cruisecontrol.RobustnessComputer(robustness_formula)

attacker = architecture.Attacker(physical_model, 1, 10, 5)
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


    road_padding = 5
    road_length = 80
    road_init = -road_padding
    road_end = road_length + road_padding
    n_points = 10
    px = np.linspace(0, road_end, n_points)
    py = -np.abs((px - road_length/2)*.35)
    if mode is not None:
        py = -py
    py = py - min(py)
    py = np.random.normal(py, 1)
    coeff = np.polyfit(px, py, 5)
    p = np.poly1d(coeff)

    t = 0
    for i in range(steps):
        oa = torch.tensor(physical_model.agent.status)
        
        with torch.no_grad():
            def_policy = defender(oa)

        atk_input = torch.tensor(p.deriv().c)
        def_input = def_policy(dt)

        physical_model.step(atk_input, def_input, dt)

        sim_ag_acc.append(def_input)
        sim_t.append(t)
        sim_ag_pos.append(physical_model.agent.position)
        sim_ag_vel.append(physical_model.agent.velocity)

        t += dt
        
    x = np.linspace(0, road_length, 100)

    return {'init': conf_init,
            'space': {'x': x, 'y': p(x), 'pos': p(sim_ag_pos)},
            'sim_t': np.array(sim_t),
            'sim_ag_pos': np.array(sim_ag_pos),
            'sim_ag_vel': np.array(sim_ag_vel),
            'sim_ag_acc': np.array(sim_ag_acc),
    }

records = []
for i in range(args.repetitions):
    sim = {}
    sim['up'] = run()
    sim['down'] = run('ciao')
    
    records.append(sim)
    
with open(os.path.join(args.dirname, 'sims.pkl'), 'wb') as f:
    pickle.dump(records, f)
