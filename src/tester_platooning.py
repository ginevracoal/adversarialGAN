import os
import pickle
import model_platooning
from misc import *
import architecture
import torch
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

################
### SETTINGS ###
################

agent_position = 0
agent_velocity = np.linspace(0, 20, 10)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 20, 10)

atk_arch = {'hidden':3, 'size':10, 'noise':2, 'coef':4}
def_arch = {'hidden':3, 'size':10, 'coef':4}
train_par = {'train_steps': 1,'atk_steps':3, 'def_steps':5, 'horizon':5., 'dt': 0.05, 'lr':1. }
test_par = {'test_steps': 300, 'dt': 0.05}

################

parser = ArgumentParser()
parser.add_argument("-d", "--dir", default="platooning", help="model's directory")
parser.add_argument("-r", "--repetitions", type=int, default=1, help="simulation repetions")
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

pg = ParametersHyperparallelepiped(agent_position, agent_velocity, leader_position, leader_velocity)
physical_model = model_platooning.Model(pg.sample(sigma=0.05), device=args.device)

attacker = architecture.Attacker(physical_model, *atk_arch.values())
defender = architecture.Defender(physical_model, *def_arch.values())

relpath = args.dir+"_lr="+str(train_par["lr"])+"_dt="+str(train_par["dt"])+\
          "_horizon="+str(train_par["horizon"])+"_train_steps="+str(train_par["train_steps"])
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
                atk_policy = lambda x: torch.tensor(2.) if i > int(test_par["test_steps"]/3) \
                                        and i < int(test_par["test_steps"]/2) \
                                        else torch.tensor(-2.)
            elif mode == 1:
                atk_policy = lambda x: torch.tensor(2.) if i > int(test_par["test_steps"]/3) \
                                        else torch.tensor(-2.)
            elif mode == 2:
                atk_policy = lambda x: torch.tensor(2.) if i < int(test_par["test_steps"]/3) \
                                        else torch.tensor(-2.)
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
    
filename = 'sims_reps='+str(args.repetitions)+'_dt='+str(test_par["dt"])+\
           '_test_steps='+str(test_par["test_steps"])+'.pkl'
           
with open(os.path.join(EXP+relpath, filename), 'wb') as f:
    pickle.dump(records, f)
