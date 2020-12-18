import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from argparse import ArgumentParser

from utils.misc import *
from architecture.cartpole_target import *
from model.cartpole_target import *
from model.cartpole_target_classic import *
from settings.cartpole_target import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
args = parser.parse_args()

cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
        atk_arch, def_arch, train_par, test_par, \
        robustness_theta, robustness_dist = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="cartpole_target_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

pg = ParametersHyperparallelepiped(cart_position, cart_velocity, 
                                        pole_angle, pole_ang_velocity, x_target)

model = Model(pg.sample())
model_classic = Model_classic(pg.sample())

attacker = Attacker(model, *atk_arch.values())
defender = Defender(model, *def_arch.values())
load_models(attacker, defender, EXP+relpath)

env_signal_class = Environment_signal(test_par["test_steps"])
environment_signal = env_signal_class.get_signal(dt=test_par["dt"])


def run(random_init, mode=None, classic_control=False):

    if classic_control is True:
        physical_model = model_classic

    else:
        physical_model = model

    physical_model.reinitialize(*random_init)

    conf_init = {
        'x': physical_model.agent.x,
        'dot_x': physical_model.agent.dot_x,
        'theta': physical_model.agent.theta,                     
        'dot_theta': physical_model.agent.dot_theta,
        'dist': physical_model.environment.dist,
    }

    sim_t = []
    sim_x = []
    sim_theta = []
    sim_dot_x = []
    sim_ddot_x = []
    sim_dot_theta = []
    sim_x_target = []
    sim_env_mu = []
    sim_ag_action = []
    sim_dist = []

    t = 0
    dt = test_par["dt"]
    for i in range(test_par["test_steps"]):

        with torch.no_grad():

            oa = torch.tensor(physical_model.agent.status)
            oe = torch.tensor(physical_model.environment.status)
            z = torch.rand(attacker.noise_size)
            
            if mode == 0:
                env_input = (torch.tensor(1.0), torch.tensor(0.0))

            elif mode == 1:
                env_input = environment_signal[i]

            else:
                env_input = attacker(torch.cat((z, oe)))

            if classic_control is True:
                st = oa.cpu().detach().numpy()
                physical_model.cartpole.computeControlSignals(st[:4], x_target=st[4])
                agent_input = torch.tensor(physical_model.cartpole.get_ctrl_signal())

                if physical_model.cartpole.is_unstable():
                    break

            else:
                agent_input = defender(oa)

        physical_model.step(env_input=env_input, agent_input=agent_input, dt=dt)

        sim_t.append(t)
        sim_x.append(physical_model.agent.x.item())
        sim_theta.append(physical_model.agent.theta.item())
        sim_dot_x.append(physical_model.agent.dot_x.item())
        sim_ddot_x.append(physical_model.agent.ddot_x.item())
        sim_dot_theta.append(physical_model.agent.dot_theta.item())
        sim_x_target.append(physical_model.environment.x_target.item())
        sim_dist.append(physical_model.environment.dist.item())
        sim_env_mu.append(env_input[1].item())
        sim_ag_action.append(agent_input.item())

        t += dt

    # print(sim_x[-5:], "\n",sim_x_target[-5:], "\n",sim_dist[-5:])

    return {'init': conf_init,
            'sim_t': np.array(sim_t),
            'sim_x': np.array(sim_x),
            'sim_theta': np.array(sim_theta),
            'sim_dot_x': np.array(sim_dot_x),
            'sim_ddot_x': np.array(sim_ddot_x),
            'sim_dot_theta': np.array(sim_dot_theta),
            'sim_x_target': np.array(sim_x_target),
            'sim_dist': np.array(sim_dist),
            'sim_env_mu': np.array(sim_env_mu),
            'sim_ag_action': np.array(sim_ag_action),
            }

records = []
for i in tqdm(range(args.repetitions)):
    sim = {}

    random_init = next(model._param_generator)
    sim['const'] = run(random_init, 0)
    sim['classic_const'] = run(random_init, 0, classic_control=True)

    random_init = next(model._param_generator)
    sim['pulse'] = run(random_init, 1)
    sim['classic_pulse'] = run(random_init, 1, classic_control=True)

    print(sim['pulse']['sim_x_target'][:5], "\n", sim['classic_pulse']['sim_x_target'][:5])

    random_init = next(model._param_generator)
    sim['atk'] = run(random_init)
    sim['classic_atk'] = run(random_init, classic_control=True)

    records.append(sim)

with open(os.path.join(EXP+relpath, sims_filename), 'wb') as f:
    pickle.dump(records, f)
