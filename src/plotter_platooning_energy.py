import os
import torch
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from argparse import ArgumentParser

from utils.misc import *
from model.platooning_energy import *
from settings.platooning_energy import *
from architecture.default import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=10, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
parser.add_argument("--plot_evolution", default=True, type=eval)
parser.add_argument("--scatter", default=True, type=eval, help="Generate scatterplot")
parser.add_argument("--hist", default=True, type=eval, help="Generate histograms")
parser.add_argument("--dark", default=False, type=eval, help="Use dark theme")
args = parser.parse_args()

agent_position, agent_velocity, leader_position, leader_velocity, \
            atk_arch, def_arch, train_par, test_par, \
            robustness_dist, robustness_power = get_settings(args.architecture, mode="train")

safe_dist1 = 2
safe_dist2 = 10
safe_power = 100
alpha = 0.9

relpath = get_relpath(main_dir="platooning_energy_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

if args.dark:
    plt.style.use('utils/qb-common_dark.mplstyle')
    
with open(os.path.join(EXP+relpath, sims_filename), 'rb') as f:
    records = pickle.load(f)

def hist(time, atk, filename): 
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharex=True)

    ax.plot(time, atk *100)
    ax.fill_between(time, atk *100, alpha=0.5)
    ax.set(xlabel='time (s)', ylabel='% correct')
    ax.title.set_text('Against attacker')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def scatter(robustness_array, delta_pos_array, delta_vel_array, filename):
    fig, ax = plt.subplots(figsize=(5, 4))

    customnorm = mcolors.TwoSlopeNorm(0)
    im = ax.scatter(delta_vel_array, delta_pos_array, c=robustness_array, cmap='BrBG', norm=customnorm, s=10)
    ax.set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('robustness', rotation=90, labelpad=-45)

    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)


def plot_evolution(sim_time, sim_agent_pos, sim_agent_dist, sim_env_pos, 
    sim_ag_e_torque, sim_ag_br_torque, sim_env_e_torque, sim_env_br_torque, filename):
    fig, ax = plt.subplots(4, 1, figsize=(8, 6))

    ax[0].plot(sim_time, sim_agent_pos, label='follower', color='darkblue')
    ax[0].plot(sim_time, sim_env_pos, label='leader', color='darkorange')
    ax[0].set(ylabel=r'car position ($m$)')
    ax[0].legend()

    ax[1].plot(sim_time, sim_agent_dist, color='darkblue')
    ax[1].set(ylabel=r'distance ($m$)')
    ax[1].axhline(2, ls='--', label='safe distance', color='red')
    ax[1].axhline(10, ls='--', color='red')
    ax[1].legend()

    ax[2].plot(sim_time, sim_ag_e_torque, label='follower', color='darkblue')
    ax[2].plot(sim_time, sim_env_e_torque, label='leader', color='darkorange')
    ax[2].set(ylabel=r'e_torque')
    ax[2].legend()

    ax[3].plot(sim_time, sim_ag_br_torque, label='follower', color='darkblue')
    ax[3].plot(sim_time, sim_env_br_torque, label='leader', color='darkorange')
    ax[3].set(xlabel=r'time ($s$)', ylabel=r'br_torque')
    ax[3].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter:
    size = len(records)

    robustness_computer = RobustnessComputer(robustness_dist, robustness_power)

    robustness_array = np.zeros(size)
    delta_pos_array = np.zeros(size)
    delta_vel_array = np.zeros(size)

    for mode in ["atk"]:

        for i in range(size):
            delta_pos = records[i]['atk']['init']['env_pos'] - records[i]['atk']['init']['ag_pos']
            delta_vel = records[i]['atk']['init']['env_vel'] - records[i]['atk']['init']['ag_vel']
            trace_dist = torch.tensor(records[i][mode]['sim_ag_dist'])
            trace_power = torch.tensor(records[i][mode]['sim_ag_power'])

            rob_dist = robustness_computer.dqs_dist.compute(dist=trace_dist)
            rob_power = robustness_computer.dqs_power.compute(power=trace_power)
            robustness = alpha*rob_dist+(1-alpha)*rob_power

            robustness_array[i] = robustness
            delta_pos_array[i] = delta_pos
            delta_vel_array[i] = delta_vel

        scatter(robustness_array, delta_pos_array, delta_vel_array, mode+'_scatterplot.png')

if args.plot_evolution:

    if len(records)>=1000:
        n=551
    else:
        n = random.randrange(len(records))
        
    print(n)
    for case in ['atk']:
        print(case, records[n][case]['init'])
        plot_evolution(records[n][case]['sim_t'], records[n][case]['sim_ag_pos'], 
             records[n][case]['sim_ag_dist'], records[n][case]['sim_env_pos'],
             records[n][case]['sim_ag_e_torque'], records[n][case]['sim_ag_br_torque'], 
             records[n][case]['sim_env_e_torque'], records[n][case]['sim_ag_br_torque'], 'evolution_'+case+'.png')

if args.hist:
    size = len(records)
    atk_pct = np.zeros_like(records[0]['atk']['sim_ag_dist'])

    robustness = lambda dist,power: alpha*np.logical_and(dist >= -safe_dist1, dist <= safe_dist2)+\
                                    (1-alpha)*(power <= safe_power)

    for i in range(size):
        dist = records[i]['atk']['sim_ag_dist']
        power = records[i]['atk']['sim_ag_power']
        atk_pct = atk_pct + robustness(dist, power)

    time = records[0]['atk']['sim_t']
    atk_pct = atk_pct / size

    hist(time, atk_pct, 'pct_histogram.png')