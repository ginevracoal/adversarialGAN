import os
import torch
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from argparse import ArgumentParser

from utils.misc import *
from model.platooning import *
from settings.platooning import *
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
            robustness_formula = get_settings(args.architecture, mode="train")

relpath = get_relpath(main_dir="platooning_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

if args.dark:
    plt.style.use('utils/qb-common_dark.mplstyle')
    
with open(os.path.join(EXP+relpath, sims_filename), 'rb') as f:
    records = pickle.load(f)

def hist(time, pulse, step_up, step_down, atk, filename):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharex=True)

    ax[0].plot(time, step_up *100)
    ax[0].fill_between(time, step_up *100, alpha=0.5)
    ax[0].set(xlabel='time (s)', ylabel='% correct')
    ax[0].title.set_text('Sudden acceleration')

    ax[1].plot(time, step_down *100)
    ax[1].fill_between(time, step_down *100, alpha=0.5)
    ax[1].set(xlabel='time (s)', ylabel='% correct')
    ax[1].title.set_text('Sudden brake')

    ax[2].plot(time, pulse *100)
    ax[2].fill_between(time, pulse *100, alpha=0.5)
    ax[2].set(xlabel='time (s)', ylabel='% correct')
    ax[2].title.set_text('Acceleration pulse')

    ax[3].plot(time, atk *100)
    ax[3].fill_between(time, atk *100, alpha=0.5)
    ax[3].set(xlabel='time (s)', ylabel='% correct')
    ax[3].title.set_text('Against attacker')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def scatter(robustness_array, delta_pos_array, delta_vel_array, filename):
    fig, ax = plt.subplots(figsize=(6, 4))

    customnorm = mcolors.TwoSlopeNorm(0)
    im = ax.scatter(delta_vel_array, delta_pos_array, c=robustness_array, cmap='RdBu', norm=customnorm)
    ax.set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('robustness', rotation=90, labelpad=-50)

    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)


def plot_evolution(sim_time, sim_agent_pos, sim_agent_dist, sim_agent_acc, sim_env_pos, sim_env_acc, filename):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[0].plot(sim_time, sim_agent_pos, label='follower')
    ax[0].plot(sim_time, sim_env_pos, label='leader', color='tab:red')
    ax[0].set(xlabel=r'time ($s$)', ylabel=r'car position $x$ (m)')
    ax[0].legend()

    ax[1].plot(sim_time, sim_agent_dist)
    ax[1].set(xlabel=r'time ($s$)', ylabel=r'distance between cars $\|x_l-x_f\|$ (m)')
    ax[1].axhline(2, ls='--', label='safe distance', color='tab:orange')
    ax[1].axhline(10, ls='--', color='tab:orange')
    ax[1].legend()

    ax[2].plot(sim_time, sim_agent_acc, label='follower')
    ax[2].plot(sim_time, sim_env_acc, label='leader', color='tab:red')
    ax[2].set(xlabel=r'time ($s$)', ylabel=r'cart acceleration $\ddot x$ ($ms^{-2}$)')
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)


def plot_evolution_small(sim_time, sim_agent_pos, sim_agent_dist, sim_agent_acc, sim_env_pos, sim_env_acc, filename):
    fig, ax = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)

    ax[0].plot(sim_time, sim_agent_dist, color='tab:blue')
    ax[0].set(ylabel=r'distance between cars (m)')
    ax[0].axhline(2, ls='--', label='safe distance', color='tab:orange')
    ax[0].axhline(10, ls='--', color='tab:orange')
    ax[0].legend()

    ax[1].plot(sim_time, sim_agent_acc, label='follower', color='tab:blue')
    ax[1].plot(sim_time, sim_env_acc, label='leader', color='tab:red')
    ax[1].set(xlabel=r'time ($s$)', ylabel=r'cart acceleration ($ms^{-2}$)')
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter:
    size = len(records)

    robustness_computer = RobustnessComputer(robustness_formula)

    robustness_array = np.zeros(size)
    delta_pos_array = np.zeros(size)
    delta_vel_array = np.zeros(size)

    for i in range(size):
        sample_trace = torch.tensor(records[i]['atk']['sim_ag_dist'])#[10:])
        robustness = float(robustness_computer.dqs.compute(dist=sample_trace))
        delta_pos = records[i]['atk']['init']['env_pos'] - records[i]['atk']['init']['ag_pos']
        delta_vel = records[i]['atk']['init']['env_vel'] - records[i]['atk']['init']['ag_vel']

        robustness_array[i] = robustness
        delta_pos_array[i] = delta_pos
        delta_vel_array[i] = delta_vel

    scatter(robustness_array, delta_pos_array, delta_vel_array, 'atk_scatterplot.png')

if args.plot_evolution:

    n = random.randrange(len(records))
    # n=372

    print(n)
    for case in ['pulse', 'step_up', 'step_down', 'atk']:
        print(case, records[n][case]['init'])
        plot_evolution_small(records[n][case]['sim_t'], records[n][case]['sim_ag_pos'], 
             records[n][case]['sim_ag_dist'], 
             records[n][case]['sim_ag_acc'], records[n][case]['sim_env_pos'], 
             records[n][case]['sim_env_acc'], 'triplot_'+case+'.png')

if args.hist:
    size = len(records)
    pulse_pct = np.zeros_like(records[0]['pulse']['sim_ag_dist'])
    step_up_pct = np.zeros_like(records[0]['step_up']['sim_ag_dist'])
    step_down_pct = np.zeros_like(records[0]['step_down']['sim_ag_dist'])
    atk_pct = np.zeros_like(records[0]['atk']['sim_ag_dist'])

    for i in range(size):
        t = records[i]['pulse']['sim_ag_dist']
        pulse_pct = pulse_pct + np.logical_and(t > 2, t < 10)
        t = records[i]['step_up']['sim_ag_dist']
        step_up_pct = step_up_pct + np.logical_and(t > 2, t < 10)
        t = records[i]['step_down']['sim_ag_dist']
        step_down_pct = step_down_pct + np.logical_and(t > 2, t < 10)
        t = records[i]['atk']['sim_ag_dist']
        atk_pct = atk_pct + np.logical_and(t > 2, t < 10)

    time = records[0]['pulse']['sim_t']
    pulse_pct = pulse_pct / size
    step_up_pct = step_up_pct / size
    step_down_pct = step_down_pct / size
    atk_pct = atk_pct / size

    hist(time, pulse_pct, step_up_pct, step_down_pct, atk_pct, 'pct_histogram.png')