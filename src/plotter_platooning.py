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
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
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

def hist(time, pulse, atk, filename):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=True)

    ax[0].plot(time, pulse *100)
    ax[0].fill_between(time, pulse *100, alpha=0.5)
    ax[0].set(xlabel='time (s)', ylabel='% correct')
    ax[0].title.set_text('Acceleration pulse')

    ax[1].plot(time, atk *100)
    ax[1].fill_between(time, atk *100, alpha=0.5)
    ax[1].set(xlabel='time (s)', ylabel='% correct')
    ax[1].title.set_text('Against attacker')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def scatter(robustness_array, delta_pos_array, delta_vel_array, filename):
    fig, ax = plt.subplots(figsize=(5, 4))

    cmap = plt.cm.get_cmap('Spectral')
    vmin = min(robustness_array)
    vmax = max(robustness_array)

    im = ax.scatter(delta_vel_array, delta_pos_array, c=robustness_array, cmap=cmap, vmin=vmin, vmax=vmax, s=8)
    ax.set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('robustness', rotation=90, labelpad=-45)

    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)


def plot_evolution(sim_time, sim_agent_pos, sim_agent_dist, sim_agent_acc, sim_env_pos, sim_env_acc, filename):

    plt.style.use('seaborn')
    cmap = plt.cm.get_cmap('Spectral', 512)
    col = cmap(np.linspace(0, 1, 20))
    def_col = col[19]
    atk_col = col[3]
    safe_col = col[0]
    lw=1

    # fig, ax = plt.subplots(3, 1, figsize=(6, 5))

    # ax[0].plot(sim_time, sim_agent_pos, label='follower', color=def_col)
    # ax[0].plot(sim_time, sim_env_pos, label='leader', color=atk_col)
    # ax[0].set(ylabel=r'car position ($m$)')
    # ax[0].legend()

    # ax[1].plot(sim_time, sim_agent_dist, color=def_col)
    # ax[1].set(ylabel=r'distance ($m$)')
    # ax[1].axhline(2, ls='--', label='safe distance', color=safe_col, lw=lw)
    # ax[1].axhline(10, ls='--', color=safe_col, lw=lw)
    # ax[1].legend()

    # ax[2].plot(sim_time, sim_agent_acc, label='follower', color=def_col)
    # ax[2].plot(sim_time, sim_env_acc, label='leader', color=atk_col, lw=lw)
    # ax[2].set(xlabel=r'time ($s$)', ylabel=r'car acceleration ($ms^{-2}$)')
    # ax[2].legend()

    fig, ax = plt.subplots(2, 1, figsize=(6, 3.5))

    ax[0].plot(sim_time, sim_agent_dist, color=def_col, label='distance from leader')
    ax[0].set(ylabel=r'distance ($m$)')
    ax[0].axhline(2, ls='--', label='safe distance', color=safe_col, lw=lw)
    ax[0].axhline(10, ls='--', color=safe_col, lw=lw)
    ax[0].legend()

    ax[1].plot(sim_time, sim_agent_acc, label='follower', color=def_col)
    ax[1].plot(sim_time, sim_env_acc, label='leader', color=atk_col, lw=lw)
    ax[1].set(xlabel=r'time ($s$)', ylabel=r'car acceleration ($ms^{-2}$)')
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
        sample_trace = torch.tensor(records[i]['atk']['sim_ag_dist'][50:])
        robustness = float(robustness_computer.dqs.compute(dist=sample_trace))
        delta_pos = records[i]['atk']['init']['env_pos'] - records[i]['atk']['init']['ag_pos']
        delta_vel = records[i]['atk']['init']['env_vel'] - records[i]['atk']['init']['ag_vel']

        robustness_array[i] = robustness
        delta_pos_array[i] = delta_pos
        delta_vel_array[i] = delta_vel

    scatter(robustness_array, delta_pos_array, delta_vel_array, 'platooning_atk_scatterplot.png')

if args.plot_evolution:

    if len(records)>=1000:
        n=604
    else:
        n = random.randrange(len(records))
        
    for case in ['pulse', 'atk']:
        print(case, n, records[n][case]['init'], 'dist: ',
              records[i]['atk']['init']['env_pos'] - records[i]['atk']['init']['ag_pos'])
        plot_evolution(records[n][case]['sim_t'], records[n][case]['sim_ag_pos'], 
             records[n][case]['sim_ag_dist'], 
             records[n][case]['sim_ag_acc'], records[n][case]['sim_env_pos'], 
             records[n][case]['sim_env_acc'], 'platooning_evolution_'+case+'.png')

if args.hist:
    size = len(records)
    pulse_pct = np.zeros_like(records[0]['pulse']['sim_ag_dist'])
    atk_pct = np.zeros_like(records[0]['atk']['sim_ag_dist'])

    for i in range(size):
        t = records[i]['pulse']['sim_ag_dist']
        pulse_pct = pulse_pct + np.logical_and(t > 2, t < 10)
        t = records[i]['atk']['sim_ag_dist']
        atk_pct = atk_pct + np.logical_and(t > 2, t < 10)

    time = records[0]['pulse']['sim_t']
    pulse_pct = pulse_pct / size
    atk_pct = atk_pct / size

    hist(time, pulse_pct, atk_pct, 'platooning_pct_histogram.png')