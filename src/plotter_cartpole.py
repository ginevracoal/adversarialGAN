import os
import random
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from argparse import ArgumentParser

from utils.misc import *
from model.cartpole import *
from settings.cartpole import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
parser.add_argument("--plot_evolution", default=True, type=eval)
parser.add_argument("--scatter", default=True, type=eval, help="Generate scatterplot")
parser.add_argument("--hist", default=True, type=eval, help="Generate histograms")
parser.add_argument("--dark", default=False, type=eval, help="Use dark theme")
args = parser.parse_args()

cart_position, cart_velocity, pole_angle, pole_ang_velocity, \
    arch, train_par, test_par, robustness_formula = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="cartpole_"+args.architecture, train_params=train_par)
net_filename = get_net_filename(arch["hidden"], arch["size"])
sims_filename = get_sims_filename(args.repetitions, test_par["dt"], test_par["test_steps"])

safe_theta = 0.2
safe_dist = 0.5
mc = 1.
mp = .1

if args.dark:
    plt.style.use('utils/qb-common_dark.mplstyle')
    
with open(os.path.join(EXP+relpath, sims_filename), 'rb') as f:
    records = pickle.load(f)

def hist(time, const, filename):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3), sharex=True)

    ax.plot(time, const *100)
    ax.fill_between(time, const *100, alpha=0.5)
    ax.set(xlabel='time (s)', ylabel='% correct')
    ax.title.set_text('Acceleration const')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def scatter(robustness_array, cart_pos_array, pole_ang_array, cart_vel_array, pole_ang_vel_array, filename):
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))
    fig.tight_layout(pad=4.0)

    cmap = plt.cm.get_cmap('Spectral')
    vmin = min(robustness_array)
    vmax = max(robustness_array)
    # print(cart_pos_array, "\n", pole_ang_array, "\n", cart_vel_array, "\n", pole_ang_vel_array)

    im = ax[0].scatter(cart_pos_array, cart_vel_array, c=robustness_array, cmap=cmap, vmin=vmin, vmax=vmax, s=8)
    ax[0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')

    im = ax[1].scatter(pole_ang_array, pole_ang_vel_array, c=robustness_array, cmap=cmap, vmin=vmin, vmax=vmax, s=8)
    ax[1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole angular frequency ($rad/s$)')
    
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('robustness', rotation=90, labelpad=-60)

    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def plot_evolution(sim_time, sim_x, sim_theta, sim_dot_x, sim_ddot_x, sim_dot_theta, 
         sim_action, filename):

    plt.style.use('seaborn')
    cmap = plt.cm.get_cmap('Spectral', 512)
    col = cmap(np.linspace(0, 1, 20))
    def_col = col[19]
    safe_col = col[0]
    lw=1

    fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True)

    ax[0].plot(sim_time, sim_ddot_x, label='true acceleration', color=def_col)
    ax[0].set(ylabel= r'cart acceleration ($ms^{-2}$)')

    ax[1].axhline(-safe_theta, ls='--', color=safe_col, label="safe theta", lw=lw)
    ax[1].axhline(safe_theta, ls='--', color=safe_col, lw=lw)
    ax[1].plot(sim_time, sim_theta, label='', color=def_col)
    ax[1].set(ylabel=r'pole angle ($rad$)')
    ax[1].legend()

    ax[2].plot(sim_time, sim_action, label='', color=def_col)
    ax[2].set(xlabel=r'time ($s$)', ylabel= r'cart control ($N$)')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter is True:

    size = len(records)

    robustness_computer = RobustnessComputer(robustness_formula)

    robustness_array = np.zeros(size)
    cart_pos_array = np.zeros(size)
    pole_ang_array = np.zeros(size)
    cart_vel_array = np.zeros(size)
    pole_ang_vel_array = np.zeros(size)

    for mode in ["const"]:
        for i in range(size):

            trace_theta = torch.tensor(records[i][mode]['sim_theta'])
            robustness = float(robustness_computer.dqs.compute(theta=trace_theta))
            cart_pos = records[i][mode]['init']['x'] 
            pole_ang = records[i][mode]['init']['theta'] 
            cart_vel = records[i][mode]['init']['dot_x'] 
            pole_ang_vel = records[i][mode]['init']['dot_theta'] 

            robustness_array[i] = robustness
            cart_pos_array[i] = cart_pos
            pole_ang_array[i] = pole_ang
            cart_vel_array[i] = cart_vel
            pole_ang_vel_array[i] = pole_ang_vel

        scatter(robustness_array, cart_pos_array, pole_ang_array, cart_vel_array, pole_ang_vel_array, 'atk_scatterplot.png')

if args.plot_evolution is True:

    if len(records)>=1000:
        n=721
    else:
        n = random.randrange(len(records))

    print(n)
    for mode in ["const"]:

        print(mode+":", records[n][mode]['init'])
        plot_evolution(records[n][mode]['sim_t'], 
             records[n][mode]['sim_x'], records[n][mode]['sim_theta'], 
             records[n][mode]['sim_dot_x'], records[n][mode]['sim_ddot_x'], 
             records[n][mode]['sim_dot_theta'], records[n][mode]['sim_action'], 
             'evolution_'+mode+'.png')

if args.hist is True:

    size = len(records)
    const_pct = np.zeros_like(records[0]['const']['sim_theta'])

    for i in range(size):

        x = records[i]['const']['sim_x']
        t = records[i]['const']['sim_theta']
        const_pct = const_pct +  np.logical_and(t > -safe_theta, t < safe_theta)
        
    time = records[0]['const']['sim_t']
    const_pct = const_pct / size

    hist(time, const_pct, 'pct_histogram.png')
