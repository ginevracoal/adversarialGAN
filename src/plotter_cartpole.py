import os
import random
import pickle

import model_cartpole
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", default="../experiments/cartpole", dest="dirname",
                    help="model's directory")
parser.add_argument("--triplots", default=False, action="store_true" , help="Generate triplots")
parser.add_argument("--scatter", default=False, action="store_true" , help="Generate scatterplot")
parser.add_argument("--hist", default=False, action="store_true" , help="Generate histograms")
parser.add_argument("--dark", default=False, action="store_true" , help="Use dark theme")
args = parser.parse_args()

if args.dark:
    plt.style.use('./qb-common_dark.mplstyle')
    
with open(os.path.join(args.dirname, 'sims.pkl'), 'rb') as f:
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
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

# def scatter(robustness_array, delta_pos_array, delta_vel_array, filename):
#     fig, ax = plt.subplots(figsize=(10, 5))

#     customnorm = mcolors.TwoSlopeNorm(0)
#     sp = ax.scatter(delta_vel_array, delta_pos_array, c=robustness_array, cmap='RdYlGn', norm=customnorm)
#     ax.set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')

#     cb = fig.colorbar(sp)
#     cb.ax.set_xlabel('$\\rho$')

#     fig.suptitle('Initial conditions vs robustness $\\rho$')
#     fig.savefig(os.path.join(args.dirname, filename), dpi=150)

def plot(sim_time, sim_c_pos, sim_p_ang, sim_c_acc, sim_p_acc, filename):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[0].plot(sim_time, sim_c_pos, label='cart')
    ax[0].set(xlabel='time (s)', ylabel='position')
    ax[0].legend()

    ax[1].axhline(-0.785, ls='--', color='r')
    ax[1].axhline(0.785, ls='--', color='r')
    ax[1].plot(sim_time, sim_p_ang, label='pole')
    ax[1].set(xlabel='time (s)', ylabel='angle')

    ax[2].plot(sim_time, sim_c_acc, label='cart')
    ax[2].plot(sim_time, sim_p_acc, label='pole')
    ax[2].set(xlabel='time (s)', ylabel='acceleration')
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

if args.scatter:

    raise NotImplementedError()
    # size = len(records)

    # robustness_formula = 'G(theta >= -0.785 & theta <= 0.785)'
    # robustness_computer = model_cartpole.RobustnessComputer(robustness_formula)

    # robustness_array = np.zeros(size)
    # delta_pos_array = np.zeros(size)
    # delta_vel_array = np.zeros(size)

    # for i in range(size):
    #     sample_trace = torch.tensor(records[i]['atk']['sim_ag_dist'][-150:])
    #     robustness = float(robustness_computer.dqs.compute(dist=sample_trace))
    #     delta_pos = records[i]['atk']['init']['env_pos'] - records[i]['atk']['init']['ag_pos']
    #     delta_vel = records[i]['atk']['init']['env_vel'] - records[i]['atk']['init']['ag_vel']

    #     robustness_array[i] = robustness
    #     delta_pos_array[i] = delta_pos
    #     delta_vel_array[i] = delta_vel

    # scatter(robustness_array, delta_pos_array, delta_vel_array, 'atk_scatterplot.png')

if args.triplots:
    n = random.randrange(len(records))
    print('pulse:', records[n]['pulse']['init'])
    plot(records[n]['pulse']['sim_t'], records[n]['pulse']['sim_c_pos'], records[n]['pulse']['sim_p_ang'], records[n]['pulse']['sim_c_acc'], records[n]['pulse']['sim_p_acc'], 'triplot_pulse.png')

    print('step_up:', records[n]['step_up']['init'])
    plot(records[n]['step_up']['sim_t'], records[n]['step_up']['sim_c_pos'], records[n]['step_up']['sim_p_ang'], records[n]['step_up']['sim_c_acc'], records[n]['step_up']['sim_c_acc'], 'triplot_step_up.png')

    print('step_down:', records[n]['step_down']['init'])
    plot(records[n]['step_down']['sim_t'], records[n]['step_down']['sim_c_pos'], records[n]['step_down']['sim_p_ang'], records[n]['step_down']['sim_c_acc'], records[n]['step_down']['sim_c_acc'], 'triplot_step_down.png')

    print('attacker:', records[n]['atk']['init'])
    plot(records[n]['atk']['sim_t'], records[n]['atk']['sim_c_pos'],records[n]['step_down']['sim_p_ang'], records[n]['step_down']['sim_c_acc'], records[n]['step_down']['sim_c_acc'], 'triplot_attacker.png')

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
