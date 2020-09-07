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
parser.add_argument("--ode_idx", default=0, type=int, help="Choose ode idx")
parser.add_argument("--fourplots", default=True, action="store_true" , help="Generate four plots")
parser.add_argument("--scatter", default=False, action="store_true" , help="Generate scatterplot")
parser.add_argument("--hist", default=False, action="store_true" , help="Generate histograms")
parser.add_argument("--dark", default=False, action="store_true" , help="Use dark theme")
args = parser.parse_args()

if args.dark:
    plt.style.use('./qb-common_dark.mplstyle')
    
with open(os.path.join(args.dirname+str(args.ode_idx), 'sims.pkl'), 'rb') as f:
    records = pickle.load(f)

def hist(time, pulse, push, pull, atk, filename):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharex=True)

    ax[0].plot(time, push *100)
    ax[0].fill_between(time, push *100, alpha=0.5)
    ax[0].set(xlabel='time (s)', ylabel='% correct')
    ax[0].title.set_text('Sudden acceleration')

    ax[1].plot(time, pull *100)
    ax[1].fill_between(time, pull *100, alpha=0.5)
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
    fig.savefig(os.path.join(args.dirname+str(args.ode_idx), filename), dpi=150)

def scatter(robustness_array, delta_pos_array, delta_vel_array, filename):
    fig, ax = plt.subplots(figsize=(10, 5))

    customnorm = mcolors.TwoSlopeNorm(0)
    sp = ax.scatter(delta_vel_array, delta_pos_array, c=robustness_array, cmap='RdYlGn', norm=customnorm)
    ax.set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')

    cb = fig.colorbar(sp)
    cb.ax.set_xlabel('$\\rho$')

    fig.suptitle('Initial conditions vs robustness $\\rho$')
    fig.savefig(os.path.join(args.dirname+str(args.ode_idx), filename), dpi=150)

def plot(sim_time, sim_x, sim_theta, sim_ddot_x, sim_attack, filename):
    fig, ax = plt.subplots(2, 2, figsize=(10, 4))

    ax[0,0].plot(sim_time, sim_x, label='')
    ax[0,0].set(xlabel='time (s)', ylabel='cart position')

    ax[1,0].axhline(-0.785, ls='--', color='r')
    ax[1,0].axhline(0.785, ls='--', color='r')
    ax[1,0].plot(sim_time, sim_theta, label='')
    ax[1,0].set(xlabel='time (s)', ylabel='pole angle')

    if args.ode_idx==0:
        ax[0,1].plot(sim_time, sim_ddot_x, label='')
        ax[0,1].set(xlabel='time (s)', ylabel='defender acceleration')
    elif args.ode_idx==1:
        ax[0,1].plot(sim_time, sim_ddot_x, label='')
        ax[0,1].set(xlabel='time (s)', ylabel='cart acceleration')

    if args.ode_idx==0:
        ax[1,1].plot(sim_time, sim_attack, label='')
        ax[1,1].set(xlabel='time (s)', ylabel='attacker acceleration')

    elif args.ode_idx==1:
        ax[1,1].plot(sim_time, sim_attack, label='')
        ax[1,1].set(xlabel='time (s)', ylabel='cart friction')

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname+str(args.ode_idx), filename), dpi=150)

if args.scatter:

    size = len(records)

    robustness_formula = 'G(theta >= -0.785 & theta <= 0.785)'
    robustness_computer = model_cartpole.RobustnessComputer(robustness_formula)

    robustness_array = np.zeros(size)
    pole_angle_array = np.zeros(size)
    cart_acc_array = np.zeros(size)

    for i in range(size):
        sample_trace = torch.tensor(records[i]['atk']['sim_theta'][-150:])
        robustness = float(robustness_computer.dqs.compute(theta=sample_trace))
        pole_angle = records[i]['atk']['init']['theta']
        cart_acc = records[i]['atk']['init']['dot_x'] 

        robustness_array[i] = robustness
        pole_angle_array[i] = pole_angle
        cart_acc_array[i] = cart_acc

    scatter(robustness_array, pole_angle_array, cart_acc_array, 'atk_scatterplot.png')

if args.fourplots:
    n = random.randrange(len(records))
    print('pulse:', records[n]['pulse']['init'])
    plot(records[n]['pulse']['sim_t'], records[n]['pulse']['sim_x'], 
         records[n]['pulse']['sim_theta'], records[n]['pulse']['sim_ddot_x'], 
         records[n]['pulse']['sim_attack'], 'triplot_pulse.png')

    print('push:', records[n]['push']['init'])
    plot(records[n]['push']['sim_t'], records[n]['push']['sim_x'], records[n]['push']['sim_theta'], 
         records[n]['push']['sim_ddot_x'], records[n]['push']['sim_attack'], 'triplot_push.png')

    print('pull:', records[n]['pull']['init'])
    plot(records[n]['pull']['sim_t'], records[n]['pull']['sim_x'], records[n]['pull']['sim_theta'],
         records[n]['pull']['sim_ddot_x'], records[n]['pull']['sim_attack'], 'triplot_pull.png')

    print('attacker:', records[n]['atk']['init'])
    plot(records[n]['atk']['sim_t'], records[n]['atk']['sim_x'],records[n]['atk']['sim_theta'], 
         records[n]['atk']['sim_ddot_x'], records[n]['atk']['sim_attack'], 
         'triplot_attacker.png')

if args.hist:

    size = len(records)
    pulse_pct = np.zeros_like(records[0]['pulse']['sim_theta'])
    push_pct = np.zeros_like(records[0]['push']['sim_theta'])
    pull_pct = np.zeros_like(records[0]['pull']['sim_theta'])
    atk_pct = np.zeros_like(records[0]['atk']['sim_theta'])

    for i in range(size):
        t = records[i]['pulse']['sim_theta']
        pulse_pct = pulse_pct + np.logical_and(t > 2, t < 10)
        t = records[i]['push']['sim_theta']
        push_pct = push_pct + np.logical_and(t > 2, t < 10)
        t = records[i]['pull']['sim_theta']
        pull_pct = pull_pct + np.logical_and(t > 2, t < 10)
        t = records[i]['atk']['sim_theta']
        atk_pct = atk_pct + np.logical_and(t > 2, t < 10)

    time = records[0]['pulse']['sim_t']
    pulse_pct = pulse_pct / size
    push_pct = push_pct / size
    pull_pct = pull_pct / size
    atk_pct = atk_pct / size

    hist(time, pulse_pct, push_pct, pull_pct, atk_pct, 'pct_histogram.png')
