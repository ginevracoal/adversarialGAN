import os
import random
import pickle

import model_cartpole_target
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", default="../experiments/cartpole_target", dest="dirname",
                    help="model's directory")
parser.add_argument("--plot_evolution", default=True, type=eval)
parser.add_argument("--scatter", default=True, type=eval, help="Generate scatterplot")
parser.add_argument("--hist", default=True, type=eval, help="Generate histograms")
parser.add_argument("--dark", default=False, type=eval, help="Use dark theme")
args = parser.parse_args()

safe_theta = 0.392
safe_dist = 0.5
mc = 1.
mp = .1

if args.dark:
    plt.style.use('./qb-common_dark.mplstyle')
    
with open(os.path.join(args.dirname, 'sims.pkl'), 'rb') as f:
    records = pickle.load(f)

def hist(time, const, atk, filename):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharex=True)

    ax[0].plot(time, const *100)
    ax[0].fill_between(time, const *100, alpha=0.5)
    ax[0].set(xlabel='time (s)', ylabel='% correct')
    ax[0].title.set_text('Acceleration const')

    ax[1].plot(time, atk *100)
    ax[1].fill_between(time, atk *100, alpha=0.5)
    ax[1].set(xlabel='time (s)', ylabel='% correct')
    ax[1].title.set_text('Against attacker')

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

def scatter(robustness_array, cart_pos_array, pole_ang_array, cart_vel_array, pole_ang_vel_array, filename):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    print(cart_pos_array, "\n", pole_ang_array, "\n", cart_vel_array, "\n", pole_ang_vel_array)

    customnorm = mcolors.TwoSlopeNorm(0)
    sp0 = ax[0].scatter(cart_pos_array, cart_vel_array, c=robustness_array, cmap='RdYlGn', norm=customnorm)
    ax[0].set(xlabel='cart position', ylabel='cart velocity')
    # cb = fig.colorbar(sp0)
    # cb.ax[0].set_label('$\\rho$')

    sp1 = ax[1].scatter(pole_ang_array, pole_ang_vel_array, c=robustness_array, cmap='RdYlGn', norm=customnorm)
    ax[1].set(xlabel='pole angle', ylabel='pole angular frequency')
    # cb = fig.colorbar(sp1)
    # cb.ax[1].set_label('$\\rho$')

    fig.suptitle('Initial conditions vs robustness $\\rho$')
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

def plot(sim_time, sim_x, sim_theta, sim_dot_x, sim_ddot_x, sim_dot_theta, 
         sim_x_target, sim_def_acc, sim_dist, sim_attack_mu, sim_attack_nu, filename):
    fig, ax = plt.subplots(3, 2, figsize=(10, 6))

    ax[0,0].axhline(-safe_dist, ls='--', color='tab:orange', label="safe distance")
    ax[0,0].axhline(safe_dist, ls='--', color='tab:orange')
    ax[0,0].plot(sim_time, sim_x-sim_x_target, label='')    
    ax[0,0].set(xlabel='time (s)', ylabel='distance from target')# (m)')
    ax[0,0].legend()

    ax[1,0].plot(sim_time, sim_x, label='true x')
    ax[1,0].plot(sim_time, sim_x_target, label='target x', color='tab:red')
    ax[1,0].set(xlabel='time (s)', ylabel='cart position')# (m)')
    ax[1,0].legend()

    # ax[1,0].plot(sim_time, sim_dot_x, label='')
    # ax[1,0].set(xlabel='time (s)', ylabel='cart velocity (m/s)')

    ax[2,1].plot(sim_time, (mp + mc) * sim_def_acc, label='', color='tab:green')
    ax[2,1].set(xlabel='time (s)', ylabel= f'cart control')# (N)')

    ax[0,1].axhline(-safe_theta, ls='--', color='tab:orange', label="safe theta")
    ax[0,1].axhline(safe_theta, ls='--', color='tab:orange')
    ax[0,1].plot(sim_time, sim_theta, label='')
    ax[0,1].set(xlabel='time (s)', ylabel='pole angle')# (rad)')
    ax[0,1].legend()

    # ax[1,1].plot(sim_time, sim_dot_theta)
    # ax[1,1].set(xlabel='time (s)', ylabel='pole angular frequency (rad/s)')
    
    # ax[1,1].plot(sim_time, sim_def_acc, label='defender acceleration', color='g')
    ax[2,0].plot(sim_time, sim_ddot_x, label='true acceleration')
    ax[2,0].set(xlabel='time (s)', ylabel= f'cart acceleration')# (m/s^2)')

    print(sim_attack_mu)

    ax[1,1].plot(sim_time, sim_attack_mu, label='cart friction', color='tab:red')
    # ax[1,1].plot(sim_time, sim_attack_nu, label='air friction', color='tab:red', linestyle='--')
    ax[1,1].set(xlabel='time (s)', ylabel='friction coefficients')
    ax[1,1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

if args.scatter is True:

    size = len(records)

    robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta} & dist <= {safe_dist})'
    robustness_computer = model_cartpole_target.RobustnessComputer(robustness_formula)

    robustness_array = np.zeros(size)
    cart_pos_array = np.zeros(size)
    pole_ang_array = np.zeros(size)
    cart_vel_array = np.zeros(size)
    pole_ang_vel_array = np.zeros(size)

    for mode in ["const", "atk"]:
        for i in range(size):

            trace_theta = torch.tensor(records[i][mode]['sim_theta'])
            trace_dist = torch.tensor(records[i][mode]['sim_dist'])
            robustness = float(robustness_computer.dqs.compute(dist=trace_dist, theta=trace_theta))
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
    n = random.randrange(len(records))

    for mode in ["const","atk"]:

        print(mode+":", records[n][mode]['init'])
        plot(records[n][mode]['sim_t'], 
             records[n][mode]['sim_x'], records[n][mode]['sim_theta'], 
             records[n][mode]['sim_dot_x'], records[n][mode]['sim_ddot_x'], records[n][mode]['sim_dot_theta'],
             records[n][mode]['sim_x_target'], records[n][mode]['sim_def_acc'], 
             records[n][mode]['sim_dist'], records[n][mode]['sim_attack_mu'], records[n][mode]['sim_attack_nu'], 
             'evolution_'+mode+'.png')

if args.hist is True:

    size = len(records)
    const_pct = np.zeros_like(records[0]['const']['sim_theta'])
    atk_pct = np.zeros_like(records[0]['atk']['sim_theta'])

    for i in range(size):

        x = records[i]['const']['sim_x']
        t = records[i]['const']['sim_theta']
        dist = records[i]['const']['sim_dist']
        const_pct = const_pct +  np.logical_and(np.logical_and(t > -safe_theta, t < safe_theta), dist < safe_dist)

        x = records[i]['atk']['sim_x']
        t = records[i]['atk']['sim_theta']
        dist = records[i]['atk']['sim_dist']
        atk_pct = atk_pct + np.logical_and(np.logical_and(t > -safe_theta, t < safe_theta), dist < safe_dist)

    time = records[0]['const']['sim_t']
    const_pct = const_pct / size
    atk_pct = atk_pct / size

    hist(time, const_pct, atk_pct, 'pct_histogram.png')
