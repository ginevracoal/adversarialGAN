import os
import random
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from argparse import ArgumentParser

from utils.misc import *
from architecture.cartpole_target import *
from model.cartpole_target import *
from settings.cartpole_target import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
parser.add_argument("--plot_evolution", default=True, type=eval)
parser.add_argument("--scatter", default=True, type=eval, help="Generate scatterplot")
parser.add_argument("--hist", default=True, type=eval, help="Generate histograms")
parser.add_argument("--dark", default=False, type=eval, help="Use dark theme")
args = parser.parse_args()

cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
        atk_arch, def_arch, train_par, test_par, \
        robustness_theta, robustness_dist = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="cartpole_target_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

safe_theta = 0.2
safe_dist = 0.1
mc = 1.
mp = .1
alpha = 0.4


if args.dark:
    plt.style.use('utils/qb-common_dark.mplstyle')
    
with open(os.path.join(EXP+relpath, sims_filename), 'rb') as f:
    records = pickle.load(f)

def hist(time, const, pulse, atk, filename):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharex=True)

    ax[0].plot(time, const *100)
    ax[0].fill_between(time, const *100, alpha=0.5)
    ax[0].set(xlabel='time (s)', ylabel='% correct')
    ax[0].title.set_text('Acceleration const')

    ax[1].plot(time, pulse *100)
    ax[1].fill_between(time, pulse *100, alpha=0.5)
    ax[1].set(xlabel='time (s)', ylabel='% correct')
    ax[1].title.set_text('Acceleration pulse')

    ax[2].plot(time, atk *100)
    ax[2].fill_between(time, atk *100, alpha=0.5)
    ax[2].set(xlabel='time (s)', ylabel='% correct')
    ax[2].title.set_text('Against attacker')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def scatter(robustness_array, cart_pos_array, pole_ang_array, cart_vel_array, pole_ang_vel_array, filename):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3.5))
    fig.tight_layout(pad=4.0)

    customnorm = mcolors.TwoSlopeNorm(0)
    im = ax[0].scatter(cart_pos_array, cart_vel_array, c=robustness_array, cmap='BrBG', norm=customnorm, s=10)
    ax[0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')

    im = ax[1].scatter(pole_ang_array, pole_ang_vel_array, c=robustness_array, cmap='BrBG', norm=customnorm, s=10)
    ax[1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole angular frequency ($rad/s$)')
    
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('robustness', rotation=90, labelpad=-58)

    # fig.suptitle('Initial conditions vs robustness $\\rho$')
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)


def plot_evolution(def_records, cl_records, filename):

    fig, ax = plt.subplots(5, 1, figsize=(6, 8), sharex=True)

    ax[0].plot(def_records['sim_t'], def_records['sim_x'], label='defender',  color='darkblue')
    ax[0].plot(def_records['sim_t'], def_records['sim_x_target'], label='def. target', color='darkorange')
    ax[0].plot(cl_records['sim_t'], cl_records['sim_x'], label='classic', color='teal', lw=1)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_x_target'], label='cl. target', color='purple', lw=1)
    ax[0].set(ylabel=r'cart position ($m$)')
    ax[0].legend()

    ax[1].axhline(-safe_dist, ls='--', color='red', label="safe distance", lw=1)
    ax[1].axhline(safe_dist, ls='--', color='red', lw=1)
    ax[1].plot(def_records['sim_t'], def_records['sim_dist'], color='darkblue')    
    ax[1].plot(def_records['sim_t'], cl_records['sim_dist'], color='teal', lw=1)    
    ax[1].set(ylabel=r'distance from target ($m$)')
    ax[1].legend()

    ax[2].axhline(-safe_theta, ls='--', color='red', label="safe angle", lw=1)
    ax[2].axhline(safe_theta, ls='--', color='red', lw=1)
    ax[2].plot(def_records['sim_t'], def_records['sim_theta'], color='darkblue',  label='defender')
    ax[2].plot(cl_records['sim_t'], cl_records['sim_theta'], color='teal', label='classic', lw=1)
    ax[2].set(ylabel=r'pole angle ($rad$)')
    ax[2].legend()

    ax[3].plot(def_records['sim_t'], def_records['sim_attack_mu'], color='darkorange', label='def. attack')
    ax[3].plot(cl_records['sim_t'], cl_records['sim_attack_mu'], color='purple',label='cl. attack', lw=1)
    ax[3].set(ylabel=r'friction coefficient')
    ax[3].legend()

    ax[4].plot(def_records['sim_t'], def_records['sim_action'], label='defender', color='darkblue')
    ax[4].plot(cl_records['sim_t'], cl_records['sim_action'], label='defender', color='teal', lw=1)
    ax[4].set(xlabel=r'time ($s$)')
    ax[4].set(ylabel= r'cart control ($N$)')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter is True:

    size = len(records)

    robustness_computer = RobustnessComputer(robustness_theta, robustness_dist)

    robustness_array = np.zeros(size)
    cart_pos_array = np.zeros(size)
    pole_ang_array = np.zeros(size)
    cart_vel_array = np.zeros(size)
    pole_ang_vel_array = np.zeros(size)

    for mode in ["atk", "classic_atk"]:
        for i in range(size):

            trace_theta = torch.tensor(records[i][mode]['sim_theta'])
            trace_dist = torch.tensor(records[i][mode]['sim_dist'])
            rob_theta = robustness_computer.dqs_theta.compute(theta=trace_theta)
            rob_dist = robustness_computer.dqs_dist.compute(dist=trace_dist)
            robustness = alpha*rob_dist+(1-alpha)*rob_theta

            cart_pos = records[i][mode]['init']['x'] 
            pole_ang = records[i][mode]['init']['theta'] 
            cart_vel = records[i][mode]['init']['dot_x'] 
            pole_ang_vel = records[i][mode]['init']['dot_theta'] 

            robustness_array[i] = robustness
            cart_pos_array[i] = cart_pos
            pole_ang_array[i] = pole_ang
            cart_vel_array[i] = cart_vel
            pole_ang_vel_array[i] = pole_ang_vel

        scatter(robustness_array, cart_pos_array, pole_ang_array, cart_vel_array, pole_ang_vel_array, 
                str(mode)+'_scatterplot.png')
    

if args.plot_evolution is True:

    if len(records)>=1000:
        n=654
    else:
        n = random.randrange(len(records))

    print(n)
    for mode in ["const","pulse","atk"]:

        print(mode+":", records[n][mode]['init'])
        plot_evolution(records[n][mode], records[n]["classic_"+mode], 'evolution_'+mode+'.png')

if args.hist is True:

    size = len(records)
    const_pct = np.zeros_like(records[0]['const']['sim_theta'])
    pulse_pct = np.zeros_like(records[0]['pulse']['sim_theta'])
    atk_pct = np.zeros_like(records[0]['atk']['sim_theta'])

    robustness = lambda dist,t: alpha*(dist < safe_dist)+(1-alpha)*np.logical_and(t > -safe_theta, t < safe_theta)

    for i in range(size):

        x = records[i]['const']['sim_x']
        t = records[i]['const']['sim_theta']
        dist = records[i]['const']['sim_dist']
        const_pct = const_pct + robustness(dist, t)
        
        x = records[i]['pulse']['sim_x']
        t = records[i]['pulse']['sim_theta']
        dist = records[i]['pulse']['sim_dist']
        pulse_pct = pulse_pct + robustness(dist, t)

        x = records[i]['atk']['sim_x']
        t = records[i]['atk']['sim_theta']
        dist = records[i]['atk']['sim_dist']
        atk_pct = atk_pct + robustness(dist, t)

    time = records[0]['const']['sim_t']
    const_pct = const_pct / size
    pulse_pct = pulse_pct / size
    atk_pct = atk_pct / size

    hist(time, const_pct, pulse_pct, atk_pct, 'pct_histogram.png')
