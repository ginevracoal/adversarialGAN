import os
import random
from misc import *
import pickle
import model_platooning
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from argparse import ArgumentParser

################
### SETTINGS ###
################

train_par = {'train_steps':20000, 'atk_steps':3, 'def_steps':5, 'horizon':5., 'dt': 0.05, 'lr':1.}
test_par = {'test_steps':300, 'dt':0.05}

################

parser = ArgumentParser()
parser.add_argument("-d", "--dir", default="platooning", help="model's directory")
parser.add_argument("-r", "--repetitions", type=int, default=1, help="simulation repetions")
parser.add_argument("--triplots", default=True, help="Generate triplots")
parser.add_argument("--scatter", default=True, help="Generate scatterplot")
parser.add_argument("--hist", default=False, help="Generate histograms")
parser.add_argument("--dark", default=False, help="Use dark theme")
args = parser.parse_args()

relpath = args.dir+"_lr="+str(train_par["lr"])+"_dt="+str(train_par["dt"])+\
          "_horizon="+str(train_par["horizon"])+"_train_steps="+str(train_par["train_steps"])+\
          "_atk="+str(train_par["atk_steps"])+"_def="+str(train_par["def_steps"])

sims_filename = 'sims_reps='+str(args.repetitions)+'_dt='+str(test_par["dt"])+\
           '_test_steps='+str(test_par["test_steps"])+'.pkl'

if args.dark:
    plt.style.use('./qb-common_dark.mplstyle')
    
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
    fig.savefig(os.path.join(args.dir, filename), dpi=150)

def scatter(robustness_array, delta_pos_array, delta_vel_array, filename):
    fig, ax = plt.subplots(figsize=(10, 5))

    customnorm = mcolors.TwoSlopeNorm(0)
    sp = ax.scatter(delta_vel_array, delta_pos_array, c=robustness_array, cmap='RdYlGn', norm=customnorm)
    ax.set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')

    cb = fig.colorbar(sp)
    cb.ax.set_xlabel('$\\rho$')

    fig.suptitle('Initial conditions vs robustness $\\rho$')
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def plot(sim_time, sim_agent_pos, sim_agent_dist, sim_agent_acc, sim_env_pos, sim_env_acc, filename):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[0].plot(sim_time, sim_agent_pos, label='follower')
    ax[0].plot(sim_time, sim_env_pos, label='leader')
    ax[0].set(xlabel='time (s)', ylabel='position (m)')
    ax[0].legend()

    ax[1].plot(sim_time, sim_agent_dist)
    ax[1].set(xlabel='time (s)', ylabel='distance (m)')
    ax[1].axhline(2, ls='--', color='r')
    ax[1].axhline(10, ls='--', color='r')

    ax[2].plot(sim_time, sim_agent_acc, label='follower')
    ax[2].plot(sim_time, sim_env_acc, label='leader')
    # ax[2].plot(sim_time, np.clip(sim_agent_acc, -3, 3), label='follower')
    # ax[2].plot(sim_time, np.clip(sim_env_acc, -3, 3), label='leader')
    ax[2].set(xlabel='time (s)', ylabel='acceleration ($m/s^2$)')
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter:
    size = len(records)

    robustness_formula = 'G(dist <= 10 & dist >= 2)'
    robustness_computer = model_platooning.RobustnessComputer(robustness_formula)

    robustness_array = np.zeros(size)
    delta_pos_array = np.zeros(size)
    delta_vel_array = np.zeros(size)

    for i in range(size):
        sample_trace = torch.tensor(records[i]['atk']['sim_ag_dist'][-150:])
        robustness = float(robustness_computer.dqs.compute(dist=sample_trace))
        delta_pos = records[i]['atk']['init']['env_pos'] - records[i]['atk']['init']['ag_pos']
        delta_vel = records[i]['atk']['init']['env_vel'] - records[i]['atk']['init']['ag_vel']

        robustness_array[i] = robustness
        delta_pos_array[i] = delta_pos
        delta_vel_array[i] = delta_vel

    scatter(robustness_array, delta_pos_array, delta_vel_array, 'atk_scatterplot.png')

if args.triplots:
    n = random.randrange(len(records))
    print('pulse:', records[n]['pulse']['init'])
    plot(records[n]['pulse']['sim_t'], records[n]['pulse']['sim_ag_pos'], records[n]['pulse']['sim_ag_dist'], records[n]['pulse']['sim_ag_acc'], records[n]['pulse']['sim_env_pos'], records[n]['pulse']['sim_env_acc'], 'triplot_pulse.png')

    print('step_up:', records[n]['step_up']['init'])
    plot(records[n]['step_up']['sim_t'], records[n]['step_up']['sim_ag_pos'], records[n]['step_up']['sim_ag_dist'], records[n]['step_up']['sim_ag_acc'], records[n]['step_up']['sim_env_pos'], records[n]['step_up']['sim_env_acc'], 'triplot_step_up.png')

    print('step_down:', records[n]['step_down']['init'])
    plot(records[n]['step_down']['sim_t'], records[n]['step_down']['sim_ag_pos'], records[n]['step_down']['sim_ag_dist'], records[n]['step_down']['sim_ag_acc'], records[n]['step_down']['sim_env_pos'], records[n]['step_down']['sim_env_acc'], 'triplot_step_down.png')

    print('attacker:', records[n]['atk']['init'])
    plot(records[n]['atk']['sim_t'], records[n]['atk']['sim_ag_pos'], records[n]['atk']['sim_ag_dist'], records[n]['atk']['sim_ag_acc'], records[n]['atk']['sim_env_pos'], records[n]['atk']['sim_env_acc'], 'triplot_attacker.png')

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