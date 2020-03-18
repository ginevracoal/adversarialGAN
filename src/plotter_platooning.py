import os
import random
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="dirname",
                    help="model's directory")
parser.add_argument("--triplots", default=False, action="store_true" , help="Generate triplots")
parser.add_argument("--hist", default=False, action="store_true" , help="Generate histograms")
args = parser.parse_args()
    
with open(os.path.join(args.dirname, 'sims.pkl'), 'rb') as f:
    records = pickle.load(f)

def hist(pulse, step_up, step_down, atk, filename):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharex=True)

    config = {
        'norm_hist': True,
        #'bins': range(0, 100, 5),
        'color': 'b',
        'kde': False,
        'axlabel': '% of $dist$ in limits',
    }
    sns.distplot(step_up * 100, **config, ax=ax[0]).set_title('Sudden acceleration')
    sns.distplot(step_down * 100, **config, ax=ax[1]).set_title('Sudden brake')
    sns.distplot(pulse * 100, **config, ax=ax[2]).set_title('Acceleration pulse')
    sns.distplot(atk * 100, **config, ax=ax[3]).set_title('Against attacker')

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

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

    ax[2].plot(sim_time, np.clip(sim_agent_acc, -3, 3), label='follower')
    ax[2].plot(sim_time, np.clip(sim_env_acc, -3, 3), label='leader')
    ax[2].set(xlabel='time (s)', ylabel='acceleration ($m/s^2$)')
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

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
    pulse_pct = np.zeros(size)
    step_up_pct = np.zeros(size)
    step_down_pct = np.zeros(size)
    atk_pct = np.zeros(size)

    for i in range(size):
        t = records[i]['pulse']['sim_ag_dist']
        pulse_pct[i] = np.sum(np.logical_and(t > 2, t < 10)) / len(t)
        t = records[i]['step_up']['sim_ag_dist']
        step_up_pct[i] = np.sum(np.logical_and(t > 2, t < 10)) / len(t)
        t = records[i]['step_down']['sim_ag_dist']
        step_down_pct[i] = np.sum(np.logical_and(t > 2, t < 10)) / len(t)
        t = records[i]['atk']['sim_ag_dist']
        atk_pct[i] = np.sum(np.logical_and(t > 2, t < 10)) / len(t)

    hist(pulse_pct, step_up_pct, step_down_pct, atk_pct, 'pct_histogram.png')
