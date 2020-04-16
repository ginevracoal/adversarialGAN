import os
import random
import pickle

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

def hist(time, up, down, atk, filename):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex=True)

    ax[0].plot(time, up *100)
    ax[0].fill_between(time, up *100, alpha=0.5)
    ax[0].set(xlabel='time (s)', ylabel='% correct')
    ax[0].title.set_text('Hill')

    ax[1].plot(time, down *100)
    ax[1].fill_between(time, down *100, alpha=0.5)
    ax[1].set(xlabel='time (s)', ylabel='% correct')
    ax[1].title.set_text('Valley')

    ax[2].plot(time, atk *100)
    ax[2].fill_between(time, atk *100, alpha=0.5)
    ax[2].set(xlabel='time (s)', ylabel='% correct')
    ax[2].title.set_text('Attacker (RBF)')

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

def plot(space, sim_time, sim_agent_pos, sim_agent_vel, sim_agent_acc, filename):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    #ax[0].scatter(0, space['init_ypos'], color='g', label='start')
    #ax[0].scatter(sim_agent_pos[-1], space['end_ypos'], color='r', label='end')
    ax[0].axvline(0, ls='--', color='orange', label='start')
    ax[0].axvline(sim_agent_pos[-1], ls='--', color='g', label='end')
    ax[0].plot(space['x'], space['y'], zorder=-1)
    ax[0].set(xlabel='space (m)', ylabel='elevation (m)')
    ax[0].axis('equal')
    ax[0].legend()

    ax[1].axhline(4.75, ls='--', color='r')
    ax[1].axhline(5.25, ls='--', color='r')
    ax[1].plot(sim_time, sim_agent_vel)
    ax[1].set(xlabel='time (s)', ylabel='velocity (m/s)')

    ax[2].plot(sim_time, np.clip(sim_agent_acc, -5, 5))
    ax[2].set(xlabel='time (s)', ylabel='acceleration ($m/s^2$)')

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)

if args.triplots:
    n = random.randrange(len(records))
    print('up:', records[n]['up']['init'])
    plot(records[n]['up']['space'], records[n]['up']['sim_t'], records[n]['up']['sim_ag_pos'], records[n]['up']['sim_ag_vel'], records[n]['up']['sim_ag_acc'], 'triplot_up.png')

    print('down:', records[n]['down']['init'])
    plot(records[n]['down']['space'], records[n]['down']['sim_t'], records[n]['down']['sim_ag_pos'], records[n]['down']['sim_ag_vel'], records[n]['down']['sim_ag_acc'], 'triplot_down.png')

    print('atk:', records[n]['atk']['init'])
    plot(records[n]['atk']['space'], records[n]['atk']['sim_t'], records[n]['atk']['sim_ag_pos'], records[n]['atk']['sim_ag_vel'], records[n]['atk']['sim_ag_acc'], 'triplot_atk.png')

if args.hist:
    size = len(records)
    up_pct = np.zeros_like(records[0]['up']['sim_ag_vel'])
    down_pct = np.zeros_like(records[0]['down']['sim_ag_vel'])
    atk_pct = np.zeros_like(records[0]['atk']['sim_ag_vel'])

    for i in range(size):
        t = records[i]['up']['sim_ag_vel']
        up_pct = up_pct + np.logical_and(t > 4.75, t < 5.25)
        t = records[i]['down']['sim_ag_vel']
        down_pct = down_pct + np.logical_and(t > 4.75, t < 5.25)
        t = records[i]['atk']['sim_ag_vel']
        atk_pct = atk_pct + np.logical_and(t > 4.75, t < 5.25)

    time = records[0]['up']['sim_t']
    up_pct = up_pct / size
    down_pct = down_pct / size
    atk_pct = atk_pct / size

    hist(time, up_pct, down_pct, atk_pct, 'pct_histogram.png')
