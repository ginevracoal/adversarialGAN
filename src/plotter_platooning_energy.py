import os
import torch
import random
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from argparse import ArgumentParser

from utils.misc import *
from model.platooning_energy import *
from settings.platooning_energy import *
from architecture.platooning_energy import *

parser = ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, default=1000, help="simulation repetions")
parser.add_argument("--architecture", type=str, default="default", help="architecture's name")
parser.add_argument("--plot_evolution", default=True, type=eval)
parser.add_argument("--scatter", default=True, type=eval, help="Generate scatterplot")
parser.add_argument("--hist", default=True, type=eval, help="Generate histograms")
parser.add_argument("--dark", default=False, type=eval, help="Use dark theme")
args = parser.parse_args()

agent_position, agent_velocity, leader_position, leader_velocity, \
    atk_arch, def_arch, train_par, test_par, robustness_dist, robustness_power, \
    safe_dist_lower, safe_dist_upper, safe_power, alpha = get_settings(args.architecture, mode="test")

relpath = get_relpath(main_dir="platooning_energy_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

if args.dark:
    plt.style.use('utils/qb-common_dark.mplstyle')
    
with open(os.path.join(EXP+relpath, sims_filename), 'rb') as f:
    records = pickle.load(f)

def hist(time, pulse, atk, filename): 
    fig, ax = plt.subplots(1, 2, figsize=(5, 3), sharex=True)

    ax[0].plot(time, pulse *100)
    ax[0].fill_between(time, pulse *100, alpha=0.5)
    ax[0].set(xlabel='time (s)', ylabel='% correct')
    ax[0].title.set_text('Against pulse')

    ax[1].plot(time, atk *100)
    ax[1].fill_between(time, atk *100, alpha=0.5)
    ax[1].set(xlabel='time (s)', ylabel='% correct')
    ax[1].title.set_text('Against attacker')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def scatter(robustness_array, delta_pos_array, delta_vel_array, 
            cl_robustness_array, cl_delta_pos_array, cl_delta_vel_array, filename, plot_differences=False):

    cmap = plt.cm.get_cmap('Spectral')

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.8))
    fig.tight_layout(pad=3.0)

    vmax = max(max(cl_robustness_array), max(robustness_array))
    vmin = min(min(cl_robustness_array), min(robustness_array))
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmax=vmax, vmin=vmin)

    im = ax[0].scatter(cl_delta_vel_array, cl_delta_pos_array, c=cl_robustness_array, 
                         cmap=cmap, norm=norm, s=8)
    # ax[0].set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')
    ax[0].set_title('Classic follower', weight='bold', size=10, pad=8)
    # plt.colorbar(im, ax=ax[0])
    
    vmax = max(abs(robustness_array))
    vmin = -vmax

    im = ax[1].scatter(delta_vel_array, delta_pos_array, c=robustness_array, cmap=cmap, norm=norm, s=8)
    # ax[1].set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')
    ax[1].set_title('Defender follower', weight='bold', size=10, pad=8)
    # plt.colorbar(im, ax=ax[1])

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.9, 0.18, 0.03, 0.7])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar_ax.set_ylabel('Robustness', rotation=90, labelpad=-55)

    fig.text(0.5, 0.08, '$\Delta$v between leader and follower ($m/s$)', ha='center')
    fig.text(0.03, 0.5, 'Distance ($m$)', va='center', rotation='vertical')

    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

    ## robustness differences

    if plot_differences:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.tight_layout(pad=3.0)

        robustness_differences = robustness_array - cl_robustness_array
        norm = mpl.colors.TwoSlopeNorm(vcenter=0)

        ax.scatter(delta_vel_array, delta_pos_array, c=robustness_differences, cmap=cmap, norm=norm, s=8)
        ax.set(xlabel='$\Delta$v between leader and follower ($m/s$)', ylabel='Distance ($m$)')

        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, ax=ax, cax=cbar_ax)
        cbar_ax.set_ylabel('Defender rob. - Classic rob.', rotation=90, labelpad=-45)
        plt.figtext(0.48, 0.95, 'Robustness difference vs initial configuration', ha='center', va='center', weight='bold')

        fig.savefig(os.path.join(EXP+relpath, "diff_"+filename), dpi=150)

def plot_evolution_full(def_records, cl_records, filename):

    plt.style.use('seaborn')
    cmap = plt.cm.get_cmap('Spectral', 512)
    col = cmap(np.linspace(0, 1, 20))
    def_col = col[19]
    cl_col = col[16]
    def_atk_col = col[3]
    cl_atk_col = col[5]
    safe_col = col[0]
    lw=1
    
    fig, ax = plt.subplots(5, 1, figsize=(6, 9))

    ax[0].plot(def_records['sim_t'], def_records['sim_ag_pos'], label='defender follower', color=def_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_env_pos'], label='defender leader', color=def_atk_col, lw=lw)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_ag_pos'], label='classic follower', color=cl_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_env_pos'], label='classic leader', color=cl_atk_col, lw=lw)
    ax[0].set(ylabel=r'car position ($m$)')

    ax[1].plot(def_records['sim_t'], def_records['sim_ag_dist'], color=def_col)
    ax[1].plot(cl_records['sim_t'], cl_records['sim_ag_dist'], color=cl_col)
    ax[1].set(ylabel=r'distance ($m$)')
    ax[1].axhline(safe_dist_lower, ls='--', label='safe distance', color=safe_col, lw=lw)
    ax[1].axhline(safe_dist_upper, ls='--', color=safe_col, lw=lw)

    ax[2].plot(def_records['sim_t'], def_records['sim_ag_e_torque'], color=def_col)
    ax[2].plot(def_records['sim_t'], def_records['sim_env_e_torque'], color=def_atk_col, lw=lw)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_ag_e_torque'], color=cl_col)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_env_e_torque'], color=cl_atk_col, lw=lw)
    ax[2].set(ylabel=r'e_torque ($N \cdot m$)')

    ax[3].plot(def_records['sim_t'], def_records['sim_ag_br_torque'], color=def_col)
    ax[3].plot(def_records['sim_t'], def_records['sim_env_br_torque'], color=def_atk_col, lw=lw)
    ax[3].plot(cl_records['sim_t'], cl_records['sim_ag_br_torque'], color=cl_col)
    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_br_torque'], color=cl_atk_col, lw=lw)
    ax[3].set(ylabel=r'br_torque ($N \cdot m$)')

    ax[4].plot(def_records['sim_t'], def_records['sim_ag_power'], color=def_col)
    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_power'], color=cl_col)
    ax[4].set(xlabel=r'time ($s$)', ylabel=r'e_power (W)')

    lines = []
    labels = []

    for axis in ax:
        axLine, axLabel = axis.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
        
    fig.legend(lines, labels, loc = 'upper right', framealpha=.9, facecolor='white', frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def plot_evolution_classic(cl_records, filename):

    plt.style.use('seaborn')
    cmap = plt.cm.get_cmap('Spectral', 512)
    col = cmap(np.linspace(0, 1, 20))
    def_col = col[19]
    cl_col = col[16]
    def_atk_col = col[3]
    cl_atk_col = col[5]
    safe_col = col[0]
    lw=1
    
    fig, ax = plt.subplots(5, 1, figsize=(6, 9))

    ax[0].plot(cl_records['sim_t'], cl_records['sim_ag_pos'], label='classic follower', color=cl_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_env_pos'], label='classic leader', color=cl_atk_col)
    ax[0].set(ylabel=r'car position ($m$)')

    ax[1].plot(cl_records['sim_t'], cl_records['sim_ag_dist'], color=cl_col)
    ax[1].set(ylabel=r'distance ($m$)')
    ax[1].axhline(safe_dist_lower, ls='--', label='safe distance', color=safe_col, lw=lw)
    ax[1].axhline(safe_dist_upper, ls='--', color=safe_col, lw=lw)

    ax[2].plot(cl_records['sim_t'], cl_records['sim_ag_e_torque'], color=cl_col)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_env_e_torque'], color=cl_atk_col)
    ax[2].set(ylabel=r'e_torque ($N \cdot m$)')

    ax[3].plot(cl_records['sim_t'], cl_records['sim_ag_br_torque'], color=cl_col)
    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_br_torque'], color=cl_atk_col)
    ax[3].set(ylabel=r'br_torque ($N \cdot m$)')

    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_power'], color=cl_col)
    ax[4].set(xlabel=r'time ($s$)', ylabel=r'e_power (W)')

    lines = []
    labels = []

    for axis in ax:
        axLine, axLabel = axis.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
        
    fig.legend(lines, labels, loc = 'upper right', framealpha=.9, facecolor='white', frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def plot_evolution_fixed_env(def_records, cl_records, filename):

    plt.style.use('seaborn')
    cmap = plt.cm.get_cmap('Spectral', 512)
    col = cmap(np.linspace(0, 1, 20))
    def_col = col[19]
    cl_col = col[16]
    def_atk_col = col[3]
    cl_atk_col = col[5]
    safe_col = col[0]
    lw=1
    
    fig, ax = plt.subplots(5, 1, figsize=(6, 8))

    ax[0].plot(def_records['sim_t'], def_records['sim_ag_pos'], label='defender follower', color=def_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_ag_pos'], label='classic follower', color=cl_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_env_pos'], color=def_atk_col, lw=lw)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_env_pos'], label='leader', color=cl_atk_col, lw=lw)
    ax[0].set(ylabel=r'car position ($m$)')

    ax[1].plot(def_records['sim_t'], def_records['sim_ag_dist'], color=def_col)
    ax[1].plot(cl_records['sim_t'], cl_records['sim_ag_dist'], color=cl_col)
    ax[1].set(ylabel=r'distance ($m$)')
    ax[1].axhline(safe_dist_lower, ls='--', label='safe distance', color=safe_col, lw=lw)
    ax[1].axhline(safe_dist_upper, ls='--', color=safe_col, lw=lw)

    ax[2].plot(def_records['sim_t'], def_records['sim_ag_e_torque'], color=def_col)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_ag_e_torque'], color=cl_col)
    ax[2].plot(def_records['sim_t'], def_records['sim_env_e_torque'], color=def_atk_col, lw=lw)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_env_e_torque'], color=cl_atk_col, lw=lw)
    ax[2].set(ylabel=r'e_torque ($N \cdot m$)')

    ax[3].plot(def_records['sim_t'], def_records['sim_ag_br_torque'], color=def_col)
    ax[3].plot(cl_records['sim_t'], cl_records['sim_ag_br_torque'], color=cl_col)
    ax[3].plot(def_records['sim_t'], def_records['sim_env_br_torque'], color=def_atk_col, lw=lw)
    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_br_torque'], color=cl_atk_col, lw=lw)
    ax[3].set(ylabel=r'br_torque ($N \cdot m$)')

    ax[4].plot(def_records['sim_t'], def_records['sim_ag_power'], color=def_col)
    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_power'], color=cl_col)
    ax[4].set(xlabel=r'time ($s$)', ylabel=r'e_power (W)')

    lines = []
    labels = []

    for axis in ax:
        axLine, axLabel = axis.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
        
    fig.legend(lines, labels, loc = 'upper right', framealpha=.9, facecolor='white', frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter:
    size = len(records)

    robustness_computer = RobustnessComputer(robustness_dist, robustness_power, alpha)

    robustness_array = np.zeros(size)
    delta_pos_array = np.zeros(size)
    delta_vel_array = np.zeros(size)
    cl_robustness_array = np.zeros(size)
    cl_delta_pos_array = np.zeros(size)
    cl_delta_vel_array = np.zeros(size)

    for mode in ['atk']:

        for i in range(size):
            delta_pos = records[i][mode]['init']['env_pos'] - records[i][mode]['init']['ag_pos']
            delta_vel = records[i][mode]['init']['env_vel'] - records[i][mode]['init']['ag_vel']
            trace_dist = torch.tensor(records[i][mode]['sim_ag_dist'])
            trace_power = torch.tensor(records[i][mode]['sim_ag_power'])
            rob_dist = robustness_computer.dqs_dist.compute(dist=trace_dist, k=-150)
            rob_power = robustness_computer.dqs_power.compute(power=trace_power, k=-150)
            robustness = rob_dist
            # robustness = alpha*rob_dist+(1-alpha)*rob_power
            robustness_array[i] = robustness
            delta_pos_array[i] = delta_pos
            delta_vel_array[i] = delta_vel

            cl_delta_pos = records[i]['classic_'+mode]['init']['env_pos']-records[i]['classic_'+mode]['init']['ag_pos']
            cl_delta_vel = records[i]['classic_'+mode]['init']['env_vel']-records[i]['classic_'+mode]['init']['ag_vel']
            cl_trace_dist = torch.tensor(records[i]['classic_'+mode]['sim_ag_dist'])
            cl_trace_power = torch.tensor(records[i]['classic_'+mode]['sim_ag_power'])
            cl_rob_dist = robustness_computer.dqs_dist.compute(dist=cl_trace_dist, k=-150)
            cl_rob_power = robustness_computer.dqs_power.compute(power=cl_trace_power, k=-150)
            cl_robustness = cl_rob_dist
            # cl_robustness = alpha*cl_rob_dist+(1-alpha)*cl_rob_power
            cl_robustness_array[i] = cl_robustness
            cl_delta_pos_array[i] = cl_delta_pos
            cl_delta_vel_array[i] = cl_delta_vel

        scatter(robustness_array, delta_pos_array, delta_vel_array, 
                cl_robustness_array, cl_delta_pos_array, cl_delta_vel_array, 
                'platooning_energy_'+mode+'_scatterplot.png')

if args.plot_evolution:

    n=828 if len(records)>=1000 else random.randrange(len(records))
    # mode = 'const'
    # print(mode+" "+str(n)+":", records[n][mode]['init'])
    # plot_evolution_full(records[n][mode], records[n]["classic_"+mode], 'platooning_energy_evolution_'+mode+'_full.png')

    mode='atk'
    print(mode+" "+str(n)+":", records[n][mode]['init'])    
    plot_evolution_classic(records[n]["classic_"+mode], 'platooning_energy_evolution_'+mode+'_classic.png')
    plot_evolution_full(records[n][mode], records[n]["classic_"+mode], 'platooning_energy_evolution_'+mode+'_full.png')

    mode='pulse'
    print(mode+" "+str(n)+":", records[n][mode]['init'])
    plot_evolution_fixed_env(records[n][mode], records[n]["classic_"+mode], 'platooning_energy_evolution_'+mode+'_fixed.png')

if args.hist:
    size = len(records)

    pulse_pct = np.zeros_like(records[0]['pulse']['sim_ag_dist'])  
    atk_pct = np.zeros_like(records[0]['atk']['sim_ag_dist'])

    # robustness = lambda dist,power: alpha*np.logical_and(dist >= safe_dist1, dist <= safe_dist2)+\
    #                                 (1-alpha)*np.logical_and(power <= safe_power, power >= safe_power)
    robustness = lambda dist: np.logical_and(dist >= safe_dist_lower, dist <= safe_dist_upper)

    for i in range(size):

        dist = records[i]['pulse']['sim_ag_dist']
        # power = records[i]['pulse']['sim_ag_power']
        pulse_pct = pulse_pct + robustness(dist)#, power)

        dist = records[i]['atk']['sim_ag_dist']
        # power = records[i]['atk']['sim_ag_power']
        atk_pct = atk_pct + robustness(dist)#, power)

    time = records[0]['atk']['sim_t']
    pulse_pct = pulse_pct / size
    atk_pct = atk_pct / size

    hist(time, pulse_pct, atk_pct, 'platooning_energy_pct_histogram.png')

e_motor = ElMotor()
fig = e_motor.plotEffMap()
fig.savefig(os.path.join(EXP+relpath, 'efficiency_map.png'), dpi=150)