import os
import random
import pickle
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from argparse import ArgumentParser
from matplotlib.colors import Normalize, ListedColormap

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
        robustness_theta, robustness_dist, alpha = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="cartpole_target_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

safe_theta = 0.2
safe_dist = 0.5

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


def scatter(sims, sims_classic, filename, plot_differences=False):
    fig, ax = plt.subplots(2, 2, figsize=(6, 5))
    fig.tight_layout(pad=3.0)
    scatter_size=8

    cmap = plt.cm.get_cmap('Spectral')
    vmax = max([max(abs(sims['rob'])), max(abs(sims_classic['rob']))])
    vmin = -vmax

    plt.figtext(0.48, 0.95, 'Defender controller', ha='center', va='center', weight='bold')

    ax[0,0].scatter(sims['x'], sims['dot_x'], c=sims['rob'], 
                        cmap=cmap, vmin=vmin, vmax=vmax, s=scatter_size)
    ax[0,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    ax[0,1].scatter(sims['theta'], sims['dot_theta'], c=sims['rob'], 
                        cmap=cmap, vmin=vmin, vmax=vmax, s=scatter_size)
    ax[0,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')

    plt.figtext(0.48, 0.48, 'Classic controller', ha='center', va='center', weight='bold')

    ax[1,0].scatter(sims_classic['x'], sims_classic['dot_x'], c=sims_classic['rob'], 
                        cmap=cmap, vmin=vmin, vmax=vmax, s=scatter_size)
    ax[1,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[1,1].scatter(sims_classic['theta'], sims_classic['dot_theta'], c=sims_classic['rob'], 
                        cmap=cmap, vmin=vmin, vmax=vmax, s=scatter_size)
    ax[1,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.75])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('robustness', labelpad=-60)

    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)
    plt.close()

    if plot_differences:

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        fig.tight_layout(pad=3.0)

        robustness_differences = sims['rob']-sims_classic['rob']
        # vmax = max(abs(robustness_differences))
        vmax = max(robustness_differences)
        vmin = -vmax

        ax[0].scatter(sims['x'], sims['dot_x'], 
                        c=robustness_differences, cmap=cmap, vmin=vmin, vmax=vmax, s=scatter_size)
        ax[0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')

        im = ax[1].scatter(sims['theta'], sims['dot_theta'], 
                        c=robustness_differences, cmap=cmap, vmin=vmin, vmax=vmax, s=scatter_size)
        ax[1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
        
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.87, 0.22, 0.03, 0.6])
        cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
        cbar.set_label('Defender rob. - Classic rob.', labelpad=-61)
        plt.figtext(0.48, 0.9, 'Robustness difference vs initial configuration', ha='center', va='center', weight='bold')

        fig.savefig(os.path.join(EXP+relpath, "diff_"+filename), dpi=150)

def plot_evolution(def_records, cl_records, filename):

    plt.style.use('seaborn')
    cmap = plt.cm.get_cmap('Spectral', 512)
    col = cmap(np.linspace(0, 1, 20))
    def_col = col[19]
    cl_col = col[16]
    def_atk_col = col[3]
    cl_atk_col = col[5]
    safe_col = col[0]
    lw=1

    fig, ax = plt.subplots(5, 1, figsize=(6, 9), sharex=True)

    ax[0].plot(cl_records['sim_t'], cl_records['sim_x'], label='classic', color=cl_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_x_target'], label='cl. target', color=cl_atk_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_x'], label='defender',  color=def_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_x_target'], label='def. target', color=def_atk_col)
    ax[0].set(ylabel=r'cart position ($m$)')
    ax[0].legend()

    ax[1].axhline(-safe_dist, ls='--', color=safe_col, label="safe distance", lw=lw)
    ax[1].axhline(safe_dist, ls='--', color=safe_col, lw=lw)
    ax[1].plot(def_records['sim_t'], def_records['sim_dist'], color=def_col, label='defender')    
    ax[1].plot(def_records['sim_t'], cl_records['sim_dist'], color=cl_col, label='classic')    
    ax[1].set(ylabel=r'distance from target ($m$)')
    ax[1].legend()

    ax[2].axhline(-safe_theta, ls='--', color=safe_col, label="safe angle", lw=lw)
    ax[2].axhline(safe_theta, ls='--', color=safe_col, lw=lw)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_theta'], color=cl_col)
    ax[2].plot(def_records['sim_t'], def_records['sim_theta'], color=def_col)
    ax[2].set(ylabel=r'pole angle ($rad$)')
    ax[2].legend()

    ax[3].plot(def_records['sim_t'], def_records['sim_env_mu'], color=def_atk_col, label='defender env.')
    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_mu'], color=cl_atk_col, label='classic env.')
    ax[3].set(ylabel=r'friction coefficient')
    ax[3].legend()

    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_action'], label='classic', color=cl_col)
    ax[4].plot(def_records['sim_t'], def_records['sim_ag_action'], label='defender', color=def_col)
    ax[4].set(xlabel=r'time ($s$)')
    ax[4].set(ylabel= r'cart control ($N$)')

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

    fig, ax = plt.subplots(5, 1, figsize=(6, 9), sharex=True)

    ax[0].plot(cl_records['sim_t'], cl_records['sim_x'], label='classic', color=cl_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_x_target'], label='cl. target', color=def_atk_col)
    ax[0].set(ylabel=r'cart position ($m$)')
    ax[0].legend()

    ax[1].axhline(-safe_dist, ls='--', color=safe_col, label="safe distance", lw=lw)
    ax[1].axhline(safe_dist, ls='--', color=safe_col, lw=lw)
    ax[1].plot(cl_records['sim_t'], cl_records['sim_dist'], color=cl_col, label='classic')    
    ax[1].set(ylabel=r'distance from target ($m$)')
    ax[1].legend()

    ax[2].axhline(-safe_theta, ls='--', color=safe_col, label="safe angle", lw=lw)
    ax[2].axhline(safe_theta, ls='--', color=safe_col, lw=lw)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_theta'], color=cl_col)
    ax[2].set(ylabel=r'pole angle ($rad$)')
    ax[2].legend()

    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_mu'], color=def_atk_col, label='classic env.')
    ax[3].set(ylabel=r'friction coefficient')
    ax[3].legend()

    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_action'], label='classic', color=cl_col)
    ax[4].set(xlabel=r'time ($s$)')
    ax[4].set(ylabel= r'cart control ($N$)')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

def plot_evolution_pulse(def_records, cl_records, filename):

    plt.style.use('seaborn')
    cmap = plt.cm.get_cmap('Spectral', 512)
    col = cmap(np.linspace(0, 1, 20))
    def_col = col[19]
    cl_col = col[16]
    def_atk_col = col[3]
    cl_atk_col = col[5]
    safe_col = col[0]
    lw=1

    fig, ax = plt.subplots(5, 1, figsize=(6, 9), sharex=True)

    ax[0].plot(cl_records['sim_t'], cl_records['sim_x'], label='classic controller', color=cl_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_x'], label='defender controller',  color=def_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_x_target'], color=def_atk_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_x_target'], label='target position', color=cl_atk_col)
    ax[0].set(ylabel=r'cart position ($m$)')
    ax[0].legend()

    ax[1].axhline(-safe_dist, ls='--', color=safe_col, label="safe distance", lw=lw)
    ax[1].axhline(safe_dist, ls='--', color=safe_col, lw=lw)
    ax[1].plot(def_records['sim_t'], def_records['sim_dist'], color=def_col)    
    ax[1].plot(def_records['sim_t'], cl_records['sim_dist'], color=cl_col)    
    ax[1].set(ylabel=r'distance from target ($m$)')
    ax[1].legend()

    ax[2].axhline(-safe_theta, ls='--', color=safe_col, label="safe angle", lw=lw)
    ax[2].axhline(safe_theta, ls='--', color=safe_col, lw=lw)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_theta'], color=cl_col)
    ax[2].plot(def_records['sim_t'], def_records['sim_theta'], color=def_col)
    ax[2].set(ylabel=r'pole angle ($rad$)')
    ax[2].legend()

    ax[3].plot(def_records['sim_t'], def_records['sim_env_mu'], color=def_atk_col)
    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_mu'], color=cl_atk_col, label='environment')
    ax[3].set(ylabel=r'friction coefficient')
    ax[3].legend()

    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_action'], label='classic', color=cl_col)
    ax[4].plot(def_records['sim_t'], def_records['sim_ag_action'], label='defender', color=def_col)
    ax[4].set(xlabel=r'time ($s$)')
    ax[4].set(ylabel= r'cart control ($N$)')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter is True:

    size = len(records)

    robustness_computer = RobustnessComputer(robustness_theta, robustness_dist, alpha)

    for case in ['atk', 'pulse']:

        rob_dict = {case:{}, 'classic_'+case:{}}

        for mode in [case, 'classic_'+case]:

            robustness_array = np.zeros(size)
            cart_pos_array = np.zeros(size)
            pole_ang_array = np.zeros(size)
            cart_vel_array = np.zeros(size)
            pole_ang_vel_array = np.zeros(size)

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

            rob_dict[mode]['rob'] = robustness_array
            rob_dict[mode]['x'] = cart_pos_array
            rob_dict[mode]['theta'] = pole_ang_array
            rob_dict[mode]['dot_x'] = cart_vel_array
            rob_dict[mode]['dot_theta'] = pole_ang_vel_array

        scatter(rob_dict[case], rob_dict['classic_'+case], 'cartpole_target_'+case+'_robustness_scatterplot.png',
                plot_differences=True)


if args.plot_evolution is True:

    n=317 if len(records)>=1000 else random.randrange(len(records))
    
    # mode = 'const'
    # print(mode+" "+str(n)+":", records[n][mode]['init'])
    # plot_evolution(records[n][mode], records[n]["classic_"+mode], 'cartpole_target_evolution_'+mode+'.png')

    mode = 'atk'
    print(mode+" "+str(n)+":", records[n][mode]['init'])
    plot_evolution_classic(records[n]["classic_"+mode], 'cartpole_target_evolution_'+mode+'.png')

    n=88 if len(records)>=1000 else random.randrange(len(records))

    mode = "pulse"
    print(mode+" "+str(n)+":", records[n][mode]['init'])
    plot_evolution_pulse(records[n][mode], records[n]["classic_"+mode], 'cartpole_target_evolution_'+mode+'.png')

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

    hist(time, const_pct, pulse_pct, atk_pct, 'cartpole_target_pct_histogram.png')
