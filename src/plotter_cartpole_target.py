import os
import random
import pickle
import torch
import matplotlib as mpl
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
        robustness_theta, robustness_dist, \
        alpha, safe_theta, safe_dist, norm_theta, norm_dist = get_settings(args.architecture, mode="test")
relpath = get_relpath(main_dir="cartpole_target_"+args.architecture, train_params=train_par)
sims_filename = get_sims_filename(args.repetitions, test_par)

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

scatter_size=8

def scatter(sims, sims_classic, filename):

    fig, ax = plt.subplots(2, 2, figsize=(6, 5))
    fig.tight_layout(pad=3.0)

    cmap = plt.cm.get_cmap('Spectral')
    vmax = max([max(sims['rob']), max(sims_classic['rob']), 0.000001])
    vmin = min([min(sims['rob']), min(sims_classic['rob']), -0.000001])
    norm = mpl.colors.TwoSlopeNorm(vcenter=0., vmax=vmax, vmin=vmin)

    plt.figtext(0.48, 0.95, 'Defender controller', ha='center', va='center', weight='bold')

    ax[0,0].scatter(sims['x'], sims['dot_x'], c=sims['rob'],
                        cmap=cmap, norm=norm, s=scatter_size)
    ax[0,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    ax[0,1].scatter(sims['theta'], sims['dot_theta'], c=sims['rob'],
                        cmap=cmap, norm=norm,  s=scatter_size)
    ax[0,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')

    plt.figtext(0.48, 0.48, 'Classic controller', ha='center', va='center', weight='bold')

    ax[1,0].scatter(sims_classic['x'], sims_classic['dot_x'], c=sims_classic['rob'], 
                        cmap=cmap, norm=norm,  s=scatter_size)
    ax[1,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[1,1].scatter(sims_classic['theta'], sims_classic['dot_theta'], c=sims_classic['rob'], 
                        cmap=cmap, norm=norm,  s=scatter_size)
    ax[1,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.88, 0.14, 0.03, 0.75])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('robustness', labelpad=-60)

    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)
    plt.close()

def scatter_diff(sims, sims_classic, filename):

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    fig.tight_layout(pad=3.0)

    cmap = plt.cm.get_cmap('Spectral')
    robustness_differences = sims['rob']-sims_classic['rob']
    norm = mpl.colors.TwoSlopeNorm(vcenter=0)

    ax[0].scatter(sims['x'], sims['dot_x'], 
                    c=robustness_differences, cmap=cmap, norm=norm, s=scatter_size)
    ax[0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')

    im = ax[1].scatter(sims['theta'], sims['dot_theta'], 
                    c=robustness_differences, cmap=cmap, norm=norm, s=scatter_size)
    ax[1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.87, 0.22, 0.03, 0.6])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Defender rob. - Classic rob.', labelpad=-61)
    plt.figtext(0.48, 0.9, 'Robustness difference vs initial configuration', ha='center', va='center', weight='bold')

    fig.savefig(os.path.join(EXP+relpath, "diff_"+filename), dpi=150)

def scatter_diff_sep(sims, sims_classic, filename):

    fig, ax = plt.subplots(2, 2, figsize=(6, 5))
    fig.tight_layout(pad=3.0)
    cmap = plt.cm.get_cmap('Spectral')

    plt.figtext(0.45, 0.94, 'Robustness differences on the distance', ha='center', va='center', weight='bold')

    rob_dist_diff = sims['rob_dist']-sims_classic['rob_dist']
    norm = mpl.colors.TwoSlopeNorm(vcenter=0)

    ax[0,0].scatter(sims['x'], sims['dot_x'], c=rob_dist_diff, cmap=cmap, norm=norm, s=scatter_size)
    ax[0,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[0,1].scatter(sims['theta'], sims['dot_theta'], c=rob_dist_diff, cmap=cmap, norm=norm, s=scatter_size)
    ax[0,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.87, 0.6, 0.03, 0.3])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Defender rob. - Classic rob.', labelpad=-80)

    plt.figtext(0.45, 0.47, 'Robustness differences on the angle', ha='center', va='center', weight='bold')
    
    rob_theta_diff = sims['rob_theta']-sims_classic['rob_theta']
    norm = mpl.colors.TwoSlopeNorm(vcenter=0)

    ax[1,0].scatter(sims['x'], sims['dot_x'], c=rob_theta_diff, cmap=cmap, norm=norm, s=scatter_size)
    ax[1,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[1,1].scatter(sims['theta'], sims['dot_theta'], c=rob_theta_diff, cmap=cmap, norm=norm, s=scatter_size)
    ax[1,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.87, 0.14, 0.03, 0.3])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Defender rob. - Classic rob.', labelpad=-61)

    fig.savefig(os.path.join(EXP+relpath, "diff_sep_"+filename), dpi=150)

def scatter_sep(sims, sims_classic, filename):

    vmin=min(min(sims['rob_dist']), min(sims['rob_theta']), min(sims_classic['rob_dist']), min(sims_classic['rob_theta']), -0.000001)
    vmax=max(max(sims['rob_dist']), max(sims['rob_theta']), max(sims_classic['rob_dist']), max(sims_classic['rob_theta']), 0.000001)
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmax=vmax, vmin=vmin)
    cmap = plt.cm.get_cmap('Spectral')

    ##### fig Defender

    fig, ax = plt.subplots(2, 2, figsize=(6, 4.5))
    fig.tight_layout(pad=3.0)

    plt.figtext(0.45, 0.94, 'Defender robustness on the distance', ha='center', va='center', weight='bold')

    ax[0,0].scatter(sims['x'], sims['dot_x'], c=sims['rob_dist'], cmap=cmap, norm=norm, s=scatter_size)
    ax[0,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[0,1].scatter(sims['theta'], sims['dot_theta'], c=sims['rob_dist'], cmap=cmap, norm=norm, s=scatter_size)
    ax[0,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    plt.figtext(0.45, 0.48, 'Defender robustness on the angle', ha='center', va='center', weight='bold')
    
    ax[1,0].scatter(sims['x'], sims['dot_x'], c=sims['rob_theta'], cmap=cmap, norm=norm, s=scatter_size)
    ax[1,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[1,1].scatter(sims['theta'], sims['dot_theta'], c=sims['rob_theta'], cmap=cmap, norm=norm, s=scatter_size)
    ax[1,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.87, 0.18, 0.03, 0.7])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Robustness', labelpad=-70)

    fig.savefig(os.path.join(EXP+relpath, "defender_sep_"+filename), dpi=150)

    ##### fig Classic

    fig, ax = plt.subplots(2, 2, figsize=(6, 4.5))
    fig.tight_layout(pad=3.0)
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmax=vmax, vmin=vmin)

    plt.figtext(0.45, 0.94, 'Classic robustness on the distance', ha='center', va='center', weight='bold')

    ax[0,0].scatter(sims_classic['x'], sims_classic['dot_x'], c=sims_classic['rob_dist'], cmap=cmap, norm=norm, s=scatter_size)
    ax[0,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[0,1].scatter(sims_classic['theta'], sims_classic['dot_theta'], c=sims_classic['rob_dist'], cmap=cmap, norm=norm, s=scatter_size)
    ax[0,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')

    plt.figtext(0.45, 0.48, 'Classic robustness on the angle', ha='center', va='center', weight='bold')
    
    ax[1,0].scatter(sims_classic['x'], sims_classic['dot_x'], c=sims_classic['rob_theta'], cmap=cmap, norm=norm, s=scatter_size)
    ax[1,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[1,1].scatter(sims_classic['theta'], sims['dot_theta'], c=sims_classic['rob_theta'], cmap=cmap, norm=norm, s=scatter_size)
    ax[1,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.87, 0.18, 0.03, 0.7])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('Robustness', labelpad=-70)

    fig.savefig(os.path.join(EXP+relpath, "classic_sep_"+filename), dpi=150)

def scatter_full(sims, sims_classic, filename):
    fig, ax = plt.subplots(3, 2, figsize=(6, 7))
    fig.tight_layout(pad=3.0)

    cmap = plt.cm.get_cmap('Spectral')
    vmax = max([max(sims['rob']), max(sims_classic['rob']), 0.000001])
    vmin = min([min(sims['rob']), min(sims_classic['rob']), -0.000001])
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmax=vmax, vmin=vmin)

    plt.figtext(0.48, 0.95, 'Defender controller', ha='center', va='center', weight='bold')

    ax[0,0].scatter(sims['x'], sims['dot_x'], c=sims['rob'], 
                        cmap=cmap, norm=norm, s=scatter_size)
    ax[0,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    ax[0,1].scatter(sims['theta'], sims['dot_theta'], c=sims['rob'], 
                        cmap=cmap, norm=norm, s=scatter_size)
    ax[0,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')

    plt.figtext(0.48, 0.64, 'Classic controller', ha='center', va='center', weight='bold')

    ax[1,0].scatter(sims_classic['x'], sims_classic['dot_x'], c=sims_classic['rob'], 
                        cmap=cmap, norm=norm, s=scatter_size)
    ax[1,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')
    im = ax[1,1].scatter(sims_classic['theta'], sims_classic['dot_theta'], c=sims_classic['rob'], 
                        cmap=cmap, norm=norm, s=scatter_size)
    ax[1,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')

    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.88, 0.42, 0.03, 0.5])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('robustness', labelpad=-55)

    plt.figtext(0.48, 0.32, 'Defender rob. - Classic rob.', ha='center', va='center', weight='bold')

    robustness_differences = sims['rob']-sims_classic['rob']
    norm = mpl.colors.TwoSlopeNorm(vcenter=0)

    ax[2,0].scatter(sims['x'], sims['dot_x'], 
                    c=robustness_differences, cmap=cmap, norm=norm, s=scatter_size)
    ax[2,0].set(xlabel=r'cart position ($m$)', ylabel=r'cart velocity ($m/s$)')

    im = ax[2,1].scatter(sims['theta'], sims['dot_theta'], 
                    c=robustness_differences, cmap=cmap, norm=norm, s=scatter_size)
    ax[2,1].set(xlabel=r'pole angle ($rad$)', ylabel=r'pole ang. freq. ($rad/s$)')
    
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.2])
    cbar = fig.colorbar(im, ax=ax.ravel().tolist(), cax=cbar_ax)
    cbar.set_label('robustness difference', labelpad=-60)

    fig.savefig(os.path.join(EXP+relpath, "full_"+filename), dpi=150)
    plt.close()

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

    ax[4].plot(def_records['sim_t'], def_records['sim_ag_action'], label='defender', color=def_col)
    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_action'], label='classic', color=cl_col)
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

    fig, ax = plt.subplots(5, 1, figsize=(6, 7), sharex=True)

    ax[0].plot(cl_records['sim_t'], cl_records['sim_x'], label='classic', color=cl_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_x_target'], label='cl. target', color=def_atk_col)
    ax[0].set(ylabel=r'cart position ($m$)')
    ax[0].legend()

    ax[1].axhline(-safe_dist, ls='--', color=safe_col, label="safe distance", lw=lw)
    ax[1].axhline(safe_dist, ls='--', color=safe_col, lw=lw)
    ax[1].plot(cl_records['sim_t'], cl_records['sim_dist'], color=cl_col, label='classic')    
    ax[1].set(ylabel='distance from \ntarget ($m$)')
    ax[1].legend()

    ax[2].axhline(-safe_theta, ls='--', color=safe_col, label="safe angle", lw=lw)
    ax[2].axhline(safe_theta, ls='--', color=safe_col, lw=lw)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_theta'], color=cl_col)
    ax[2].set(ylabel=r'pole angle ($rad$)')
    ax[2].legend()

    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_mu'], color=def_atk_col, label='classic env.')
    ax[3].set(ylabel='friction\ncoefficient')
    ax[3].legend()

    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_action'], label='classic', color=cl_col)
    ax[4].set(xlabel=r'time ($s$)')
    ax[4].set(ylabel= r'cart control ($N$)')

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

    fig, ax = plt.subplots(5, 1, figsize=(6, 7), sharex=True)

    ax[0].plot(cl_records['sim_t'], cl_records['sim_x'], color=cl_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_x'], color=def_col)
    ax[0].plot(def_records['sim_t'], def_records['sim_x_target'], color=def_atk_col)
    ax[0].plot(cl_records['sim_t'], cl_records['sim_x_target'], label='target position', color=cl_atk_col)
    ax[0].set(ylabel=r'cart position ($m$)')
    ax[0].legend()

    ax[1].axhline(-safe_dist, ls='--', color=safe_col, label="safe distance", lw=lw)
    ax[1].axhline(safe_dist, ls='--', color=safe_col, lw=lw)
    ax[1].plot(def_records['sim_t'], def_records['sim_dist'], label='defender controller', color=def_col)    
    ax[1].plot(def_records['sim_t'], cl_records['sim_dist'], label='classic controller', color=cl_col)    
    ax[1].set(ylabel='distance from\n target ($m$)')
    ax[1].legend()

    ax[2].axhline(-safe_theta, ls='--', color=safe_col, label="safe angle", lw=lw)
    ax[2].axhline(safe_theta, ls='--', color=safe_col, lw=lw)
    ax[2].plot(cl_records['sim_t'], cl_records['sim_theta'], color=cl_col)
    ax[2].plot(def_records['sim_t'], def_records['sim_theta'], color=def_col)
    ax[2].set(ylabel=r'pole angle ($rad$)')
    ax[2].legend()

    ax[3].plot(def_records['sim_t'], def_records['sim_env_mu'], color=def_atk_col)
    ax[3].plot(cl_records['sim_t'], cl_records['sim_env_mu'], color=cl_atk_col, label='environment')
    ax[3].set(ylabel='friction \ncoefficient')
    ax[3].legend()

    ax[4].plot(def_records['sim_t'], def_records['sim_ag_action'], label='defender', color=def_col)
    ax[4].plot(cl_records['sim_t'], cl_records['sim_ag_action'], label='classic', color=cl_col)
    ax[4].set(xlabel=r'time ($s$)')
    ax[4].set(ylabel= r'cart control ($N$)')

    fig.tight_layout()
    fig.savefig(os.path.join(EXP+relpath, filename), dpi=150)

if args.scatter is True:

    size = len(records)

    robustness_computer = RobustnessComputer(formula_theta=robustness_theta, formula_dist=robustness_dist, 
        alpha=alpha, norm_theta=norm_theta, norm_dist=norm_dist)

    rob_dicts = {'atk':{}, 'pulse':{}}

    for env in ['atk','pulse']:

        rob_dict = {env:{}, 'classic_'+env:{}}

        for mode in [env, 'classic_'+env]:

            rob_dist_array = np.zeros(size)
            rob_theta_array = np.zeros(size)
            robustness_array = np.zeros(size)
            cart_pos_array = np.zeros(size)
            pole_ang_array = np.zeros(size)
            cart_vel_array = np.zeros(size)
            pole_ang_vel_array = np.zeros(size)

            for i in range(size):

                trace_theta = torch.tensor(records[i][mode]['sim_theta'])
                trace_dist = torch.tensor(records[i][mode]['sim_dist'])
                rob_theta = robustness_computer.dqs_theta.compute(theta=trace_theta, k=0)/norm_theta
                rob_dist = robustness_computer.dqs_dist.compute(dist=trace_dist, k=0)/norm_dist
                robustness = alpha*rob_dist+(1-alpha)*rob_theta
                cart_pos = records[i][mode]['init']['x'] 
                pole_ang = records[i][mode]['init']['theta'] 
                cart_vel = records[i][mode]['init']['dot_x'] 
                pole_ang_vel = records[i][mode]['init']['dot_theta'] 

                rob_dist_array[i] = rob_dist
                rob_theta_array[i] = rob_theta
                robustness_array[i] = robustness
                cart_pos_array[i] = cart_pos
                pole_ang_array[i] = pole_ang
                cart_vel_array[i] = cart_vel
                pole_ang_vel_array[i] = pole_ang_vel

            rob_dict[mode]['rob_dist'] = rob_dist_array
            rob_dict[mode]['rob_theta'] = rob_theta_array
            rob_dict[mode]['rob'] = robustness_array
            rob_dict[mode]['x'] = cart_pos_array
            rob_dict[mode]['theta'] = pole_ang_array
            rob_dict[mode]['dot_x'] = cart_vel_array
            rob_dict[mode]['dot_theta'] = pole_ang_vel_array

        rob_dicts[env] = rob_dict

    filename = 'cartpole_target_atk_robustness_scatterplot.png'
    # scatter(rob_dicts['atk']['atk'], rob_dicts['atk']['classic_atk'], filename)
    # scatter_diff(rob_dicts['atk']['atk'], rob_dicts['atk']['classic_atk'], filename)
    # scatter_full(rob_dicts['atk']['atk'], rob_dicts['atk']['classic_atk'], filename)
    scatter_sep(rob_dicts['atk']['atk'], rob_dicts['atk']['classic_atk'], filename)

    n_atk = np.where(rob_dicts['atk']['classic_atk']['rob_theta']==min(rob_dicts['atk']['classic_atk']['rob_theta']))[0][0]
    
    rob = rob_dicts['pulse']['pulse']['rob_dist']
    rob_cl = rob_dicts['pulse']['classic_pulse']['rob']
    n_pulse = np.where((rob>0)&(rob_cl>0))[0][0]

if args.plot_evolution is True:

    n = n_atk
    mode = 'atk'
    print(mode+" "+str(n)+":", records[n][mode]['init'])
    plot_evolution_full(records[n][mode], records[n]["classic_"+mode], 'cartpole_target_evolution_'+mode+'_full.png')
    plot_evolution_classic(records[n]["classic_"+mode], 'cartpole_target_evolution_'+mode+'_classic.png')

    n = n_pulse
    mode = "pulse"
    print(mode+" "+str(n)+":", records[n][mode]['init'])
    plot_evolution_fixed_env(records[n][mode], records[n]["classic_"+mode], 'cartpole_target_evolution_'+mode+'_fixed.png')

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
    
    
def save_results_for_cl_ctrl(n_list):
    
    import pickle
    
    save_dict = {}
    
    print(f'chosen simulations: {n_list}')
    #n = np.random.randint(len(records))
    
    for i in n_list:
        cl_records = records[i]["classic_" + mode]
        init_cond = np.array([ cl_records['sim_x'][0] , cl_records['sim_dot_x'][0]  , cl_records['sim_theta'][0]  , cl_records['sim_dot_theta'][0] ])
        save_dict[i] = (cl_records['sim_x_target'], cl_records['sim_env_mu'], init_cond)

    with open('test_case.npy', 'wb') as f:
        pickle.dump(save_dict,f)
        #np.save(f, save_dict)

# n_list = random.sample(range(len(records)), 10)            
# save_results_for_cl_ctrl(n_list)