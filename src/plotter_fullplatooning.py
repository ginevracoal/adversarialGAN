import os
import random
import pickle

import model_platooning
import misc
import architecture

import torch

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

class Leader:
    def __init__(self, position=.0, velocity=1.):
        self.car = model_platooning.Car()
        self.car.position = torch.tensor(position)
        self.car.velocity = torch.tensor(velocity)
        
    def set_follower(self, follower):
        self._f = follower
        
    @property
    def status(self):
        return (self.car.velocity,
               self._f.car.velocity,
               self.car.position - self._f.car.position)
    
    def update(self, acc):
        self.car.update(acc, dt)
        
class Follower:
    def __init__(self, position=.0, velocity=1.):
        self.car = model_platooning.Car()
        self.car.position = torch.tensor(position)
        self.car.velocity = torch.tensor(velocity)

    def set_followed(self, followed):
        self._f = followed
        
    @property
    def status(self):
        return (self._f.car.velocity,
               self.car.velocity,
               self._f.car.position - self.car.position)
    
    def update(self, acc):
        self.car.update(acc, dt)

parser = ArgumentParser()
parser.add_argument("-d", "--dir", dest="dirname", help="model's directory")
parser.add_argument("-n", "--nfollowers", dest="nfollowers", type=int, default=1, help="number of followers")
parser.add_argument("--dark", default=False, action="store_true" , help="Use dark theme")
args = parser.parse_args()

if args.dark:
    plt.style.use('./qb-common_dark.mplstyle')

physical_model = model_platooning.Model(None)
attacker = architecture.Attacker(physical_model, 2, 10, 2)
defender = architecture.Defender(physical_model, 2, 10)

misc.load_models(attacker, defender, args.dirname)

dt = 0.05
steps = 300

def run(n_followers=1, mode=None):
    init_dist = random.uniform(1, 8)
    init_vel = random.uniform(1,5)
    print(mode, 'd:', init_dist, 'v:', init_vel)

    leader = Leader(float(n_followers * init_dist), init_vel)
    followers = [Follower(init_dist * (float(n_followers) - i - 1), init_vel) for i in range(n_followers)]

    leader.set_follower(followers[0])
    followers[0].set_followed(leader)
    for i in range(n_followers -1):
        followers[i + 1].set_followed(followers[i])

    sim_time = []
    sim_leader_pos = []
    sim_followers_pos = []
    sim_leader_acc = []
    sim_followers_acc = []

    t = 0
    for i in range(steps):
        with torch.no_grad():
            oas = [torch.tensor(f.status) for f in followers]
            oe = torch.tensor(leader.status)
            z = torch.rand(attacker.noise_size)
            if mode == 0:
                atk_policy = lambda x: torch.tensor(2.) if i > 200 and i < 250 else torch.tensor(-2.)
            elif mode == 1:
                atk_policy = lambda x: torch.tensor(2.) if i > 150 else torch.tensor(-2.)
            elif mode == 2:
                atk_policy = lambda x: torch.tensor(2.) if i < 150 else torch.tensor(-2.)
            else:
                atk_policy = attacker(torch.cat((z, oe)))
            def_policies = [defender(oa) for oa in oas]

        atk_input = atk_policy(dt)
        def_inputs = [def_policy(dt) for def_policy in def_policies]
            
        leader.update(atk_input)
        _ = [followers[j].update(def_inputs[j]) for j in range(n_followers)]
        
        sim_followers_acc.append(def_inputs)
        sim_leader_acc.append(atk_input)
        sim_time.append(t)
        sim_followers_pos.append([f.car.position.clone() for f in followers])
        sim_leader_pos.append(leader.car.position.clone())
        t += dt

    sim_followers_pos = np.array(sim_followers_pos).T
    sim_followers_acc = np.array(sim_followers_acc).T
    
    return(sim_time, sim_leader_pos, sim_leader_acc, sim_followers_pos, sim_followers_acc)


def plot(n_followers, sim_time, sim_leader_pos, sim_leader_acc, sim_followers_pos, sim_followers_acc, filename):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    for i in reversed(range(n_followers)):
        ax[0].plot(sim_time, sim_followers_pos[i], label='follower ' + str(i+1))
    ax[0].plot(sim_time, sim_leader_pos, label='leader')
    ax[0].set(xlabel='time (s)', ylabel='position (m)')
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(reversed(handles), reversed(labels))

    for i in reversed(range(1, n_followers)):
        ax[1].plot(sim_time, sim_followers_pos[i-1] - sim_followers_pos[i], label='follower ' + str(i+1))
    ax[1].plot(sim_time, sim_leader_pos - sim_followers_pos[0], label='follower 1')
    ax[1].set(xlabel='time (s)', ylabel='distance (m)')
    ax[1].axhline(2, ls='--', color='r')
    ax[1].axhline(10, ls='--', color='r')
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(reversed(handles), reversed(labels))

    for i in reversed(range(n_followers)):
        ax[2].plot(sim_time, np.clip(sim_followers_acc[i], -3, 3), label='follower ' + str(i+1))
    ax[2].plot(sim_time, np.clip(sim_leader_acc, -3, 3), label='leader')
    ax[2].set(xlabel='time (s)', ylabel='acceleration ($m/s^2$)')
    handles, labels = ax[2].get_legend_handles_labels()
    ax[2].legend(reversed(handles), reversed(labels))

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)


sim = run(args.nfollowers)
plot(args.nfollowers, *sim, 'triplot_fullplatooning_attacker.png')
sim = run(args.nfollowers, 0)
plot(args.nfollowers, *sim, 'triplot_fullplatooning_pulse.png')
sim = run(args.nfollowers, 1)
plot(args.nfollowers, *sim, 'triplot_fullplatooning_stepup.png')
sim = run(args.nfollowers, 2)
plot(args.nfollowers, *sim, 'triplot_fullplatooning_stepdown.png')
