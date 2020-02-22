import os
import torch
import itertools
import numpy as np

def save_models(attacker_model, defender_model, path):
    destination = os.path.join(path, 'models')

    if not os.path.isdir(destination):
        os.mkdir(destination)

    atk_path = os.path.join(destination, 'attacker.pt')
    def_path = os.path.join(destination, 'defender.pt')

    torch.save(attacker_model.state_dict(), atk_path)
    torch.save(defender_model.state_dict(), def_path)

def load_models(attacker_model, defender_model, path):
    atk_path = os.path.join(path, 'models', 'attacker.pt')
    def_path = os.path.join(path, 'models', 'defender.pt')

    attacker_model.load_state_dict(torch.load(atk_path))
    defender_model.load_state_dict(torch.load(def_path))


class ParametersHyperparallelepiped:

    def __init__(self, *ranges):
        self._ranges = ranges

    def sample(self, mu=0, sigma=1):
        while True:
            yield [np.random.choice(r) + np.random.normal(mu, sigma) for r in self._ranges]
