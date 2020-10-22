import os
import torch
import itertools
import numpy as np

EXP="../experiments/"

def save_models(attacker_model, defender_model, path):
    os.makedirs(path, exist_ok=True)

    atk_name = 'attacker_hidden='+str(attacker_model.hid)+\
               '_size='+str(attacker_model.ls)+\
               '_degree='+str(attacker_model.n_coeff-1)+\
               '_noise='+str(attacker_model.noise_size)+'.pt'

    def_name = 'defender_hidden='+str(defender_model.hid)+\
               '_size='+str(defender_model.ls)+\
               '_degree='+str(defender_model.n_coeff-1)+'.pt'

    atk_path = os.path.join(path, atk_name)
    def_path = os.path.join(path, def_name)

    torch.save(attacker_model.state_dict(), atk_path)
    torch.save(defender_model.state_dict(), def_path)

def load_models(attacker_model, defender_model, path):

    atk_name = 'attacker_hidden='+str(attacker_model.hid)+\
               '_size='+str(attacker_model.ls)+\
               '_degree='+str(attacker_model.n_coeff-1)+\
               '_noise='+str(attacker_model.noise_size)+'.pt'

    def_name = 'defender_hidden='+str(defender_model.hid)+\
               '_size='+str(defender_model.ls)+\
               '_degree='+str(defender_model.n_coeff-1)+'.pt'

    atk_path = os.path.join(path, atk_name)
    def_path = os.path.join(path, def_name)

    attacker_model.load_state_dict(torch.load(atk_path))
    defender_model.load_state_dict(torch.load(def_path))


class ParametersHyperparallelepiped:
    """ Class used to sample from the hyper-grid of parameters.
        It also adds some gaussian noise to the sampled point in
        order to encourage the exploration of the space.
    """

    def __init__(self, *ranges):
        self._ranges = ranges

    def sample(self, mu=0, sigma=1):
        while True:
            yield [np.random.choice(r) + np.random.normal(mu, sigma)
                    if isinstance(r, np.ndarray) else float(r) for r in self._ranges]
