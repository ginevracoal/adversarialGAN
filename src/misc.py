import os
import torch
import itertools
import numpy as np

EXP="../experiments/"

def get_relpath(main_dir, train_params):
    return main_dir+"_lr="+str(train_params["lr"])+"_dt="+str(train_params["dt"])+\
          "_horizon="+str(train_params["horizon"])+"_train_steps="+str(train_params["train_steps"])+\
          "_atk="+str(train_params["atk_steps"])+"_def="+str(train_params["def_steps"])

def get_sims_filename(repetitions, test_params):
    return 'sims_reps='+str(repetitions)+'_dt='+str(test_params["dt"])+\
           '_test_steps='+str(test_params["test_steps"])+'.pkl'

def save_models(attacker_model, defender_model, path):
    os.makedirs(path, exist_ok=True)

    atk_name = 'attacker_hidden='+str(attacker_model.hid)+\
               '_size='+str(attacker_model.ls)+\
               '_coef='+str(attacker_model.n_coeff)+\
               '_noise='+str(attacker_model.noise_size)+'.pt'

    def_name = 'defender_hidden='+str(defender_model.hid)+\
               '_size='+str(defender_model.ls)+\
               '_coef='+str(defender_model.n_coeff)+'.pt'

    atk_path = os.path.join(path, atk_name)
    def_path = os.path.join(path, def_name)

    torch.save(attacker_model.state_dict(), atk_path)
    torch.save(defender_model.state_dict(), def_path)

def load_models(attacker_model, defender_model, path):

    atk_name = 'attacker_hidden='+str(attacker_model.hid)+\
               '_size='+str(attacker_model.ls)+\
               '_coef='+str(attacker_model.n_coeff)+\
               '_noise='+str(attacker_model.noise_size)+'.pt'

    def_name = 'defender_hidden='+str(defender_model.hid)+\
               '_size='+str(defender_model.ls)+\
               '_coef='+str(defender_model.n_coeff)+'.pt'

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

    def sample(self, mu=0, sigma=1.):
        while True:
            yield [np.random.choice(r) + np.random.normal(mu, sigma)
                    if isinstance(r, np.ndarray) else float(r) for r in self._ranges]
