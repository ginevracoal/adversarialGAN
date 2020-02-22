import os
import torch

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

