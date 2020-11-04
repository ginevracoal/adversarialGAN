import numpy as np

def get_settings(name, mode):

    robustness_formula = 'G(dist <= 10 & dist >= 2)'

    # PARAMS GRID
    
    if mode=="train":
        agent_position = 0
        agent_velocity = np.linspace(0, 20, 40)
        leader_position = np.linspace(1, 12, 15)
        leader_velocity = np.linspace(0, 20, 40)

    elif mode=="test":
        agent_position = 0
        agent_velocity = np.linspace(0, 5, 40) 
        leader_position = np.linspace(2, 10, 25)
        leader_velocity = np.linspace(0, 5, 40)

    # ARCHITECTURE

    if name=="default":

        atk_arch = {'hidden':2, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':5}
        train_par = {'train_steps':10, 'atk_steps':3, 'def_steps':5, 'horizon':5., \
                     'dt': 0.05, 'lr':0.001}
        test_par = {'test_steps':300, 'dt':0.05}

    return agent_position, agent_velocity, leader_position, leader_velocity, \
            atk_arch, def_arch, train_par, test_par, \
            robustness_formula