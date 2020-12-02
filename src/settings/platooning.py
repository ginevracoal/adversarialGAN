import numpy as np

def get_settings(name, mode):

    robustness_formula = 'G(dist <= 10 & dist >= 2)'

    # PARAMS GRID
    
    if mode=="train":
        agent_position = 0
        agent_velocity = np.linspace(0, 20, 100) 
        leader_position = np.linspace(4, 8, 100)
        leader_velocity = np.linspace(0, 20, 100)

    elif mode=="test":
        agent_position = 0
        agent_velocity = np.linspace(0, 10, 50) 
        leader_position = np.linspace(4, 8, 20)
        leader_velocity = np.linspace(0, 10, 50)

    # ARCHITECTURE

    if name=="default":

        atk_arch = {'hidden':1, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':1}
        train_par = {'train_steps':500, 'atk_steps':1, 'def_steps':3, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':200, 'dt':0.05}

    elif name=="energy":

        atk_arch = {'hidden':1, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':1}
        train_par = {'train_steps':300, 'atk_steps':1, 'def_steps':5, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':200, 'dt':0.05}

    else:
        raise NotImplementedError

    return agent_position, agent_velocity, leader_position, leader_velocity, \
            atk_arch, def_arch, train_par, test_par, robustness_formula
