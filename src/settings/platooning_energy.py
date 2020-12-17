import numpy as np

def get_settings(name, mode):

    robustness_dist = 'G(dist <= 10 & dist >= 2)'
    robustness_power = 'G(power <= 100)'

    # PARAMS GRID
    
    if mode=="train":
        agent_position = np.random.uniform(0, 2, 100) 
        agent_velocity = np.random.uniform(15, 20, 100) 
        leader_position = np.random.uniform(4, 6, 100)
        leader_velocity = np.random.uniform(15, 20, 100)

    elif mode=="test":
        agent_position = np.random.uniform(0, 2, 150) 
        agent_velocity = np.random.uniform(15, 20, 150) 
        leader_position = np.random.uniform(4, 6, 150)
        leader_velocity = np.random.uniform(15, 20, 150)

    # ARCHITECTURE

    if name=="default":

        atk_arch = {'hidden':1, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':1}
        train_par = {'train_steps':500, 'atk_steps':1, 'def_steps':2, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':200, 'dt':0.05}

    elif name=="long":

        atk_arch = {'hidden':1, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':1}
        train_par = {'train_steps':1000, 'atk_steps':1, 'def_steps':2, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':200, 'dt':0.05}

    else:
        raise NotImplementedError

    return agent_position, agent_velocity, leader_position, leader_velocity, \
            atk_arch, def_arch, train_par, test_par, robustness_dist, robustness_power
