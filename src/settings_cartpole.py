import numpy as np

def get_settings(name, mode):

    safe_theta = 0.2
    safe_dist = 0.5
    robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta})'

    # PARAMS GRID
    
    if mode=="train":
        cart_position = np.linspace(0., 0., 1)
        cart_velocity = np.linspace(-1., 1., 100)
        pole_angle = np.linspace(-0., 0., 1)
        pole_ang_velocity = np.linspace(-1., 1., 100)

    elif mode=="test":
        cart_position = np.linspace(0., 0., 1)
        cart_velocity = np.linspace(-.2, .2, 50)
        pole_angle = np.linspace(-0., 0., 1)
        pole_ang_velocity = np.linspace(-.2, .2, 50)

    # ARCHITECTURE

    if name=="default":

        atk_arch = {'hidden':2, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':1}
        train_par = {'train_steps':10000, 'atk_steps':5, 'def_steps':8, 'horizon':10., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':100, 'dt':0.05}
    
    elif name=="testing":

        atk_arch = {'hidden':2, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':1}
        train_par = {'train_steps':100, 'atk_steps':5, 'def_steps':8, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':300, 'dt':0.05}
        
    return cart_position, cart_velocity, pole_angle, pole_ang_velocity, \
            atk_arch, def_arch, train_par, test_par, \
            robustness_formula