import numpy as np

def get_settings(name, mode):

    safe_theta = 0.2
    safe_dist = 0.5
    # robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta})'
    robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta} & dist <= {safe_dist})'

    # PARAMS GRID
    
    if mode=="train":
        cart_position = np.linspace(0., 0., 1)
        cart_velocity = np.linspace(-1., 1., 100)
        pole_angle = np.linspace(-0., 0., 1)
        pole_ang_velocity = np.linspace(-1., 1., 100)
        x_target = np.linspace(-1., 1., 100)

    elif mode=="test":
        cart_position = np.linspace(0., 0., 1)
        cart_velocity = np.linspace(-.2, .2, 50)
        pole_angle = np.linspace(-0., 0., 1)
        pole_ang_velocity = np.linspace(-.2, .2, 50)
        x_target = np.linspace(-.5, .5, 50)

    # ARCHITECTURE

    if name=="default":

        atk_arch = {'hidden':2, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':5}
        train_par = {'train_steps':30000, 'atk_steps':5, 'def_steps':8, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':300, 'dt':0.05}
        
    return cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
            atk_arch, def_arch, train_par, test_par, \
            robustness_formula