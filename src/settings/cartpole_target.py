import numpy as np

def get_settings(name, mode):

    safe_theta = 0.2
    safe_dist = 0.5
    robustness_theta = f'G(theta >= -{safe_theta} & theta <= {safe_theta})'
    robustness_dist = f'G(dist <= {safe_dist})'

    # PARAMS GRID
    
    if mode=="train":
        cart_position = np.linspace(0., 0., 1)
        cart_velocity = np.linspace(-1., 1., 100)
        pole_angle = np.linspace(-0., 0., 1)
        pole_ang_velocity = np.linspace(-1., 1., 100)
        x_target = np.linspace(-0.1, 0.1, 100)

    elif mode=="test":
        cart_position = np.linspace(0., 0., 1)
        cart_velocity = np.linspace(-.5, .5, 50)
        pole_angle = np.linspace(0., 0., 1)
        pole_ang_velocity = np.linspace(-.5, .5, 50)
        x_target = np.linspace(-0.1, 0.1, 50)

    # ARCHITECTURE

    if name=="default":

        atk_arch = {'hidden':1, 'size':10, 'coef':1, 'noise':3}
        def_arch = {'hidden':3, 'size':10, 'coef':1}
        train_par = {'train_steps':300, 'atk_steps':1, 'def_steps':3, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':200, 'dt':0.05}

    else:
        raise NotImplementedError

    return cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
            atk_arch, def_arch, train_par, test_par, robustness_theta, robustness_dist