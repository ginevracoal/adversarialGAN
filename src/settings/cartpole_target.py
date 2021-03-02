import numpy as np

def get_settings(name, mode):

    safe_theta = 0.2
    safe_dist = 2.
    robustness_theta = f'G(theta >= -{safe_theta} & theta <= {safe_theta})'
    robustness_dist = f'G(dist <= {safe_dist})'
    alpha=0.4 # distace robustness weight
    norm_theta=1.5
    norm_dist=30.

    # PARAMS GRID
    
    if mode=="train":
        cart_position = np.random.uniform(-1., 1., 100)
        cart_velocity = np.random.uniform(-1., 1., 100)
        pole_angle = np.random.uniform(-.1, .1, 100)
        pole_ang_velocity = np.random.uniform(-1., 1., 100)
        x_target = np.random.uniform(-1., 1., 100)

    elif mode=="test":
        cart_position = np.random.uniform(-1., 1., 150)
        cart_velocity = np.random.uniform(-1., 1., 150)
        pole_angle = np.random.uniform(-.1, .1, 150)
        pole_ang_velocity = np.random.uniform(-1., 1., 150)
        x_target = np.random.uniform(-1., 1., 150)

    # ARCHITECTURE

    if name=="default":

        atk_arch = {'hidden':1, 'size':10, 'coef':1, 'noise':2}
        def_arch = {'hidden':2, 'size':10, 'coef':1}
        train_par = {'train_steps':500, 'atk_steps':1, 'def_steps':2, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':200, 'dt':0.05}
    
    else:
        raise NotImplementedError

    return cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target, \
            atk_arch, def_arch, train_par, test_par, robustness_theta, robustness_dist, \
            alpha, safe_theta, safe_dist, norm_theta, norm_dist