import numpy as np

def get_settings(name, mode):

    safe_theta = 0.2
    safe_dist = 0.5
    robustness_formula = f'G(theta >= -{safe_theta} & theta <= {safe_theta})'

    # PARAMS GRID
    
    if mode=="train":
        cart_position = np.linspace(0., 0., 1)
        cart_velocity = np.linspace(-1., 1., 100)
        pole_angle = np.linspace(0., 0., 1)
        pole_ang_velocity = np.linspace(-1., 1., 100)

    elif mode=="test":
        cart_position = np.linspace(-0.1, 0.1, 10)
        cart_velocity = np.linspace(-1., 1., 50)
        pole_angle = np.linspace(-0., 0., 10)
        pole_ang_velocity = np.linspace(-1., 1., 50)

    # ARCHITECTURE

    if name=="default":

        net_arch = {'hidden':3, 'size':10}
        train_par = {'train_steps':300, 'horizon':2., 'dt': 0.05, 'lr':.001}
        test_par = {'test_steps':100, 'dt':0.05}    

    else:
        raise NotImplementedError

    return cart_position, cart_velocity, pole_angle, pole_ang_velocity, \
            net_arch, train_par, test_par, robustness_formula

def get_relpath(main_dir, train_params):
    return main_dir+"_lr="+str(train_params["lr"])+"_dt="+str(train_params["dt"])+\
          "_horizon="+str(train_params["horizon"])+"_train_steps="+\
          str(train_params["train_steps"])+"/"

def get_net_filename(hidden_layers, layer_size):
    return 'policynet_hidden='+str(hidden_layers)+'_size='+str(layer_size)+'.pt'

def get_sims_filename(repetitions, dt, test_steps):
    return 'sims_reps='+str(repetitions)+'_dt='+str(dt)+'_test_steps='+str(test_steps)+'.pkl'