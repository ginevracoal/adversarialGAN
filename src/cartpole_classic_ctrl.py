# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:57:18 2020

@author: enric
"""


import numpy as np

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as pp
from scipy.integrate import odeint

import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from math import pi, trunc
from numpy import sin, cos

import control

import random

from tqdm import tqdm

#%%

def trim(x, step):
    d = trunc(x / step)
    return step * d

precision = 0.006
#k = 1000.0    # Kalman filter coefficient


#%%
import sympy #import Symbol, diff, subs
# symbolic math linearization

x = sympy.Symbol('x')
x_ddot = sympy.Symbol('x_ddot')
m = sympy.Symbol('m')
L = sympy.Symbol('L')
theta = sympy.Symbol('theta')
dot_theta = sympy.Symbol('dot_theta')
g = sympy.Symbol('g')
M = sympy.Symbol('M')
u = sympy.Symbol('u')
dot_x = sympy.Symbol('dot_x')
d1 = sympy.Symbol('d1')
d2 = sympy.Symbol('d2')


damping_x =  - d1*dot_x
damping_theta =  - d2*dot_theta

x_ddot = u + m*L*dot_theta**2* sympy.sin(theta) - m*g*sympy.cos(theta) *  sympy.sin(theta)
x_ddot = x_ddot / ( M+m-m* sympy.cos(theta)**2 ) + damping_x

theta_ddot = (g*sympy.sin( theta ) -  sympy.cos( theta )*x_ddot ) / L + damping_theta


"""
x_ddot = u - m*L*(dot_theta**2)* sympy.cos( theta ) + m*g*sympy.cos(theta) *  sympy.sin(theta)
x_ddot = x_ddot / ( M+m-m* sympy.sin(theta)**2 ) + damping_x

theta_ddot = -g/L * sympy.cos( theta ) -  sympy.sin( theta ) / L * x_ddot + damping_theta 
"""

states_f = [dot_x,x_ddot, dot_theta, theta_ddot ]
states = [x,dot_x, theta, dot_theta ]

#print(x_ddot)

linearized_symb_mat_A = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
linearized_symb_mat_B = [0,0,0,0]

for i,f_state in enumerate(states_f):
    for j,state in enumerate(states):
        linearized_symb_mat_A[i][j]= f_state.diff(state)
        print(f'{i},{j}: {linearized_symb_mat_A[i][j]}')

for i,f_state in enumerate(states_f):
    linearized_symb_mat_B[i]= f_state.diff(u)
    print(f'{i}: {linearized_symb_mat_B[i]}')

#%%

class InvertedPendulumSystem():
    # state = x, dot_x, theta, dot_theta
    
    ############################################################
    def __init__(self,  linearized_symb_mat_A = None, linearized_symb_mat_B= None, friction_map = None,):
        
        self.linearized_symb_mat_A = linearized_symb_mat_A
        self.linearized_symb_mat_B = linearized_symb_mat_B
        
        
        self.x_max = 10
        self.friction_map = friction_map
       
        self.g = 9.81
        self.L = 1.5
        self.m = 1.0
        self.M = 5.0
        self.d1 = 1.0
        self.d2 = 0.5

        self.F_max = 50

        self.unstable_system = False
        
        self.dynamic_LQR = False
        self.K = None

        # gains which worki in the simple case        
        Kp_th = 80
        Kd_th = 15
        Kp_x = .5*3.1
        Kd_x = 0.35*4.8
        self.gains = [Kp_th, Kd_th, Kp_x, Kd_x]
        
        self.ctrl_input = 0
        
        self.update_A_B()
        #self.update_A_B_new(state = [0,0,0,0], ctrl_u = 0)

    ############################################################        
    def update_A_B_new(self, state, ctrl_u):
        
        self.A = np.zeros((4,4))
        self.B = np.zeros((4,1))
        
        for i,row in enumerate(self.linearized_symb_mat_A):
            for j,col in enumerate(row):
                self.A[i][j] = float(col.subs([(u, ctrl_u),(x ,state[0]), (dot_x,state[1]),\
                                         (theta, state[2]), (dot_theta, state[3]), (m , self.m),\
                                             (M , self.M), (g,self.g), (L,self.L),\
                                                 (d1,self.d1), (d2,self.d2)]) )
            
        for i,row in enumerate(self.linearized_symb_mat_B):
            self.B[i][0] = float(row.subs([(u, ctrl_u),(x ,state[0]), (dot_x,state[1]),\
                                     (theta, state[2]), (dot_theta, state[3]), (m , self.m),\
                                         (M , self.M), (g,self.g), (L,self.L),\
                                             (d1,self.d1), (d2,self.d2)]) )
        

        # Pendulum up (linearized eq)
        # Eigen val of A : array([[ 1.        , -0.70710678, -0.07641631,  0.09212131] )
        
    ############################################################
    def update_A_B(self):
        _q = (self.m+self.M) * self.g / (self.M*self.L)
        self.A = np.array([\
                    [0,1,0,0], \
                    [0,-self.d1, -self.g*self.m/self.M,0],\
                    [0,0,0,1.],\
                    [0,self.d1/self.L,_q,-self.d2] ] )

        self.B = np.expand_dims( np.array( [0, 1.0/self.M, 0., -1/(self.M*self.L)] ) , 1 ) # 4x1
        
    ############################################################   
    def get_ctrl_signal(self, ctrl = True):
        """
        if ctrl:
            ctrl_input = np.clip(self.ctrl_input, -self.F_max, self.F_max)
            #print(ctrl_input)
        else:
            ctrl_input = 0
        """    
        return self.ctrl_input

    ############################################################
    def get_friction(self, state):
        return self.friction_map(state[0])
        
    ############################################################
    def derivatives(self, state,  ctrl = True):
        
        if self.friction_map is not None:
            friction = self.get_friction(state)
        else:
            friction = 0.01
        
        ctrl_input = self.get_ctrl_signal(ctrl)
        
        x_ddot = ctrl_input - friction*state[1]  + self.m*self.L*state[3]**2* np.sin(state[2]) - self.m*self.g*np.cos(state[2]) *  np.sin(state[2])
        x_ddot = x_ddot / ( self.M+self.m-self.m* np.cos(state[2])**2 )
    
        theta_ddot = (self.g*np.sin( state[2] ) -  np.cos( state[2] )*x_ddot ) / self.L 
    
        damping_x =  - self.d1*state[1]
        damping_theta =  - self.d2*state[3]
    
        return [ state[1], x_ddot + damping_x, state[3], theta_ddot + damping_theta ]

        
    ############################################################ 
    def derivatives_simple(self, state, ctrl = True):
        ds = np.zeros_like(state)
        # x0 = step(t)

        ctrl_input = self.get_ctrl_signal(ctrl)
        
        ds[0] = state[1]
        ds[1] = ctrl_input/self.M
        ds[2] = state[3]
        ds[3] = (self.g * np.sin(state[2]) - ctrl_input/self.M * np.cos(state[2])) / self.L 
        
        return ds
    
    ############################################################ 
    def compute_K(self, desired_eigs = [-0.1, -0.2, -0.3, -0.4] ):
        #print(f'[compute_K] desired_eigs= {desired_eigs}')
        self.K = control.place( self.A, self.B,  desired_eigs )

    ############################################################ 
    def get_K(self):
        return self.K
    
    
    ############################################################
    def is_unstable(self):
        return self.unstable_system

    ############################################################ 
    # LQR control law
    def LQR_law(self, state, Q, R , x0 = 0 , dt = 0.05):

        Kp = 30
        err_x = x0 - state[0]  
        x_thd = 1
        Kp_min = 5
        if abs(err_x) <x_thd:
            Kp = Kp_min + (Kp-Kp_min)*abs(err_x)/x_thd

        if self.K is None or self.dynamic_LQR:
                    # K : State feedback for stavility
            # S : Solution to Riccati Equation
            # E : Eigen values of the closed loop system
            K, S, E = control.lqr( self.A, self.B, Q, R )
            self.compute_K(desired_eigs = E ) # Arbitarily set desired eigen values
        
        u = -  np.matmul( self.K , state - np.array([x0,0,0,0]) )
        
        self.ctrl_input = u[0,0] - Kp * err_x 

        
        if (abs(self.ctrl_input)>1.5*self.F_max) or (abs(state[2])>np.pi/2 ) :
            self.unstable_system = True
        
        return u[0]

    ############################################################ 
    def PD_control_law(self, state, x0 = 0):
        Kp_th = self.gains[0]
        Kd_th = self.gains[1]
        Kp_x = self.gains[2]
        Kd_x = self.gains[3]
        _x = state[0]
        _Z = state[1]
        _th = state[2]
        _Y = state[3]

        u = Kp_th * _th + Kd_th * _Y + Kp_x * (_x - x0) + Kd_x * _Z
        
        self.ctrl_input = u
        
        round_to_tenths = [round(num, 1) for num in [Kp_th * _th , Kd_th * _Y , Kp_x * (_x - x0), Kd_x * _Z]]
        
        return u

    ############################################################ 
    def generate_gif(self, solution,target_pos, dt, sample_step=1):
    
        x_max = 5    
    
        ths = solution[:, 2]
        xs = solution[:, 0]
        
        pxs = self.L * np.sin(ths) + xs
        pys = self.L * np.cos(ths)
        
        fig = pp.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-x_max, x_max), ylim=(-0.5, 2))
        ax.set_aspect('equal')
        ax.grid()
        
        patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))
        
        target= ax.scatter([],[], s = 20, color =  'r')
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
        cart_width = 0.3
        cart_height = 0.2
        
         
        def init():
            line.set_data([], [])
            time_text.set_text('')
            patch.set_xy((-cart_width/2, -cart_height/2))
            patch.set_width(cart_width)
            patch.set_height(cart_height)
            target.set_offsets([])
            return line, time_text, patch, target
        
        
        def animate(i):
            thisx = [xs[i], pxs[i]]
            thisy = [0, pys[i]]
        
            line.set_data(thisx, thisy)
            time_text.set_text(time_template % (i*dt))
            patch.set_x(xs[i] - cart_width/2)
            target.set_offsets((target_pos[i],-0.2))
            return line, time_text, patch, target
        
        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(solution), sample_step),
                                      interval=round(sample_step/dt), blit=True, init_func=init)
        
        pp.show()
        
        # Set up formatting for the movie files
        print("Writing video...")
        Writer = animation.writers['imagemagick']
        writer = Writer(fps=round(1000*dt/sample_step), metadata=dict(artist='Sergey Royz'), bitrate=1800)
        ani.save('controlled-cart.gif', writer=writer)


#%%
    
    
# External Variables
ss = InvertedPendulumSystem(linearized_symb_mat_A, linearized_symb_mat_B)
# Eigen Values set by LQR
Q = np.diag( [1,1,5,1] )
R = np.diag( [1] )
    
#%%

# initial conditions
Y = .0         # pendulum angular velocity
th = pi/10    # pendulum angle
x = .0        # cart position
x0 = 0        # desired cart position
Z = .0        # cart velocity

#state = np.array([th, Y, x, Z, trim(th, precision), .0])
state = np.array([ x, Z,th, Y], dtype = np.float32) 


# simulation time
dt = 0.05
Tmax = 100
time_line = np.arange(0.0, Tmax, dt)

solution = np.array(state)[np.newaxis,:]
target_pos = np.zeros((1,))

#ss.update_A_B_new(state, 0)
ss.update_A_B()

print("Integrating...")


import cProfile
import pstats
import io

pr = cProfile.Profile()
pr.enable()


for i,t in enumerate(tqdm(time_line[:-1])):
    
    omega_1 = .2
    omega_2 = .5
    omega_3 = .4

    x_target = 2*np.sin(omega_1*t) + 0.5*np.sin(omega_2*t + np.pi/7) + 0.8*np.sin(omega_3*t - np.pi/12)
    
    def ode_fun( state, dt, x_target):
        

        ss.LQR_law(state, Q, R, x0 = x_target, dt = dt)
        #ss.PD_control_law(state)
        f_state = np.array(ss.derivatives(state))
        #f_state = np.array(ss.derivatives_simple(state, ctrl = True))
        return f_state
    
    delta_t = [t, time_line[i+1]]
    new_state = odeint(ode_fun, state, delta_t, args = (x_target,) )
    
    #ss.update_A_B_new(state, ss.ctrl_input)
    
    solution = np.append(solution, new_state[-1:,:], axis=0)
    target_pos = np.append(target_pos,np.array([x_target]) , axis=0)
    
    state = np.array(new_state)[-1,:]
    
    if ss.is_unstable():
        print('unstable system')
        break

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())


#%%


if (abs(solution[-50:,2]) < 0.25).all() or True:
    ss.generate_gif(solution,target_pos,dt, sample_step=10)

#u = ss.PD_control_law( state, gains, x0 )




# integrate your ODE using scipy.integrate.
#solution = integrate.odeint(derivatives, state, t, inputs)



#%%
# generate animation
