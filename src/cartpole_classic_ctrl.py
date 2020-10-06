# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:57:18 2020

@author: enric
"""


import numpy as np

import matplotlib


import matplotlib.pyplot as plt
default_backend = plt.get_backend()

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
    def __init__(self,  linearized_symb_mat_A = None, linearized_symb_mat_B= None):
        
        self.linearized_symb_mat_A = linearized_symb_mat_A
        self.linearized_symb_mat_B = linearized_symb_mat_B
        
        
        self.x_max = 5
        
        self.friction_min = 1
        self.friction_max = 18
        
        self.g = 9.81
        self.L = 1.5
        self.m = 1.0
        self.M = 5.0
        self.d1 = 1.0
        self.d2 = 0.5

        self.F_max = 50

        self.SMC_control = True
        self.correct_x = True
        
        self.ctrl_Kp_max = 40
        self.ctrl_LQR_max = 40

        self.u_LQR_ctrl = 0
        self.u_Kp_correction = 0
        self.u_SMC = 0
        self.Kp_multiplier = 5
        
        self.alpha_sliding = 2
        self.K_smc = 75

        self.unstable_system = False
        
        self.dynamic_LQR = False
        self.K = None

        # gains which worki in the simple case        
        Kp_th = 80
        Kd_th = 15
        Kp_x = .5*3.1
        Kd_x = 0.35*4.8
        self.gains = [Kp_th, Kd_th, Kp_x, Kd_x]
        
        
        
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
    def get_ctrl_signal(self, split_components = False):
        ctrl = self.u_LQR_ctrl
       
        if self.SMC_control:
            ctrl += self.u_SMC
            
        if self.correct_x:
            ctrl += self.u_Kp_correction
        
        if split_components:
            return np.array([self.u_LQR_ctrl, self.u_Kp_correction, self.u_SMC])[np.newaxis,:]
        else:
            return np.array([ctrl])

    ############################################################
    def computeControlSignals(self, state, Q, R, x_target):
        self.SMC_law(state)
        self.LQR_law(state, Q, R, x0 = x_target)
        self.Kp_x_law(state, x0 = x_target)

    ############################################################
    def get_friction(self, state):
        intervals = [[-5,-2],[1,2.5]]
        
        if len(state) == 4:
            if intervals[0][0] < state[0] < intervals[0][1] or intervals[1][0] < state[0] < intervals[1][1]:
                friction = self.friction_max
            else:
                friction = self.friction_min
            
            return friction
        else:
            friction_array = self.friction_min*np.ones(len(state))
            friction_array[ np.bitwise_or(np.bitwise_and(state>intervals[0][0] ,state<intervals[0][1]),\
                                          np.bitwise_and(state>intervals[1][0] , state<intervals[1][1]) )  ] \
                                        = self.friction_max
            
            return friction_array
        
    ############################################################
    def derivatives(self, state,  ctrl = True):

        friction = self.get_friction(state)
        
        ctrl_input = self.get_ctrl_signal()
        
        x_ddot = ctrl_input - friction*state[1]  + self.m*self.L*state[3]**2* np.sin(state[2]) - self.m*self.g*np.cos(state[2]) *  np.sin(state[2])
        x_ddot = x_ddot / ( self.M+self.m-self.m* np.cos(state[2])**2 )
    
        theta_ddot = (self.g*np.sin( state[2] ) -  np.cos( state[2] )*x_ddot ) / self.L 
    
        damping_x =  - self.d1*state[1]
        damping_theta =  - self.d2*state[3]
    
        return [ state[1], x_ddot + damping_x, state[3], theta_ddot + damping_theta ]


    ############################################################ 
    def f_g_SMC(self, theta, dot_theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        f_x = self.g/self.L*sin_theta-(cos_theta*(self.m*self.L+dot_theta**2*sin_theta-self.m*self.g*cos_theta*sin_theta))/   \
            (self.L+(self.M+self.m*(1-cos_theta**2)))
            
        g_x = - cos_theta / (self.L * (self.M+self.m*(1-cos_theta**2)))
        
        return f_x, g_x

    
    ############################################################ 
    def robustness_map(self, alpha = 2, theta_max = 0.9*(np.pi/2), dot_theta_max = 1, size = [150,100]):
        
        robustness_map = np.zeros(size)
        f_x_map = np.zeros(size)
        g_x_map = np.zeros(size)
        
        theta_axis = np.arange(-theta_max, theta_max, 2*theta_max/(size[0]))
        dot_theta_axis = np.arange(-dot_theta_max, dot_theta_max, 2*dot_theta_max/(size[1]))
        for i,theta in enumerate(theta_axis ):
            for j,dot_theta in enumerate(dot_theta_axis ):
                
                f_x, g_x = self.f_g_SMC(theta, dot_theta)
                robustness_map[i,j] = abs((f_x  + alpha*dot_theta)/g_x)
                f_x_map[i,j] = f_x  #+ alpha*dot_theta
                g_x_map[i,j] = g_x
                
                
        return theta_axis,dot_theta_axis,robustness_map, f_x_map, g_x_map
        
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
    def dot_z(self, state):
        """        
        if isinstance(state,list);:
            state = np.array(state)[:,np.newaxis]
        """
        return -np.dot(self.sliding_surf_coeffs ,(np.dot(self.A,state)+ np.dot(self.B,self.ctrl_input)))[0,0]


    ############################################################
    def get_s0(self, state):
        return  np.dot(self.sliding_surf_coeffs, state)  
        
        
    ############################################################
    def saturate(self, signal, max_value):
        return np.clip(signal, -max_value, max_value)
        
    ############################################################
    def SMC_law(self, state):
        
        if self.SMC_control:
           #z = odeint(ode_fun, self.state_z, delta_t)
            zero_action_bound = 0.05
            saturation_init = 1
            
            sigma = state[2]*self.alpha_sliding + state[3]
            
            control_action = 0
            if abs(sigma) > zero_action_bound:
                if abs(sigma)> saturation_init:
                    control_action = np.sign(sigma);
                else:
                    control_action = np.sign(sigma)*(abs(sigma)-zero_action_bound)/(saturation_init-zero_action_bound);
            
            self.u_SMC  = control_action *self.K_smc
        
        return self.u_SMC 

    ############################################################ 
    # proportional control for x correction
    def Kp_x_law(self, state, x0 = 0):  
    
        if self.correct_x:
            err_x = x0 - state[0]
                    #self.max_err = 2.5
            Kp = 10
            #err_x = np.clip(x0 - state[0], -self.max_err, self.max_err)  
            x_thd = 1
            Kp_min = 1
            if abs(err_x) <x_thd:
                Kp = Kp_min + (Kp-Kp_min)*abs(err_x)/x_thd
            Kp *= self.Kp_multiplier
            self.u_Kp_correction =  self.saturate(-Kp * err_x, self.ctrl_Kp_max)
            
        return self.u_Kp_correction



    ############################################################ 
    # LQR control law
    def LQR_law(self, state, Q, R , x0 = 0):

        if self.K is None or self.dynamic_LQR:
            # K : State feedback for stavility
            # S : Solution to Riccati Equation
            # E : Eigen values of the closed loop system
            K, S, E = control.lqr( self.A, self.B, Q, R )
            self.compute_K(desired_eigs = E ) # Arbitarily set desired eigen values
        
        LQR_ctrl = -np.matmul( self.K , state - np.array([x0,0,0,0]) )[0,0]   
        self.u_LQR_ctrl =  self.saturate(LQR_ctrl, self.ctrl_LQR_max)
        
        if (abs(self.u_LQR_ctrl)>2*self.F_max) or (abs(state[2])>np.pi/2 ) :
            self.unstable_system = True
            
        return self.u_LQR_ctrl
        

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
    
        ths = solution[:, 2]
        xs = solution[:, 0]
        
        pxs = self.L * np.sin(ths) + xs
        pys = self.L * np.cos(ths)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-self.x_max, self.x_max), ylim=(-0.5, 2))
        ax.set_aspect('equal')
        ax.grid()
        
        patch = ax.add_patch(Rectangle((0, 0), 0, 0, linewidth=1, edgecolor='k', facecolor='g'))
        
        target= ax.scatter([],[], s = 20, color =  'r')
        line, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        
               
        x_friction_plot = np.arange(-self.x_max, self.x_max, 0.1)
        normalize = matplotlib.colors.Normalize(vmin=self.friction_min, vmax=self.friction_max)
        
        xy_scat = np.concatenate((x_friction_plot[:,np.newaxis],-0.2*np.ones((len(x_friction_plot),1)) ), axis = 1) 
        frict_level = self.get_friction(x_friction_plot)
        
        frict_scat =  ax.scatter( xy_scat[:,0],xy_scat[:,1], c = frict_level ,s = 5, cmap = plt.cm.jet, norm = normalize)
                
        
        
        cart_width = 0.3
        cart_height = 0.2
        
         
        def init():
            line.set_data([], [])
            time_text.set_text('')
            patch.set_xy((-cart_width/2, -cart_height/2))
            patch.set_width(cart_width)
            patch.set_height(cart_height)
            target.set_offsets([])
            
            frict_scat.set_offsets( xy_scat  )
                        
            return line, time_text, patch, target, frict_scat
        
        
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
        
        #plt.show()
        
        # Set up formatting for the movie files
        print("Writing video...")
        Writer = animation.writers['imagemagick']
        writer = Writer(fps=round(1000*dt/sample_step), metadata=dict(artist='Sergey Royz'), bitrate=1800)
        ani.save('controlled-cart.gif', writer=writer)


class DiscreteLowPassFilter():
    def __init__(self, a = 0.9):
        self.a = a
        self.old_output = 0
        
    def applyFilter(self, in_signal):
        self.old_output = self.a*self.old_output + ( (1-self.a)*in_signal)        
        return self.old_output
    
    def resetFilter(self):
        self.old_output = 0


#%%
def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

ss_test = InvertedPendulumSystem()
theta_axis,dot_theta_axis,robustness_map, f_x_map, g_x_map = ss_test.robustness_map()

"""
fig = plt.figure()
plt.imshow(f_x_map, extent=extents(theta_axis) + extents(dot_theta_axis))
plt.colorbar()

fig = plt.figure()
plt.imshow(g_x_map, extent=extents(theta_axis) + extents(dot_theta_axis))
plt.colorbar()
"""
import matplotlib as mpl
fig = plt.figure()
plt.imshow(robustness_map, extent=extents(theta_axis) + extents(dot_theta_axis), cmap=mpl.cm.jet, norm=mpl.colors.PowerNorm(.8,1,200))
plt.colorbar()
plt.xlabel('theta')
plt.ylabel('dot theta')

#%%
    
    
# External Variables
ss = InvertedPendulumSystem(linearized_symb_mat_A, linearized_symb_mat_B)

# Eigen Values set by LQR
Q = np.diag( [1,1,20,1] )
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
ctrl_inputs = np.zeros((1,3))

#ss.update_A_B_new(state, 0)
ss.update_A_B()

print("Integrating...")

#%%
import cProfile
import pstats
import io

pr = cProfile.Profile()
pr.enable()

for i,t in enumerate(tqdm(time_line[:-1])):
    
    omega_1 = .2
    omega_2 = .1
    omega_3 = .3

    x_target = 2*np.sin(omega_1*t) + 0.5*np.sin(omega_2*t + np.pi/7) + 0.8*np.sin(omega_3*t - np.pi/12)
    
    def ode_fun( state, dt, x_target):
        ss.computeControlSignals(state, Q, R, x_target = x_target)
        f_state = np.array(ss.derivatives(state), dtype = np.float32)
        #f_state = np.array(ss.derivatives_simple(state, ctrl = True))
        return f_state
   
    delta_t = [t, time_line[i+1]]
    
    #update state and z
    new_state = odeint(ode_fun, state, delta_t, args = (x_target,) )
   
    #ss.update_A_B_new(state, ss.ctrl_input)
    solution = np.append(solution, new_state[-1:,:], axis=0)
    
    target_pos = np.append(target_pos,np.array([x_target]) , axis=0)
    ctrl_inputs = np.append(ctrl_inputs, ss.get_ctrl_signal(True) ,axis = 0)
    #z_vect = np.append(z_vect, z ,  axis = 0)
    
    state = np.array(new_state)[-1,:]
        
    if ss.is_unstable():
        print('unstable system')
        break

#plot graph
fig_graphs = plt.figure()
ax1 = fig_graphs.add_subplot(221)
ax2 = fig_graphs.add_subplot(222)
ax3 = fig_graphs.add_subplot(223)
ax4 = fig_graphs.add_subplot(224)

ax1.plot(time_line[:len(target_pos)], target_pos)
ax1.plot(time_line[:len(target_pos)], solution[:,0])
ax1.legend(('target','actual'))

ax2.plot(time_line[:len(target_pos)], solution[:,2])
ax2.legend(['angle'])

ax3.plot(time_line[:len(target_pos)], ctrl_inputs[:,0])
ax3.plot(time_line[:len(target_pos)], ctrl_inputs[:,1])
ax3.plot(time_line[:len(target_pos)], ctrl_inputs[:,2])
ax3.legend(('LQR', 'Kp correction', 'SMC'))


"""
ax4.plot(time_line[:len(target_pos)], z_vect)
ax4.legend('z vector')
"""
plt.show()
##

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())

#%%


if (abs(solution[-50:,2]) < 0.25).all() and False:
    matplotlib.use('TKAgg')
    ss.generate_gif(solution,target_pos,dt, sample_step=5)
    matplotlib.use(default_backend)
#u = ss.PD_control_law( state, gains, x0 )




# integrate your ODE using scipy.integrate.
#solution = integrate.odeint(derivatives, state, t, inputs)



#%%
# generate animation
