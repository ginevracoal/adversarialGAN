# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 17:57:18 2020

@author: Enrico Regolin
"""

import numpy as np
from scipy.integrate import odeint
import control
from tqdm import tqdm


import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib as mpl
import matplotlib.pyplot as plt
default_backend = plt.get_backend()



#%%

class InvertedPendulum_LQR():
    # state = x, dot_x, theta, dot_theta
    
    ############################################################
    def __init__(self, Q = np.diag([1,1,1,1]), R = 1):
        
        # cartpole params
        self.g = 9.81
        self.L = 1.5
        self.m = 1.0
        self.M = 5.0
        self.d1 = 1.0
        self.d2 = 0.5
        
        #Ginevra's cartpole params
        self.L = 1.0
        self.m = .1
        self.M = 1.0
        
        # force max
        self.force_max = 3*(self.m + self.M)
        self.input_dynamics = False
        self.force_dynamics = DiscreteLowPassFilter()
        
        
        # friction options
        self.change_friction = True
        self.friction_min = 0.1
        self.friction_max = (self.m + self.M)*1

        # map generation
        self.theta_max = 0.9*(np.pi/2)
        self.dot_theta_max = 2

        # anim generation
        self.x_max = 5

        # LQR controller params
        self.ctrl_LQR_max = self.force_max*0.55
        self.u_LQR_ctrl = 0
        self.K = None
        self.Q = Q
        self.R = R

        # proportional x controller parameters
        self.correct_x = True
        self.ctrl_Kp_max = self.force_max*0.65
        self.u_Kp_correction = 0
        self.Kp_multiplier =  self.force_max/10 #2

        # robust SMC parameters
        self.SMC_control = True
        self.u_SMC = 0
        self.alpha_sliding = 3.5
        self.K_smc = self.force_max*2

        # instability flag (to stop simulation)
        self.unstable_system = False

        # initialize linearized system A,B
        self.update_A_B()
        self.reset_store(np.array([0,0,0,0]))
        
        

        
    ############################################################
    def reset_store(self, state):
        
        self.solution = np.array(state)[np.newaxis,:]
        self.target_pos = np.zeros((1,))
        self.ctrl_inputs = np.zeros((1,3))

        
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
        
        if self.input_dynamics:
            ctrl_out = self.force_dynamics.applyFilter(ctrl)
        else:
            ctrl_out = ctrl
        
        if split_components:
            return np.array([self.u_LQR_ctrl, self.u_Kp_correction, self.u_SMC])[np.newaxis,:]
        else:
            return np.array([np.clip(ctrl_out,-self.force_max ,self.force_max)])

    ############################################################
    def computeControlSignals(self, state, Q, R, x_target):
        self.SMC_law(state)
        self.LQR_law(state, Q, R, x0 = x_target)
        self.Kp_x_law(state, x0 = x_target)

    ############################################################
    def get_friction(self, state):
        intervals = [[-3.5,-2],[1,2.5]]
        if self.change_friction:
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
        else:
            if len(state) == 4:
                return self.friction_min
            else:
                return self.friction_min*np.ones(len(state))
        
    ############################################################
    def derivativesGin(self, state,  ctrl = True):
        
        friction = self.get_friction(state)
        ctrl_input = self.get_ctrl_signal()

        num = (-ctrl_input-self.m*self.L*dot_theta**2*np.sin(theta)+self.mu*np.sign(dot_x))/(self.m+self.M)

        den = 4/3-(self.m*np.cos(theta)**2)/(self.m+self.M)

        ddot_theta = (self.g*np.sin(theta)+np.cos(theta)*num)/(self.L*den)

        ddot_x = (ctrl_input+self.m*self.l*(dot_theta**2*np.sin(theta)-ddot_theta*np.cos(theta))\
                 -self.mu*np.sign(dot_x))/(self.m+self.M)
            
        return [ state[1], ddot_x , state[3], ddot_theta ]
        
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
    # used to plot phase plan
    def f_g_SMC(self, theta, dot_theta):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        f_x = self.g/self.L*sin_theta-(cos_theta*(self.m*self.L+dot_theta**2*sin_theta-self.m*self.g*cos_theta*sin_theta))/   \
            (self.L+(self.M+self.m*(1-cos_theta**2)))
            
        g_x =  -cos_theta / (self.L * (self.M+self.m*(1-cos_theta**2)))
        
        return f_x, g_x
    
    ############################################################ 
    def robustness_map(self, size = [150,100]):
    # used to plot phase plan  
        robustness_map = np.zeros(size)
        f_x_map = np.zeros(size)
        g_x_map = np.zeros(size)
        
        theta_axis = np.arange(-self.theta_max, self.theta_max, 2*self.theta_max/(size[0]))
        dot_theta_axis = np.arange(-self.dot_theta_max, self.dot_theta_max, 2*self.dot_theta_max/(size[1]))
        for i,theta in enumerate(theta_axis ):
            for j,dot_theta in enumerate(dot_theta_axis ):
                
                f_x, g_x = self.f_g_SMC(theta, dot_theta)
                robustness_map[i,j] = abs(f_x  + self.alpha_sliding*dot_theta )/abs(g_x)
                f_x_map[i,j] = f_x  #+ alpha*dot_theta
                g_x_map[i,j] = g_x
                
        return theta_axis,dot_theta_axis, np.flipud(robustness_map.T), f_x_map, g_x_map
    #np.flipud(robustness_map), np.flipud(f_x_map), np.flipud(g_x_map) 
    #robustness_map, f_x_map, g_x_map
    

    
    ############################################################ 
    def compute_K(self, desired_eigs = [-0.1, -0.2, -0.3, -0.4] ):
        #print(f'[compute_K] desired_eigs= {desired_eigs}')
        self.K = control.place( self.A, self.B,  desired_eigs )
  
    ############################################################
    def is_unstable(self):
        return self.unstable_system

    ############################################################
    # used to saturate any signal
    def saturate(self, signal, max_value):
        return np.clip(signal, -max_value, max_value)
        
    ############################################################
    def SMC_law(self, state):
        
        if self.SMC_control:
           #z = odeint(ode_fun, self.state_z, delta_t)
            zero_action_bound = 0.1
            saturation_init = 1.5
            
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
            x_thd = .2
            Kp_min = 5
            
            if abs(err_x) <x_thd:
                Kp = Kp_min + (Kp-Kp_min)*abs(err_x)/x_thd
            Kp *= self.Kp_multiplier
                        
            self.u_Kp_correction =  self.saturate(-Kp * err_x , self.ctrl_Kp_max)
            
        return self.u_Kp_correction

    ############################################################ 
    # LQR control law
    def LQR_law(self, state, Q, R , x0 = 0):

        if self.K is None:
            # K : State feedback for stavility
            # S : Solution to Riccati Equation
            # E : Eigen values of the closed loop system
            K, S, E = control.lqr( self.A, self.B, Q, R )
            self.compute_K(desired_eigs = E ) # Arbitarily set desired eigen values
        
        theta_max = 0.1
        state[2] = np.clip(state[2],-theta_max, theta_max)
        
        LQR_ctrl = -np.matmul( self.K , state - np.array([x0,0,0,0]) )[0,0]   
        self.u_LQR_ctrl =  self.saturate(LQR_ctrl, self.ctrl_LQR_max)
        
        if  abs(state[2])>np.pi/2  :
            self.unstable_system = True
            
        return self.u_LQR_ctrl

    ############################################################ 
    # run simulation
    def run_simulation(self,state, timeline, model_Gin = False):
    
        self.reset_store(state)
        
        for i,t in enumerate(tqdm(time_line[:-1])):
            
            x_target = target_generator(t)
            
            if model_Gin:
                def ode_fun( state, dt, x_target):
                    self.computeControlSignals(state, self.Q, self.R, x_target = x_target)
                    f_state = np.array(self.derivatives(state), dtype = np.float32)
                    return f_state
            else:
                def ode_fun( state, dt, x_target):
                    self.computeControlSignals(state, self.Q, self.R, x_target = x_target)
                    f_state = np.array(self.derivatives(state), dtype = np.float32)
                    return f_state
           
            delta_t = [t, time_line[i+1]]
            
            #integrate
            new_state = odeint(ode_fun, state, delta_t, args = (x_target,) )
           
            # store data for graphs    
            self.solution = np.append(self.solution, new_state[-1:,:], axis=0)
            self.target_pos = np.append(self.target_pos,np.array([x_target]) , axis=0)
            self.ctrl_inputs = np.append(self.ctrl_inputs, self.get_ctrl_signal(True) ,axis = 0)
            
            #update state
            state = np.array(new_state)[-1,:]
                
            if self.is_unstable():
                print('unstable system')
                break

    ############################################################ 
    #plot graph
    def plot_graphs(self, save = False, no_norm = False):
        
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        fig4 = plt.figure()
        
        ax1 = fig1.add_subplot(311)
        ax1_a = fig1.add_subplot(312)
        ax2 = fig1.add_subplot(313)

        
        ax3 = fig2.add_subplot(211)
        ax4 = fig2.add_subplot(212)
        
        ax5 = fig3.add_subplot(111)
        
        ax6 = fig4.add_subplot(211)
        ax7 = fig4.add_subplot(212)
        
        sim_end = len(self.target_pos)
        
        # fig 1
        ax1.plot(time_line[:sim_end], self.target_pos)
        ax1.plot(time_line[:sim_end], self.solution[:,0])
        ax1.legend(('target','actual'))
        
        thd_err_1 = .9
        thd_err_2 = 1.4
        ax1_a.plot(time_line[:sim_end], thd_err_1*np.ones(time_line[:sim_end].shape) ,'r--')
        ax1_a.plot(time_line[:sim_end], -thd_err_1*np.ones(time_line[:sim_end].shape),'r--' )
        ax1_a.plot(time_line[:sim_end], thd_err_2*np.ones(time_line[:sim_end].shape) ,'r')
        ax1_a.plot(time_line[:sim_end], -thd_err_2*np.ones(time_line[:sim_end].shape),'r' )
        ax1_a.plot(time_line[:sim_end], (self.solution[:,0]- self.target_pos) )
        ax1_a.legend(['tracking error'])
        
        ax2.plot(time_line[:sim_end], self.solution[:,2])
        ax2.legend(['angle'])
        
        # fig 2
        ax3.plot(time_line[:sim_end], self.ctrl_inputs[:,0])
        ax3.plot(time_line[:sim_end], self.ctrl_inputs[:,1])
        ax3.plot(time_line[:sim_end], self.ctrl_inputs[:,2])
        ax3.legend(('LQR', 'Kp correction', 'SMC'))
        
        ax4.plot(time_line[:sim_end], np.sum(self.ctrl_inputs,axis = 1))
        ax4.legend(['tot ctrl'])
        
        # fig 3
        theta_axis,dot_theta_axis,robustness_map, f_x_map, g_x_map = self.robustness_map()
        if no_norm:
            mappable = ax5.imshow(robustness_map, aspect = 'auto',extent=extents(theta_axis) + extents(dot_theta_axis), cmap=mpl.cm.jet)
        else:
            mappable = ax5.imshow(robustness_map, aspect = 'auto',extent=extents(theta_axis) + extents(dot_theta_axis), cmap=mpl.cm.jet, norm=mpl.colors.PowerNorm(.8,1,2*self.force_max))
        plt.colorbar(mappable=mappable, ax = ax5)
        fontsize = 15
        plt.xlabel(r'$\theta$', fontsize=fontsize)
        plt.ylabel(r'$\dot\theta$', fontsize=fontsize, rotation=0)
        
        ax5.scatter(self.solution[:,2],self.solution[:,3],s=7,color='g')
        idx_SMC_active = np.where(np.abs(self.ctrl_inputs[:,2])>10)
        ax5.scatter(self.solution[idx_SMC_active,2][0,:],self.solution[idx_SMC_active,3][0,:],s=4,color='r')
        
        idx_plot = np.where(np.abs(self.alpha_sliding*theta_axis)<dot_theta_axis[-1])
        ax5.plot(theta_axis[idx_plot],-self.alpha_sliding*theta_axis[idx_plot],'k')
        
        #fig 4
        ax6.plot(time_line[:sim_end], self.solution[:,1])
        ax6.legend(['x dot'])
        
        ax7.plot(time_line[:sim_end], self.solution[:,3])
        ax7.legend(['theta dot'])
        
        
        plt.show()
        
        if save:
            fig1.savefig('states.eps')
            fig2.savefig('ctrl_signals.eps')
            fig3.savefig('phase_plan.eps')
            fig4.savefig('state_derivatives.eps')
        ##


    ############################################################ 
    def generate_gif(self, dt, sample_step=1):
    
        ths = self.solution[:, 2]
        xs = self.solution[:, 0]
        
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
        normalize = mpl.colors.Normalize(vmin=self.friction_min, vmax=self.friction_max)
        
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
            target.set_offsets((self.target_pos[i],-0.2))
            
            return line, time_text, patch, target
        
        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(self.solution), sample_step),
                                      interval=round(sample_step/dt), blit=True, init_func=init)

        # Set up formatting for the movie files
        print("Writing video...")
        Writer = animation.writers['imagemagick']
        writer = Writer(fps=round(1000*dt/sample_step), metadata=dict(artist='Sergey Royz'), bitrate=1800)
        if self.change_friction:
            ani.save('robust_friction.gif', writer=writer)
        else:
            ani.save('robust_no_friction.gif', writer=writer)
        print("Video ready!")

#%%

############################################################ 
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

def target_generator(time):
    
    speed_factor = 1
    
    omega_1 = .3*speed_factor
    omega_2 = .1*speed_factor
    omega_3 = .2*speed_factor
    omega_4 = .4*speed_factor

    def fun_val(t):
        return 2*np.sin(omega_1*t) + 0.5*np.sin(omega_2*t + np.pi/7) + 0.8*np.sin(omega_3*t - np.pi/12)+ 1*np.cos(omega_4*t+np.pi/5)
    
    alpha = 0.02
    x_target = fun_val(time) - fun_val(0)*np.exp(-alpha*time)

    return x_target


#%%
# initialize system
    
# Eigen Values set by LQR
Q = np.diag( [1,.01,1,1] )
R = np.diag( [1] )
sys = InvertedPendulum_LQR(Q,R) # Q,R not used, default values instead
sys.change_friction = False
#sys.correct_x = False
#sys.Kp_multiplier = 1.5 # to be used for LQR+Kp without SMC and change of friction
sys.SMC_control = True
sys.alpha_sliding = 3.5
#sys.K_smc = 75

# initial conditions
# state = [x, dot_x, theta, dot_theta]
state = np.array([ 0, 0, np.pi/10, 0], dtype = np.float32)

# simulation time
dt = 0.05
Tmax = 100
time_line = np.arange(0.0, Tmax, dt)

#%%
"""
fig = plt.figure()
ax = fig.add_subplot(111)
theta_axis,dot_theta_axis,robustness_map, f_x_map, g_x_map = sys.robustness_map()
mappable = ax.imshow(robustness_map, aspect = 'auto',extent=extents(theta_axis) + extents(dot_theta_axis), cmap=mpl.cm.jet, norm=mpl.colors.PowerNorm(.8,1,200))
plt.colorbar(mappable=mappable, ax = ax)
fontsize = 15
idx_plot = np.where(np.abs(sys.alpha_sliding*theta_axis)<dot_theta_axis[-1])
ax.plot(theta_axis[idx_plot],-sys.alpha_sliding*theta_axis[idx_plot],'r', linewidth=3)
plt.title('Minimum required SMC gain')
plt.xlabel(r'$\theta$', fontsize=fontsize)
plt.ylabel(r'$\dot\theta$', fontsize=fontsize, rotation=0)
"""
#%%

sys.run_simulation(state, time_line, model_Gin = True)
sys.plot_graphs(save = True, no_norm = False)

# generate animation
if False:
    mpl.use('TKAgg')
    sys.generate_gif(dt, sample_step=5)
    mpl.use(default_backend)
