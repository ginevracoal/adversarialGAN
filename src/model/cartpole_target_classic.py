import scipy.linalg
import numpy as np
from model.cartpole_target import *


class CartPole_classic(CartPole):
    def __init__(self, Q = np.diag([1,1,1,1]), R = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        # cartpole params
        self.d1 = 0.0
        self.d2 = 0.0
        
        # force max
        self.force_max = 3*(self.mpole + self.mcart)
        #self.input_dynamics = False
        #self.force_dynamics = DiscreteLowPassFilter()
        

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
        #self.reset_store(np.array([0,0,0,0]))
        
        # used in noise adder
        self.old_state = None
        
        #debugging tools
        self.state_archive = None
    
    ############################################################
    def is_unstable(self):
        return self.unstable_system
    
    ############################################################
    def update_A_B(self):
        _q = (self.mpole+self.mcart) * self.gravity / (self.mcart*self.lpole)
        self.A = np.array([\
                    [0,1,0,0], \
                    [0,-self.d1, -self.gravity*self.mpole/self.mcart,0],\
                    [0,0,0,1.],\
                    [0,self.d1/self.lpole,_q,-self.d2] ] )

        self.B = np.expand_dims( np.array( [0, 1.0/self.mcart, 0., -1/(self.mcart*self.lpole)] ) , 1 ) # 4x1
        
    ############################################################   
    def get_ctrl_signal(self, split_components = False):
       
        ctrl = self.u_LQR_ctrl
       
        if self.SMC_control:
            ctrl += self.u_SMC
            
        if self.correct_x:
            ctrl += self.u_Kp_correction
        
        #if self.input_dynamics:
        #    ctrl_out = self.force_dynamics.applyFilter(ctrl)
        #else:
        ctrl_out = ctrl
        
        if split_components:
            return np.array([self.u_LQR_ctrl, self.u_Kp_correction, self.u_SMC])[np.newaxis,:]
        else:
            return np.array([np.clip(ctrl_out,-self.force_max ,self.force_max)])

    ############################################################
    def computeControlSignals(self, state, x_target=0):
        self.SMC_law(state)
        self.LQR_law(state, x_target)
        self.Kp_x_law(state, x0 = x_target)
        
    
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
    def LQR_law(self, state_in, x0):

        state = state_in.copy()
        if self.K is None:
            # K : State feedback for stavility

            K, X, eigVals = lqr (self.A, self.B, self.Q, self.R )
            self.K = K
        
        theta_max = 0.1
        state[2] = np.clip(state[2],-theta_max, theta_max)
        
        LQR_ctrl = -np.matmul( self.K , state - np.array([x0,0,0,0]) )[0,0]   
        self.u_LQR_ctrl =  self.saturate(LQR_ctrl, self.ctrl_LQR_max)
        
        if  abs(state[2])>np.pi/2  :
            self.unstable_system = True
            
        return self.u_LQR_ctrl


class Model_classic(Model):

    def __init__(self, param_generator):
        super().__init__(param_generator)
        # setting of the initial conditions
        self.cartpole = CartPole_classic()


def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    if isinstance(R, float) or isinstance(R, int):
        R = np.array([[R]])
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K, X, eigVals
