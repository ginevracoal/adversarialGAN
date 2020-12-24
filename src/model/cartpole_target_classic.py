import scipy.linalg
import numpy as np
from model.cartpole_target import *


class Environment_signal():

    def __init__(self, test_steps):
        np.random.seed(0)

        self.test_steps = test_steps
        self.x_target = self.eps = self.x = self.duration = 0

        self.reference_speed_factor = 2. 

        phi= np.random.normal(0.5, 0.5, 4)
        phi_1 = np.pi/4*(phi[0]-0.5)
        phi_2 = np.pi/4*(phi[1]-0.5)
        phi_3 = np.pi/4*(phi[2]-0.5)
        phi_4 = np.pi/4*(phi[3]-0.5)
        self.phi = np.array([phi_1, phi_2, phi_3, phi_4])

        self.mu = np.random.normal(0.5, 0.5, test_steps)

    def fun_val(self, t):
        omega_1 = .3*self.reference_speed_factor
        omega_2 = .1*self.reference_speed_factor
        omega_3 = .2*self.reference_speed_factor
        omega_4 = .4*self.reference_speed_factor

        signal = 2*np.sin(omega_1*t+ self.phi[0]) + 0.5*np.sin(omega_2*t + np.pi/7+ self.phi[1]) + \
                0.8*np.sin(omega_3*t - np.pi/12+ self.phi[2])+ 1*np.cos(omega_4*t+np.pi/5+ self.phi[3])
        return signal

    def get_values(self, i, dt):

        alpha = 0.01
        x_target_new = self.fun_val(self.duration) - self.fun_val(0)*np.exp(-alpha*self.duration)

        eps_new = (x_target_new - self.x_target)/dt
        dot_eps = (eps_new - self.eps)/dt

        self.duration += dt
        self.x_target = x_target_new
        self.eps = eps_new

        return torch.tensor(dot_eps), torch.tensor(self.mu[i])

    def get_signal(self, dt):

        signal = []
        for i in range(self.test_steps):
            signal.append(self.get_values(i=i, dt=dt))

        return signal


class CartPole_classic(CartPole):

    def __init__(self, Q = np.diag([1,1,1,1]), R = 1):
        super(CartPole_classic, self).__init__()
        
        self.L = self.lpole/2
    
        # cartpole params
        self.d1 = 0.0
        self.d2 = 0.0
        
        # CONTROL OPTIONS FROM HERE ON        
        # force max
        self.force_max = 8*(self.mpole + self.mcart)

        # LQR controller params
        self.ctrl_LQR_max = self.force_max*0.55
        self.u_LQR_ctrl = 0
        self.K = None
        self.Q = Q
        self.R = R

        # proportional x controller parameters
        self.Kv  = 40
        self.Kp  = 10
        self.x0_old = 0
        self.dt = 0.05
        self.correct_x = True
        self.ctrl_Kp_max = self.force_max*0.65

        # robust SMC parameters
        self.smc_band_params = [0.1,0.6]
        self.SMC_control = True
        self.u_SMC = 0
        self.alpha_sliding = 1
        self.K_smc = self.force_max

        # initialize linearized system A,B
        self.update_A_B()
        #self.reset_store(np.array([0,0,0,0]))
        

    
    def update_A_B(self):
        _q = (self.mpole+self.mcart) * self.gravity / (self.mcart*self.L)
        self.A = np.array([\
                    [0,1,0,0], \
                    [0,-self.d1, -self.gravity*self.mpole/self.mcart,0],\
                    [0,0,0,1.],\
                    [0,self.d1/self.L,_q,-self.d2] ] )

        self.B = np.expand_dims( np.array( [0, 1.0/self.mcart, 0., -1/(self.mcart*self.L)] ) , 1 ) # 4x1
        
    def get_ctrl_signal(self, split_components = False):
       
        ctrl = self.u_LQR_ctrl
       
        if self.SMC_control:
            ctrl += self.u_SMC
            
        if self.correct_x:
            ctrl += self.u_Kp_correction
        
        if split_components:
            return np.array([self.u_LQR_ctrl, self.u_Kp_correction, self.u_SMC])[np.newaxis,:]
        else:
            return np.array([np.clip(ctrl,-self.force_max ,self.force_max)])

    def computeControlSignals(self, state, x_target=0):
        self.SMC_law(state)
        self.LQR_law(state, x_target)
        self.Kp_x_law(state, x0 = x_target)

    def update(self, dt, action, mu, dot_eps, fixed_env=True):

        state = [self.x.item(), self.dot_x.item(), self.theta.item(), self.dot_theta.item()]
        self.computeControlSignals(state, x_target=self.x_target)
        action = torch.tensor(self.get_ctrl_signal())
        super().update(dt=dt, action=action, mu=mu, dot_eps=dot_eps, fixed_env=fixed_env)

    def saturate(self, signal, max_value):
        """ saturate signals
        """
        return np.clip(signal, -max_value, max_value)
        
    
    ############################################################
    def SMC_law(self, state):
        if self.SMC_control:
           #z = odeint(ode_fun, self.state_z, delta_t)
            zero_action_bound = 0.1#5
            saturation_init = .6 #1.5
            sigma = state[2]*self.alpha_sliding + state[3]
            
            control_action = 0
            if abs(sigma) > zero_action_bound:
                if abs(sigma)> saturation_init:
                    control_action = np.sign(sigma)
                else:
                    control_action = np.sign(sigma)*(abs(sigma)-zero_action_bound)/(saturation_init-zero_action_bound)
            self.u_SMC  = control_action *self.K_smc
        return self.u_SMC 
   
    
    def Kp_x_law(self, state, x0 = 0):  
        """proportional control for x correction"""    
        dot_x_ref_est = (x0 - self.x0_old) / self.dt

        err_v = dot_x_ref_est - state[1]
        err_x = x0 - state[0]
            
        self.u_Kp_correction =  self.saturate(-self.Kp * err_x -self.Kv * err_v, self.ctrl_Kp_max)
        self.x0_old = x0
            
        return self.u_Kp_correction


    def LQR_law(self, state_in, x0, Q=np.eye(4), R=1 ):
        """ LQR control law """
        state = state_in.copy()
        if self.K is None:
            K, X, eigVals = lqr (self.A, self.B, Q, R )
            self.K = K
        
        theta_max = 0.1
        state[2] = np.clip(state[2],-theta_max, theta_max)
        
        LQR_ctrl = -np.matmul( self.K , state - np.array([x0,0,0,0]) )[0,0]   
        self.u_LQR_ctrl =  self.saturate(LQR_ctrl, self.ctrl_LQR_max)
            
        return self.u_LQR_ctrl


class Model_classic(Model):

    def __init__(self, param_generator):

        self.cartpole = CartPole_classic()

        self.agent = Agent(self.cartpole)
        self.environment = Environment(self.cartpole)

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

        self._param_generator = param_generator
        self.traces = None


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
