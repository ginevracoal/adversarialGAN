import json
import torch
from utils.diffquantitative import DiffQuantitativeSemantic


DEBUG=False
FRICTION=True
ALPHA=0.4
K=10

class CartPole():

    def __init__(self):

        self.gravity = 9.81 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1. # cart mass in kg
        self.mpole = .1 # pole mass in kg
        self.lpole = 1. # pole length in meters

        self.x, self.theta = (torch.tensor(0.0), torch.tensor(0.0))
        self.dot_theta, self.dot_x = (torch.tensor(0.0), torch.tensor(0.0))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.0), torch.tensor(0.))
        self.x_target = torch.tensor(0.0)
        self.dist = torch.tensor(0.0)
        self.mu = torch.tensor(0.0)

        self._max_x = 5.
        self._max_theta = 1.5
        self._max_dot_x = 10
        self._max_dot_theta = 10
        self._max_dot_eps=1.
        self._max_mu=1.

    def update(self, dt, action, mu, dot_eps):    

        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        l = self.lpole/2
            
        f = action
        dot_eps = torch.clamp(dot_eps, -self._max_dot_eps, self._max_dot_eps)
        # update_mu = np.random.binomial(n=1, p=0.1)
        mu = torch.clamp(mu, -self._max_mu, self._max_mu) #if update_mu==1 else None

        eps = dot_eps * dt
        x_target = self.x + eps
        self.x_target = torch.clamp(x_target, -self._max_x, self._max_x)
        self.dist = torch.abs(self.x-self.x_target)

        if FRICTION:
            ddot_x = f - mu*self.dot_x  \
                       + mp*l*self.dot_theta**2* torch.sin(self.theta) \
                       - mp*g*torch.cos(self.theta) * torch.sin(self.theta)
            ddot_x = ddot_x / ( mc+mp-mp* torch.cos(self.theta)**2 )
            ddot_theta = (g*torch.sin(self.theta) - torch.cos(self.theta)*ddot_x ) / l
        
        else:
            temp = (f+mp*l*self.dot_theta**2*torch.sin(self.theta))/(mp+mc)
            denom = l*(4/3-(mp*torch.cos(self.theta)**2)/(mp+mc))
            ddot_theta = (g*torch.sin(self.theta)-torch.cos(self.theta)*temp)/denom
            ddot_x = temp - (mp*l*ddot_theta*torch.cos(self.theta))/(mp+mc)
        
        dot_x = self.dot_x + dt * ddot_x
        dot_theta = self.dot_theta + dt * ddot_theta
        x = self.x + dt * dot_x
        theta = self.theta + dt * dot_theta
        
        ## print(f"\neq diff out:\t {ddot_x.item():.4f} {ddot_theta.item():.4f}")
        # print(f"x update:\t{self.x.item():.4f} -> {x.item():.4f}")
        # print(f"theta update:\t{self.theta.item():.4f} -> {theta.item():.4f}\n")

        self.x = torch.clamp(x, -self._max_x, self._max_x)
        self.theta = torch.clamp(theta, -self._max_theta, self._max_theta)
        self.dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x)
        self.dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta)

        if DEBUG:
            print(f"dist={self.dist.item():.4f}  mu={mu.item():.4f}", end="\t")
            print(f"x={self.x.item():4f}, theta={self.theta.item():4}")


class Environment:
    def __init__(self, cartpole):
        self._cartpole = cartpole

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 2
        self.sensors = len(self.status)

    @property
    def x(self):
        return self._cartpole.x.clone()

    @x.setter
    def x(self, value):
        self._cartpole.x = value

    @property
    def theta(self):
        return self._cartpole.theta.clone()

    @theta.setter
    def theta(self, value):
        self._cartpole.theta = value

    @property
    def x_target(self):
        return self._cartpole.x_target.clone()

    @x_target.setter
    def x_target(self, value):
        self._cartpole.x_target = value

    @property
    def dist(self):
        return self._cartpole.dist.clone()

    @dist.setter
    def dist(self, value):
        self._cartpole.dist = value

    @property
    def status(self):
        return (self._agent.x, self._agent.theta, self._cartpole.dist)


class Agent:
    def __init__(self, cartpole):
        self._cartpole = cartpole

    def set_environment(self, environment):
        self._environment = environment
        self.initialized()

    def initialized(self):
        self.actuators = 1
        self.sensors = len(self.status)

    @property
    def x(self):
        return self._cartpole.x.clone()

    @x.setter
    def x(self, value):
        self._cartpole.x = value

    @property
    def dot_x(self):
        return self._cartpole.dot_x.clone()

    @dot_x.setter
    def dot_x(self, value):
        self._cartpole.dot_x = value

    @property
    def ddot_x(self):
        return self._cartpole.ddot_x.clone()

    @ddot_x.setter
    def ddot_x(self, value):
        self._cartpole.ddot_x = value

    @property
    def theta(self):
        return self._cartpole.theta.clone()

    @theta.setter
    def theta(self, value):
        self._cartpole.theta = value

    @property
    def dot_theta(self):
        return self._cartpole.dot_theta.clone()

    @dot_theta.setter
    def dot_theta(self, value):
        self._cartpole.dot_theta = value

    @property
    def x_target(self):
        return self._cartpole.x_target.clone()

    @x_target.setter
    def x_target(self, value):
        self._cartpole.x_target = value

    @property
    def status(self):
        return (self.x, self.dot_x, self.theta, self.dot_theta, self.x_target)


class Model:
    
    def __init__(self, param_generator):
        # setting of the initial conditions
        self.cartpole = CartPole_ext()

        self.agent = Agent(self.cartpole)
        self.environment = Environment(self.cartpole)

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

        self._param_generator = param_generator
        self.traces = None

    def step(self, env_input, agent_input, dt):
        dot_eps, mu = env_input
        action = agent_input

        self.cartpole.update(dt=dt, action=action, mu=mu, dot_eps=dot_eps)

        self.traces['dist'].append(self.environment.dist)
        self.traces['theta'].append(self.agent.theta)

    def initialize_random(self):
        cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target = next(self._param_generator)

        self._last_init = (cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target)

        self.reinitialize(cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target):

        self.agent.x = torch.tensor(cart_position).reshape(1)
        self.agent.dot_x = torch.tensor(cart_velocity).reshape(1)
        self.agent.theta = torch.tensor(pole_angle).reshape(1)
        self.agent.dot_theta = torch.tensor(pole_ang_velocity).reshape(1)
        self.agent.x_target = torch.tensor(x_target).reshape(1)

        self.traces = {'theta': [], 'dist':[]}

class RobustnessComputer:
    def __init__(self, formula_theta, formula_dist):
        self.dqs_theta = DiffQuantitativeSemantic(formula_theta)
        self.dqs_dist = DiffQuantitativeSemantic(formula_dist)

    def compute(self, model):
        theta = model.traces['theta'][-K:]
        dist = model.traces['dist'][-K:]
        rob_theta = self.dqs_theta.compute(theta=torch.cat(theta))
        rob_dist = self.dqs_dist.compute(dist=torch.cat(dist))

        return ALPHA*rob_dist+(1-ALPHA)*rob_theta
    


################################################################################
################################################################################
################################################################################
################################################################################
    
import scipy.linalg
import numpy as np


class CartPole_ext(CartPole):
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
