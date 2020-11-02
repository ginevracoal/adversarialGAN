import json
import torch
import numpy as np
import random
from diffquantitative import DiffQuantitativeSemantic
from torchdiffeq import odeint


class CartPole():

    def __init__(self):

        self.air_drag = False
        self.cart_friction = True

        self.gravity = 9.81 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1. # cart mass in kg
        self.mpole = .1 # pole mass in kg
        self.lpole = 1. # pole length in meters

        self.x, self.theta = (torch.tensor(0.0), torch.tensor(0.0))
        self.dot_theta, self.dot_x = (torch.tensor(0.0), torch.tensor(0.0))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.0), torch.tensor(0.))
        self.x_target = torch.tensor(0.0)

        self.inp_acc = torch.tensor(0.0)
        self.eps = torch.tensor(0.0)
        self.dist = torch.tensor(0.0)
        self.mu = torch.tensor(0.0)
        self.nu = torch.tensor(0.0)
        self.f = torch.tensor(0.0)#.to(dtype=torch.float32)

        self._max_x = 100.
        self._max_theta = 1.5
        self._max_dot_x = 1000.
        self._max_dot_theta =  1000.
        self._max_ddot_x = 100.
        self._max_ddot_theta = 100.

        self.theta_0_ths = .8
        self.theta_1_ths = .8
        self.ctrl_magnitude = 100

    def update(self, dt, inp_acc=None, dot_eps=None, mu=None, nu=None):
        """
        Update the system state.
        """        
        if inp_acc is not None:
            self.inp_acc = inp_acc

        if self.cart_friction is True:

            if mu is not None:
                self.mu = mu

        if self.air_drag is True:
            if nu is not None:
                self.nu = nu

        if dot_eps is not None:

            # if torch.abs(self.theta) >= self.theta_0_ths:
            #     dot_eps = - torch.abs(dot_eps) * torch.sign(self.theta)

            eps = dot_eps * dt
            self.x_target = (self.x+eps)

        def ode_func(dt, q):

            # Locals for readability.
            g = self.gravity
            mp = self.mpole
            mc = self.mcart
            L = self.lpole
            l = L/2 
            
            x, theta, dot_x, dot_theta = q[0], q[1], q[2], q[3]
            x = torch.clamp(x, -self._max_x, self._max_x).reshape(1)
            theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
            dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
            dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)

            self.f = (mp + mc) * self.inp_acc

            # if torch.abs(theta) >= self.theta_1_ths:
            #     self.f = torch.sign(theta) * self.ctrl_magnitude

            # elif torch.abs(theta) < self.theta_0_ths:
            #     self.f = (mp + mc) * self.inp_acc

            # else:
            #     self.f = (mp + mc) * self.inp_acc if theta*dot_theta>0 else torch.sign(theta) * ctrl_magnitude
                

            ########## Barto + air drag
            # rho=1.2
            # A1 = L*0.01
            # A2 = 0.3*0.3
            # air_fric = self.nu*rho*dot_theta**2*(A1+A2)

            # num = (-self.f
            #        -mp*l*dot_theta**2*torch.sin(theta)
            #        +self.mu*torch.sign(dot_x)
            #        +air_fric
            #        )/(mp+mc)

            # den = 4/3-(mp*torch.cos(theta)**2)/(mp+mc)

            # ddot_theta = (g*torch.sin(theta)+torch.cos(theta)*num)/(l*den)

            # ddot_x = (self.f
            #          +mp*l*(dot_theta**2*torch.sin(theta)-ddot_theta*torch.cos(theta))\
            #          -self.mu*torch.sign(dot_x)
            #          -air_fric
            #          )/(mp+mc)

            #########################

            ddot_x = self.f - self.mu*dot_x  \
                       + mp*l*dot_theta**2* torch.sin(theta) \
                       - mp*g*torch.cos(theta) * torch.sin(theta)
            ddot_x = ddot_x / ( mc+mp-mp* torch.cos(theta)**2 )
        
            ddot_theta = (g*torch.sin(theta) - torch.cos(theta)*ddot_x ) / l
    
            #########################

            dqdt = torch.tensor([dot_x, dot_theta, ddot_x, ddot_theta], requires_grad=True)#\
                                     #.to(device=self.device)

            self.ddot_x = torch.clamp(ddot_x, -self._max_ddot_x, self._max_ddot_x).reshape(1)
            self.ddot_theta = torch.clamp(ddot_theta, -self._max_ddot_theta, self._max_ddot_theta).reshape(1)

            return dqdt
                    
        # Solve the ODE
        q0 = torch.tensor([self.x, self.theta, self.dot_x, self.dot_theta])#\
                               # .to(device=self.device)
        t = torch.tensor(np.linspace(0, dt, 2))#.to(device=self.device)
        q = odeint(func=ode_func, y0=q0, t=t)

        x, theta, dot_x, dot_theta = q[1]

        self.dist = torch.abs(x-self.x_target)
        self.x = torch.clamp(x, -self._max_x, self._max_x).reshape(1)
        self.theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
        self.dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
        self.dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)
        
        # print(f"x-x_target={(x-self.x_target).item():.4f}\
        #         theta={self.theta.item():.4f}\
        #         mu={self.mu.item():.4f}\
        #         f={self.f.item()}")
        
class Environment:
    def __init__(self, cartpole):
        self._cartpole = cartpole

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 3
        self.sensors = len(self.status)

    @property
    def status(self):
        return (self._agent.x,
                self._agent.theta,  
                self._agent.dot_x,
                self._agent.dot_theta,
                self._agent.x_target)

    def update(self, parameters, dt):
        # the environment updates according to the parameters
        update_mu = np.random.binomial(n=1, p=0.1)

        if update_mu==1:
            dot_eps, mu, nu = parameters
        else:
            dot_eps, _, nu = parameters
            mu = None

        self._cartpole.update(dot_eps=dot_eps, mu=mu, nu=nu, dt=dt)


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
    def dist(self):
        return self._cartpole.dist.clone()

    @dist.setter
    def dist(self, value):
        self._cartpole.dist = value

    @property
    def status(self):
        return (self.x,
                self.theta,
                self.dot_x,
                self.dot_theta,
                self.x_target)

    def update(self, parameters, dt):
        # the action take place and updates the variables
        cart_acceleration = parameters
        cart_acceleration = cart_acceleration
        self._cartpole.update(inp_acc=cart_acceleration, dt=dt)


class Model:
    
    def __init__(self, param_generator):
        # setting of the initial conditions
        cartpole = CartPole()

        self.agent = Agent(cartpole)
        self.environment = Environment(cartpole)

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

        self._param_generator = param_generator
        self.traces = None

    def step(self, env_input, agent_input, dt):

        self.environment.update(env_input, dt)
        self.agent.update(agent_input, dt)

        self.traces['dist'].append(self.agent.dist)
        self.traces['theta'].append(self.agent.theta)

    def initialize_random(self):
        cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target = next(self._param_generator)

        self._last_init = (cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target)

        self.reinitialize(cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target):
        # print("\n")

        self.agent.x = torch.tensor(cart_position).reshape(1)
        self.agent.dot_x = torch.tensor(cart_velocity).reshape(1)
        self.agent.theta = torch.tensor(pole_angle).reshape(1)
        self.agent.dot_theta = torch.tensor(pole_ang_velocity).reshape(1)
        self.agent.x_target = torch.tensor(x_target).reshape(1)

        self.traces = {'dist':[], 'theta': []}

class RobustnessComputer:
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        dist = model.traces['dist']
        theta = model.traces['theta']
        return self.dqs.compute(dist=torch.cat(dist), theta=torch.cat(theta))
        # return self.dqs.compute(dist=torch.stack(dist, dim=0), theta=torch.stack(theta, dim=0))
        
