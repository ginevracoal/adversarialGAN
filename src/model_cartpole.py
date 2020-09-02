import json
import torch
import numpy as np
import random

from diffquantitative import DiffQuantitativeSemantic
from torchdiffeq import odeint

class CartPole():

    def __init__(self):

        self.gravity = 9.8 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1.0 # cart mass in kg
        self.mpole = 0.1 # pole mass in kg
        self.lpole = 0.5 # pole length in meters

        self._max_x = 40
        self._min_x = -self._max_x
        self._max_theta = 3.1415/2
        self._min_theta = -self._max_theta

        self.x, self.theta = (torch.tensor(0.0).float(), torch.tensor(0.0).float())
        self.dot_theta, self.dot_x = (torch.tensor(0.0).float(), torch.tensor(0.0).float())
        # self.ddot_theta, self.ddot_x = (torch.tensor(0.0), torch.tensor(0.0))

    def update(self, ddot_x, dt):
        """
        Update the system state.
        """        

        # Control cart
        f = (self.mcart + self.mpole) * ddot_x

        def ode_func(dt, q):
            # Locals for readability.
            g = self.gravity
            mp = self.mpole
            mc = self.mcart
            M = mp + mc
            L = self.lpole
            
            x, theta, dot_x, dot_theta = q[0], q[1], q[2], q[3]
    
            # ODE equations        
            delta = 1/(mp*torch.sin(theta)**2 + mc)
            ddot_x = delta * (f + mp * torch.sin(theta) * (L * dot_theta**2 
                                                     + g * torch.cos(theta) ))
            ddot_theta  = delta/L * (- f * torch.cos(theta) 
                                     - mp * L * dot_theta**2 * torch.cos(theta) * torch.sin(theta)
                                     - M * g * torch.sin(theta))

            dqdt = torch.FloatTensor([dot_x, dot_theta, ddot_x, ddot_theta])
            return dqdt
        
        # Solve the ODE
        q0 = torch.FloatTensor([self.x, self.theta, self.dot_x, self.dot_theta])
        t = torch.FloatTensor(np.linspace(0, dt, 2))
        q = odeint(func=ode_func, y0=q0, t=t)

        x, theta, dot_x, dot_theta = q[1]
        self.x = torch.clamp(x, self._min_x, self._max_x).reshape(1)
        self.theta = torch.clamp(theta, self._min_theta, self._max_theta).reshape(1)
        self.dot_x = dot_x.reshape(1)
        self.dot_theta = dot_theta.reshape(1)


class Environment:
    def __init__(self, cartpole):
        self._cartpole = cartpole

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 1
        self.sensors = len(self.status)

    @property
    def status(self):
        return (self._agent.x,
                self._agent.theta,
                self._agent.dot_x,
                self._agent.dot_theta)

    def update(self, parameters, dt):
        # the environment updates according to the parameters
        cart_acceleration = parameters
        self._cartpole.update(dt=dt, ddot_x=cart_acceleration)


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
    def status(self):
        return (self.x,
                self.theta,
                self.dot_x,
                self.dot_theta)

    def update(self, parameters, dt):
        # the action take place and updates the variables
        cart_acceleration = parameters
        self._cartpole.update(dt=dt, ddot_x=cart_acceleration)


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

        self.traces['x'].append(self.agent.x)
        self.traces['theta'].append(self.agent.theta)

    def initialize_random(self):
        cart_position, cart_velocity, pole_angle, pole_ang_velocity = next(self._param_generator)

        self._last_init = (cart_position, cart_velocity, pole_angle, pole_ang_velocity)

        self.reinitialize(cart_position, cart_velocity, pole_angle, pole_ang_velocity)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, cart_position, cart_velocity, pole_angle, pole_ang_velocity):
        self.agent.x = torch.tensor(cart_position).reshape(1).float()
        self.agent.dot_x = torch.tensor(cart_velocity).reshape(1).float()
        self.agent.theta = torch.tensor(pole_angle).reshape(1).float()
        self.agent.dot_theta = torch.tensor(pole_ang_velocity).reshape(1).float()

        self.traces = {
            'x': [],
            'theta': []
        }

class RobustnessComputer:
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        x = model.traces['x']
        theta = model.traces['theta']
        return self.dqs.compute(x=torch.cat(x), theta=torch.cat(theta))
        
