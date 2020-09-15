import json
import torch
import numpy as np
import random

from diffquantitative import DiffQuantitativeSemantic
from torchdiffeq import odeint


class CartPole():

    def __init__(self, device, ode_idx):

        self.ode_idx=ode_idx

        if device == "cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device

        self.gravity = 9.8 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1.0 # cart mass in kg
        self.mpole = 0.1 # pole mass in kg
        self.lpole = 1.0 # pole length in meters
        self.c1 = 0.001

        self.x, self.theta = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.dot_theta, self.dot_x = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.).to(dtype=torch.float32), torch.tensor(0.).to(dtype=torch.float32))
        self.mu = torch.tensor(0.05).to(dtype=torch.float32) 

        self._max_x = 30.
        self._max_theta = 1.57 # 3.1415/2
        self._max_dot_x = 100.
        self._max_dot_theta = 100.
        self._max_ddot_x = 1000.
        self._max_ddot_theta = 100.
        self._max_mu=0.05

    def update(self, dt, ddot_x=None, mu=None):
        """
        Update the system state.
        """        
        if ddot_x is not None:
            self.ddot_x = ddot_x

        if mu is not None:

            if self.ode_idx == 0:
                self.mu = 0.
            else:
                self.mu = torch.clamp(mu, -self._max_mu, self._max_mu)

        def ode_func(dt, q):

            # Locals for readability.
            g = self.gravity
            mp = self.mpole
            mc = self.mcart
            M = mp + mc
            L = self.lpole
            
            x, theta, dot_x, dot_theta = q[0], q[1], q[2], q[3]
            x = torch.clamp(x, -self._max_x, self._max_x).reshape(1)
            theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
            dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
            dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)

            # Control cart
            f = M * self.ddot_x

            # ODE equations   
            if self.ode_idx==0: # no attacker

                delta = 1/(mp*torch.sin(theta)**2 + mc)
                ddot_x = delta * (f + mp * torch.sin(theta) * (L * dot_theta**2 
                                                         + g * torch.cos(theta) ))
                ddot_theta = delta/L * (- f * torch.cos(theta) 
                                         - mp * L * dot_theta**2 * torch.cos(theta) * torch.sin(theta)
                                         - M * g * torch.sin(theta))

            elif self.ode_idx==1: # air drag attacker
                
                l = L/2 
                denom = M + mp * (1+torch.cos(theta)**2)
                ddot_x = (f+mp * g * torch.sin(theta) * torch.cos(theta) + self.mu*dot_theta*torch.cos(theta)*torch.sign(dot_theta) - mp * l * (dot_theta**2) * torch.sin(theta)) / denom
                ddot_theta = (mp*g*torch.sin(theta) + self.mu * dot_theta * torch.sign(dot_theta) + mp*ddot_x * torch.cos(theta))/(mp*l)

            else:
                raise NotImplementedError()

            dqdt = torch.FloatTensor([dot_x, dot_theta, ddot_x, ddot_theta]).to(device=self.device, dtype=torch.float32)

            self.ddot_x = torch.clamp(ddot_x.reshape(1), -self._max_ddot_x, self._max_ddot_x)
            self.ddot_theta = torch.clamp(ddot_theta.reshape(1), -self._max_ddot_theta, self._max_ddot_theta)
            return dqdt
                    
        # Solve the ODE
        q0 = torch.FloatTensor([self.x, self.theta, self.dot_x, self.dot_theta]).to(device=self.device, dtype=torch.float32) 
        t = torch.FloatTensor(np.linspace(0, dt, 2)).to(device=self.device, dtype=torch.float32) 
        q = odeint(func=ode_func, y0=q0, t=t) 

        x, theta, dot_x, dot_theta = q[1]
        self.x = torch.clamp(x, -self._max_x, self._max_x).reshape(1)
        self.theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
        self.dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
        self.dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)

        print(f"x={self.x.item()}\ttheta={self.theta.item()}\tddotx={self.ddot_x.item()}\tmu={self.mu.item()}")


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
        air_drag_coef = parameters
        self._cartpole.update(mu=air_drag_coef, dt=dt)


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
    
    def __init__(self, param_generator, device="cuda", ode_idx=0):
        # setting of the initial conditions

        cartpole = CartPole(device=device, ode_idx=ode_idx)

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
        theta = model.traces['theta']
        x = model.traces['x']
        return self.dqs.compute(theta=torch.cat(theta), x=torch.cat(x))
        
