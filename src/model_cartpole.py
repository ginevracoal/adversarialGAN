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

        self.x, self.theta = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.dot_theta, self.dot_x = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.).to(dtype=torch.float32))
        self.mu, self.inp_acc = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))

        self._max_x = 30.
        self._max_theta = 1.57 # 3.1415/2
        self._max_dot_x = 100.
        self._max_dot_theta = 100.
        self._max_ddot_x = 100.
        self._max_ddot_theta = 100.
        self._max_inp_acc=100.
        self._max_mu=0.05

    def update(self, dt, inp_acc=None, mu=None):
        """
        Update the system state.
        """        
        if inp_acc is not None:
            self.inp_acc = torch.clamp(inp_acc, -self._max_inp_acc, self._max_inp_acc)

        if mu is not None:
            self.mu = torch.clamp(mu, -self._max_mu, self._max_mu)

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

            # Control cart
            f = (mp + mc) * self.inp_acc
 
            # ODE equations
            if self.ode_idx==0: # Barto 
 
                mu_c = self.mu #0.0005
                mu_p = 0.000002
                num = (-f-mp*l*dot_theta**2*torch.sin(theta)+mu_c*torch.sign(dot_x))/(mc+mp)
                den = 4/3 - (mp*torch.cos(theta)**2)/(mc+mp)
                ddot_theta = (g*torch.sin(theta)+torch.cos(theta)*num-mu_p*dot_theta/(mp*l))/(l*den)
                ddot_x = (f+mp*l*(ddot_theta**2*torch.sin(theta)-ddot_theta*torch.cos(theta))-mu_c*torch.sign(dot_x))/(mc+mp)

            elif self.ode_idx==1: # air drag attacker
                
                rho=1.2
                h=0.05
                numer = f - mp*g*torch.sin(theta)*torch.cos(theta)+self.mu*rho*dot_theta**2*h*(l+torch.cos(theta)**2)+mp*l*dot_theta*torch.sin(theta)
                denom = mc+mp*torch.sin(theta)**2
                ddot_x = numer/denom
                ddot_theta = (-mp*ddot_x*torch.cos(theta)+mp*g*torch.sin(theta)+self.mu*rho*dot_theta**2*h*torch.cos(theta))/(mp*l)

            else:
                raise NotImplementedError()
            
            dqdt = torch.FloatTensor([dot_x, dot_theta, ddot_x, ddot_theta]).to(device=self.device, dtype=torch.float32)

            self.ddot_x = torch.clamp(ddot_x, -self._max_ddot_x, self._max_ddot_x).reshape(1)
            self.ddot_theta = torch.clamp(ddot_theta, -self._max_ddot_theta, self._max_ddot_theta).reshape(1)
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

        print(f"x={self.x.item()}\ttheta={self.theta.item()}\tddot_x={self.ddot_x.item()}\tinp_acc={self.inp_acc.item()}\tmu={self.mu.item()}")

        # print(f"dot_x={self.dot_x.item()}\tddot_x={self.ddot_x.item()}")

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
        mu = parameters
        self._cartpole.update(mu=mu, dt=dt)


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
        self._cartpole.update(inp_acc=cart_acceleration, dt=dt)


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
        
