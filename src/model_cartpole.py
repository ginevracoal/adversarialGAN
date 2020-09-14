import json
import torch
import numpy as np
import random

from diffquantitative import DiffQuantitativeSemantic
from torchdiffeq import odeint
# from scipy.integrate import odeint

class CartPole():

    def __init__(self, device, ode_idx):

        self.ode_idx=ode_idx

        if device == "cuda":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.device = device

        self.gravity = 9.8 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 10.0 # cart mass in kg
        self.mpole = 1. # pole mass in kg
        self.lpole = 1.0 # pole length in meters
        self.c1 = 0.001

        self.x, self.theta = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.dot_theta, self.dot_x = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.).to(dtype=torch.float32), torch.tensor(0.).to(dtype=torch.float32))
        self.mu = torch.tensor(0.).to(dtype=torch.float32) 

        # self._max_x = 10.
        self._max_theta = 1.57 # 3.1415/2
        self._max_dot_x = 10.
        self._max_dot_theta = 10.
        self._max_ddot_x = 100.
        self._max_ddot_theta = 100.
        self._max_u = 2.

    def update(self, dt, ddot_x=None, mu=None):
        """
        Update the system state.
        """        

        if ddot_x is not None:
            self.ddot_x = ddot_x

        if mu is not None:
            self.mu = mu

        def ode_func(dt, q):

            # Locals for readability.
            g = self.gravity
            mp = self.mpole
            mc = self.mcart
            M = mp + mc
            L = self.lpole
            
            x, theta, dot_x, dot_theta = q[0], q[1], q[2], q[3]
            # x = torch.clamp(x, -self._max_x, self._max_x).reshape(1)
            theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
            dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
            dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)

            # Control cart
            if self.ode_idx==0 or self.ode_idx==1 or self.ode_idx==3:

                f = M * self.ddot_x

            elif self.ode_idx==2:
    
                u_hat = self.ddot_x 

                if torch.abs(x) < self._max_x:
                    u = max(-self._max_u, min(self._max_u, u_hat))
                
                elif x < -self._max_x:
                    u = max(0., min(self._max_u, u_hat))
                
                else: # x > self._max_x:
                    u = max(-self._max_u, min(self._max_u, u_hat))

                a = 20. * (u-dot_x)

                # a = self.ddot_x

            # ODE equations   
            if self.ode_idx==0: # Florian

                delta = 1/(mp*torch.sin(theta)**2 + mc)
                ddot_x = delta * (f + mp * torch.sin(theta) * (L * dot_theta**2 
                                                         + g * torch.cos(theta) ))
                ddot_theta = delta/L * (- f * torch.cos(theta) 
                                         - mp * L * dot_theta**2 * torch.cos(theta) * torch.sin(theta)
                                         - M * g * torch.sin(theta))

            elif self.ode_idx==1: # Barto

                l = L/2 
                mu_c, mu_p = 0.00001, 0.00001

                air_drag = self.mu * dot_theta**2 * torch.sign(dot_theta)
                Nc = M * g - mp * l * (self.ddot_theta * torch.sin(theta) + dot_theta**2 * torch.cos(theta)) 
                
                numer = (-f - mp * l * dot_theta**2 * (torch.sin(theta) + mu_c * torch.sign(Nc*dot_x)*torch.cos(theta))) / M + mu_c * g * torch.sign(Nc * dot_x)
                denom = 4/3 - mp * torch.cos(theta) * (torch.cos(theta) - mu_c * torch.sign(Nc * dot_x)) / M

                ddot_theta = (g * torch.sin(theta) + torch.cos(theta) * numer - (mu_p * dot_theta)/(mp * l))/(l * denom) - air_drag
                ddot_x = (f + mp * l * (dot_theta**2 * torch.sin(theta) - ddot_theta * torch.cos(theta)) - mu_c * Nc * torch.sign(Nc * dot_x))/M

                # print(ddot_x.item(), ddot_theta.item(), theta.item()) 

            elif self.ode_idx==2: # Balcerzak

                viscous_friction = self.c1 * dot_theta
                air_drag = self.mu * dot_theta**2 * torch.sign(dot_theta)
                ddot_x = a
                ddot_theta = (3*g/L) * torch.sin(theta) + (3*a/L) * torch.cos(theta) - viscous_friction - air_drag
            
            elif self.ode_idx==3:
                
                l = L/2 
                denom = M + mp * torch.cos(theta)**2
                ddot_x = (mp * g * torch.sin(theta) * torch.cos(theta) + self.mu * dot_theta * torch.sign(theta) * torch.cos(theta) + mp * l * dot_theta * torch.sin(theta) + f) / denom
                ddot_theta = g * torch.sin(theta)/l + self.mu * dot_theta * torch.sign(dot_theta) / (mp*l) - self.ddot_x * torch.cos(theta)/l

            else:
                raise NotImplementedError()

            dqdt = torch.FloatTensor([dot_x, dot_theta, ddot_x, ddot_theta]).to(device=self.device, dtype=torch.float32)

            self.ddot_x = torch.clamp(ddot_x.reshape(1), -self._max_ddot_x, self._max_ddot_x)
            self.ddot_theta = ddot_theta.reshape(1)
            return dqdt
        
        # Solve the ODE
        q0 = torch.FloatTensor([self.x, self.theta, self.dot_x, self.dot_theta]).to(device=self.device, dtype=torch.float32) 
        t = torch.FloatTensor(np.linspace(0, dt, 2)).to(device=self.device, dtype=torch.float32) 
        q = odeint(func=ode_func, y0=q0, t=t) 

        x, theta, dot_x, dot_theta = q[1]
        self.x = x #torch.clamp(x, -self._max_x, self._max_x).reshape(1)
        self.theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
        self.dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
        self.dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)

        print(self.x.item(), self.theta.item(), self.mu.item())#, self.ddot_theta.item())
        # print(self.mu * self.dot_theta**2 * torch.sign(self.dot_theta))


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
        return self.dqs.compute(theta=torch.cat(theta))
        
