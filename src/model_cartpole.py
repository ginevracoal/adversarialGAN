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

        self.gravity = -9.81 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1. # cart mass in kg
        self.mpole = .1 # pole mass in kg
        self.lpole = 1. # pole length in meters

        self.x, self.theta = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.dot_theta, self.dot_x = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.0).to(dtype=torch.float32))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.0).to(dtype=torch.float32), torch.tensor(0.).to(dtype=torch.float32))
        self.nu = torch.tensor(0.0).to(dtype=torch.float32)
        self.mu = torch.tensor(0.0).to(dtype=torch.float32)
        self.inp_acc = torch.tensor(0.0).to(dtype=torch.float32)

        self._max_x = 10.
        self._max_theta = 3.14 #1.57
        self._max_dot_x = 10.
        self._max_dot_theta = 3.
        self._max_ddot_x = 10.
        self._max_ddot_theta = 3.
        self._max_inp_acc=10.

    def update(self, dt, inp_acc=None, nu=None, mu=None):
        """
        Update the system state.
        """        
        if inp_acc is not None:
            self.inp_acc = inp_acc

        if nu is not None:
            self.nu = nu

        if mu is not None and self.ode_idx==2:
            self.mu = mu

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
            if self.ode_idx==0: # cart friction (Barto)
 
                mu_c = self.nu 
                mu_p = 0.000002
                numer = (- f \
                         - mp*l*dot_theta**2*torch.sin(theta)\
                         + mu_c*torch.sign(dot_x))/(mc+mp)
                denom = 4/3 - (mp*torch.cos(theta)**2)/(mc+mp)
                ddot_theta = (g*torch.sin(theta)\
                              + torch.cos(theta)*numer\
                              - mu_p*dot_theta/(mp*l))/(l*denom)
                ddot_x = (f + mp*l*(ddot_theta**2*torch.sin(theta)
                                    - ddot_theta*torch.cos(theta))
                            - mu_c*torch.sign(dot_x))/(mc+mp)

            elif self.ode_idx==1 or self.ode_idx==2: # air drag + cart friction

                rho=1.2
                A1 = L*0.01
                A2 = 0.3*0.3
                numer = f - mp*g*torch.sin(theta)*torch.cos(theta)\
                          - self.mu*rho*dot_theta**2*(A1/L)*(torch.cos(theta)**2-l)\
                          + mp*l*dot_theta*torch.sin(theta)\
                          - self.nu*(mc+mp)*g
                denom = mc+mp*torch.sin(theta)**2
                ddot_x = numer/denom
                ddot_theta = (- mp*ddot_x*torch.cos(theta)\
                              + mp*g*torch.sin(theta)\
                              - self.mu*rho*dot_theta**2*(A2/L)*torch.cos(theta))/(mp*l)

            else:
                raise NotImplementedError()
            
            dqdt = torch.FloatTensor([dot_x, dot_theta, ddot_x, ddot_theta])\
                                     .to(device=self.device, dtype=torch.float32)

            self.ddot_x = torch.clamp(ddot_x, -self._max_ddot_x, self._max_ddot_x).reshape(1)
            self.ddot_theta = torch.clamp(ddot_theta, -self._max_ddot_theta, self._max_ddot_theta).reshape(1)
            return dqdt
                    
        # Solve the ODE
        q0 = torch.FloatTensor([self.x, self.theta, self.dot_x, self.dot_theta])\
                                .to(device=self.device, dtype=torch.float32) 
        t = torch.FloatTensor(np.linspace(0, dt, 2))\
                                .to(device=self.device, dtype=torch.float32) 
        q = odeint(func=ode_func, y0=q0, t=t) 

        x, theta, dot_x, dot_theta = q[1]
        self.x = torch.clamp(x, -self._max_x, self._max_x).reshape(1)
        self.theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
        self.dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
        self.dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)

        print(f"x={self.x.item():.4f} \ttheta={self.theta.item():.4f}\
                \tinp_acc={self.inp_acc.item():.4f}\
                \tnu={self.nu.item():.4f} \tmu={self.mu.item():.4f}")

        # print(f"dot_x={self.dot_x.item():.2f}\tdot_theta={self.dot_theta.item():.2f}")

class Environment:
    def __init__(self, cartpole):
        self._cartpole = cartpole
        self._max_nu = 1.
        self._max_mu = 1.

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 2
        self.sensors = len(self.status)

    @property
    def status(self):
        return (self._agent.x,
                self._agent.theta,
                self._agent.dot_x,
                self._agent.dot_theta)

    def update(self, parameters, dt):
        # the environment updates according to the parameters
        nu, mu = parameters
        nu = torch.tanh(nu)*self._max_nu
        mu = torch.tanh(mu)*self._max_mu
        self._cartpole.update(nu=nu, mu=mu, dt=dt)


class Agent:
    def __init__(self, cartpole):
        self._cartpole = cartpole
        self._max_inp_acc = 20.

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
    def ddot_theta(self):
        return self._cartpole.ddot_theta.clone()

    @ddot_theta.setter
    def ddot_theta(self, value):
        self._cartpole.ddot_theta = value

    @property
    def status(self):
        return (self.x,
                self.theta,
                self.dot_x,
                self.dot_theta)

    def update(self, parameters, dt):
        # the action take place and updates the variables
        cart_acceleration = parameters
        cart_acceleration = torch.tanh(cart_acceleration)*self._max_inp_acc
        self._cartpole.update(inp_acc=cart_acceleration, dt=dt)


class Model:
    
    def __init__(self, param_generator, ode_idx, device="cuda"):
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
        print("\n")
        self.agent.x = torch.tensor(cart_position).reshape(1).float()
        self.agent.dot_x = torch.tensor(cart_velocity).reshape(1).float()
        self.agent.theta = torch.tensor(pole_angle).reshape(1).float()
        self.agent.dot_theta = torch.tensor(pole_ang_velocity).reshape(1).float()

        self.nu = torch.tensor(0.0).to(dtype=torch.float32)
        self.mu = torch.tensor(0.0).to(dtype=torch.float32)
        self.inp_acc = torch.tensor(0.0).to(dtype=torch.float32)

        self.traces = {'x': [], 'theta': []}

class RobustnessComputer:
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        x = model.traces['x']
        theta = model.traces['theta']
        return self.dqs.compute(x=torch.cat(x), theta=torch.cat(theta))
        
