import json
import torch
from diffquantitative import DiffQuantitativeSemantic

DEBUG=False

class CartPole():

    def __init__(self):

        self.gravity = 9.81 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1. # cart mass in kg
        self.mpole = .1 # pole mass in kg
        self.lpole = 1. # pole length in meters

        self.x, self.theta = (torch.tensor(0.0), torch.tensor(0.0))
        self.dot_theta, self.dot_x = (torch.tensor(0.0), torch.tensor(0.0))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.0), torch.tensor(0.))

        self._max_x = 5. 
        self._max_theta = 1.5 
        self._max_dot_x = 10
        self._max_dot_theta = 10

        self.actuators = 1
        self.sensors = len(self.status)

    def update(self, action, dt):
        """
        Update the system state.
        """        
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        l = self.lpole/2 
        f = action

        temp = (f+mp*l*self.dot_theta**2*torch.sin(self.theta))/(mp+mc)
        denom = l*(4/3-(mp*torch.cos(self.theta)**2)/(mp+mc))
        ddot_theta = (g*torch.sin(self.theta)-torch.cos(self.theta)*temp)/denom
        ddot_x = temp - (mp*l*ddot_theta*torch.cos(self.theta))/(mp+mc)

        dot_x = self.dot_x + dt * ddot_x
        dot_theta = self.dot_theta + dt * ddot_theta
        x = self.x + dt * dot_x
        theta = self.theta + dt * dot_theta

        self.x = torch.clamp(x, -self._max_x, self._max_x).reshape(1)
        self.theta = torch.clamp(theta, -self._max_theta, self._max_theta).reshape(1)
        self.dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x).reshape(1)
        self.dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta).reshape(1)

        if DEBUG:
            print(f"x={self.x.item():.4f}  theta={self.theta.item():.4f}  f={f.item()}")

    @property
    def status(self):
        return (self.x,
                self.dot_x,
                self.theta,
                self.dot_theta)


class Model:
    
    def __init__(self, param_generator):
        # setting of the initial conditions
        self.cartpole = CartPole()
        self._param_generator = param_generator
        self.traces = None

    def step(self, action, dt):

        self.cartpole.update(action, dt)

        self.traces['theta'].append(self.cartpole.theta)

    def initialize_random(self):
        cart_position, cart_velocity, pole_angle, pole_ang_velocity = next(self._param_generator)

        self._last_init = (cart_position, cart_velocity, pole_angle, pole_ang_velocity)

        self.reinitialize(cart_position, cart_velocity, pole_angle, pole_ang_velocity)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, cart_position, cart_velocity, pole_angle, pole_ang_velocity):

        self.cartpole.x = torch.tensor(cart_position).reshape(1)
        self.cartpole.dot_x = torch.tensor(cart_velocity).reshape(1)
        self.cartpole.theta = torch.tensor(pole_angle).reshape(1)
        self.cartpole.dot_theta = torch.tensor(pole_ang_velocity).reshape(1)
        self.traces = {'theta': []}

class RobustnessComputer:
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        theta = model.traces['theta'][-10:]
        rho = self.dqs.compute(theta=torch.cat(theta))
        return rho
        
