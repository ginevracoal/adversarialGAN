import json
import torch
import numpy as np
import random

from diffquantitative import DiffQuantitativeSemantic

class CartPole():

    def __init__(self, position_limit=40, angle_limit_radians=3.1415/2):
        self._input_cart_acc = None
        self._input_pole_acc = None

        self.gravity = 9.8 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1.0 # cart mass in kg
        self.mpole = 0.1 # pole mass in kg
        self.lpole = 0.5 # half the pole length in meters
        
        self.position_limit = position_limit
        self.angle_limit_radians = angle_limit_radians

        self.x = torch.tensor(random.uniform(-0.5 * self.position_limit, 0.5 * self.position_limit))
        self.theta = torch.tensor(random.uniform(-0.5 * self.angle_limit_radians, 0.5 * self.angle_limit_radians))

        self.v_x = torch.tensor(random.uniform(-1.0, 1.0))
        self.v_theta = torch.tensor(random.uniform(-1.0, 1.0))
        
        self.a_x = torch.tensor(random.uniform(-1.0, 1.0))
        self.a_theta = torch.tensor(random.uniform(-1.0, 1.0))

    def step(self, cart_acc, pole_acc, dt):
        """
        Update the system state.
        """        
        # Locals for readability.
        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        mt = mp + mc
        L = self.lpole

        force = cart_acc * mt

        # Remember acceleration from previous step.
        a_theta0, a_x0  = self.a_theta, self.a_x

        # Update position/angle.
        self.x += dt * self.v_x + 0.5 * a_x0 * dt**2
        self.theta += dt * self.v_theta + 0.5 * a_x0 * dt**2
        # print(self.x.item(), self.theta.item())

        # Compute new accelerations as given in "Correct equations for the dynamics of the cart-pole system"
        # by Razvan V. Florian (http://florian.io).
        # http://coneural.org/florian/papers/05_cart_pole.pdf
        st = torch.sin(self.theta)
        ct = torch.cos(self.theta)
        a_theta = (g * st + ct * (-force - mp * L * self.v_theta**2 * st) / mt) / (L * (4./3. - mp * ct**2 / mt))
        a_x = (force + mp * L * (self.v_theta**2 * st - a_theta * ct)) / mt

        # Update velocities.
        self.v_x += 0.5 * (a_x0 + a_x) * dt
        self.v_theta += 0.5 * (a_theta0 + a_theta) * dt

        # Remember current acceleration for next step.
        self.a_theta, self.a_x = a_theta, a_x

    def update(self, dt, cart_acceleration=None, pole_acceleration=None):
        if cart_acceleration is not None:
            self._input_cart_acc = cart_acceleration

        if pole_acceleration is not None:
            self._input_pole_acc = pole_acceleration

        if self._input_cart_acc is not None and self._input_pole_acc is not None:
            self.step(self._input_cart_acc, self._input_pole_acc, dt)

            self._input_cart_acc = None
            self._input_pole_acc = None


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
        return (self._agent.c_position,
                self._agent.c_velocity,
                self._agent.p_angle,
                self._agent.p_velocity)

    def update(self, parameters, dt):
        # the environment updates according to the parameters
        acceleration = parameters
        self._cartpole.update(dt, pole_acceleration=acceleration)


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
    def c_position(self):
        return self._cartpole.x.clone()

    @c_position.setter
    def c_position(self, value):
        self._cartpole.x = value

    @property
    def c_velocity(self):
        return self._cartpole.v_x.clone()

    @c_velocity.setter
    def c_velocity(self, value):
        self._cartpole.v_x = value

    @property
    def p_angle(self):
        return self._cartpole.theta.clone()

    @p_angle.setter
    def p_angle(self, value):
        self._cartpole.theta = value

    @property
    def p_velocity(self):
        return self._cartpole.v_theta.clone()

    @p_velocity.setter
    def p_velocity(self, value):
        self._cartpole.v_theta = value

    @property
    def status(self):
        return (self.c_position,
                self.c_velocity,
                self.p_angle,
                self.p_velocity)

    def update(self, parameters, dt):
        # the action take place and updates the variables
        acceleration = parameters
        self._cartpole.update(dt, cart_acceleration=acceleration)


class Model:
    
    def __init__(self, param_generator):
        # setting of the initial conditions

        cartpole = CartPole()

        self.agent = Agent(cartpole)
        self.environment = Environment(cartpole)

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

        self._param_generator = param_generator

    def step(self, env_input, agent_input, dt):
        self.environment.update(env_input, dt)
        self.agent.update(agent_input, dt)

        self.traces['pos'].append(self.agent.c_position)
        self.traces['theta'].append(self.agent.p_angle)

    def initialize_random(self):
        cart_position, cart_velocity, pole_position, pole_velocity = next(self._param_generator)

        self._last_init = (cart_position, cart_velocity, pole_position, pole_velocity)

        self.reinitialize(cart_position, cart_velocity, pole_position, pole_velocity)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, cart_position, cart_velocity, pole_position, pole_velocity):
        self.agent.c_position = torch.tensor(cart_position).reshape(1)
        self.agent.c_velocity = torch.tensor(cart_velocity).reshape(1)
        self.agent.p_angle = torch.tensor(pole_position).reshape(1)
        self.agent.p_velocity = torch.tensor(pole_velocity).reshape(1)

        self.traces = {
            'pos': [],
            'theta': []
        }

class RobustnessComputer:
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        pos = model.traces['pos']
        theta = model.traces['theta']

        return self.dqs.compute(pos=torch.cat(pos), theta=torch.cat(theta))
        
