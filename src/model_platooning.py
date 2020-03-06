import json
import torch
import numpy as np

from diffquantitative import DiffQuantitativeSemantic

class Car:
    def __init__(self):
        self._max_acceleration = 3.0
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = 5.0
        self._min_velocity = 0.0
        self.gravity = 9.81
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = 0.01

    def update(self, in_acceleration, dt):
        self.acceleration = torch.clamp(in_acceleration, self._min_acceleration, self._max_acceleration)
        if self.velocity > 0:
            self.acceleration -= self.friction_coefficient * self.gravity
        self.velocity = torch.clamp(self.velocity + self.acceleration * dt, self._min_velocity, self._max_velocity)
        self.position += self.velocity * dt


class Environment:
    def __init__(self):
        self._leader_car = Car()

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 1
        self.sensors = len(self.status)

    @property
    def l_position(self):
        return self._leader_car.position.clone()

    @l_position.setter
    def l_position(self, value):
        self._leader_car.position = value

    @property
    def l_velocity(self):
        return self._leader_car.velocity.clone()

    @l_velocity.setter
    def l_velocity(self, value):
        self._leader_car.velocity = value

    @property
    def status(self):
        return (self.l_velocity,
                self._agent.velocity,
                self._agent.distance)

    def update(self, parameters, dt):
        # the environment updates according to the parameters
        acceleration = parameters[0]
        self._leader_car.update(acceleration, dt)


class Agent:
    def __init__(self):
        self._car = Car()

    def set_environment(self, environment):
        self._environment = environment
        self.initialized()

    def initialized(self):
        self.actuators = 1
        self.sensors = len(self.status)

    @property
    def position(self):
        return self._car.position.clone()

    @position.setter
    def position(self, value):
        self._car.position = value

    @property
    def velocity(self):
        return self._car.velocity.clone()

    @velocity.setter
    def velocity(self, value):
        self._car.velocity = value

    @property
    def distance(self):
        return self._environment.l_position - self._car.position

    @property
    def status(self):
        return (self._environment.l_velocity,
                self.velocity,
                self.distance)

    def update(self, parameters, dt):
        # the action take place and updates the variables
        acceleration = parameters[0]
        self._car.update(acceleration, dt)


class Model:
    
    def __init__(self, param_generator):
        # setting of the initial conditions

        self.agent = Agent()
        self.environment = Environment()

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

        self._param_generator = param_generator

    def step(self, env_input, agent_input, dt):
        self.environment.update(env_input, dt)
        self.agent.update(agent_input, dt)

        self.traces['dist'].append(self.agent.distance)

    def initialize_random(self):
        agent_position, agent_velocity, leader_position, leader_velocity = next(self._param_generator)

        self._last_init = (agent_position, agent_velocity, leader_position, leader_velocity)

        self.reinitialize(agent_position, agent_velocity, leader_position, leader_velocity)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, agent_position, agent_velocity, leader_position, leader_velocity):
        self.agent.position = torch.tensor(agent_position).reshape(1)
        self.agent.velocity = torch.tensor(agent_velocity).reshape(1)
        self.environment.l_position = torch.tensor(leader_position).reshape(1)
        self.environment.l_velocity = torch.tensor(leader_velocity).reshape(1)

        self.traces = {
            'dist': []
        }

class RobustnessComputer:
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        d = model.traces['dist']

        return self.dqs.compute(dist=torch.cat(d))
        
