import json
import torch
import numpy as np

from diffquantitative import DiffQuantitativeSemantic

def ranged(x, max_x, min_x=None):
    if min_x is not None:
        return torch.min(torch.max(x, torch.tensor(min_x)), torch.tensor(max_x))
    else:
        return torch.min(x, torch.tensor(max_x))

class Car:
    def __init__(self):
        self._max_acceleration = 10.0
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = 150.0
        self._min_velocity = 0.0
        self.mass = 1.0
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = 0.1

    def update(self, in_acceleration, dt):
        self.acceleration = ranged(in_acceleration, self._max_acceleration, self._min_acceleration)
        if self.velocity > 0:
            self.acceleration -= self.friction_coefficient * self.mass
        self.velocity += ranged(self.acceleration * dt, self._max_velocity, self._min_velocity)
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
    
    def __init__(self):
        # setting of the initial conditions

        self.agent = Agent()
        self.environment = Environment()

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

    def step(self, env_input, agent_input, dt):
        self.environment.update(env_input, dt)
        self.agent.update(agent_input, dt)

        self.traces['dist'].append(self.agent.distance)

    def initialize_random(self):
        agent_position = np.random.rand(1) * 25
        agent_velocity = np.random.rand(1) * 20
        leader_position = 28 + np.random.rand(1) * 20
        leader_velocity = np.random.rand(1) * 20

        self.reinitialize(agent_position, agent_velocity, leader_position, leader_velocity)

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
        
