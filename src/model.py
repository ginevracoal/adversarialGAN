import convlogic
import json
import torch
import numpy as np

class Car:
    def __init__(self):
        self.mass = 1.0
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        self.friction_coefficient = 0.1

    def update(self, in_acceleration, dt):
        self.acceleration = in_acceleration
        if self.velocity > 0:
            self.acceleration -= self.friction_coefficient * self.mass
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt


class Environment:
    def __init__(self):
        self._leader_car = Car()
        self._max_acceleration = 10.0

    def set_agent(self, agent):
        self._agent = agent

    @property
    def l_position(self):
        return self._leader_car.position

    @l_position.setter
    def l_position(self, value):
        self._leader_car.position = value

    @property
    def l_velocity(self):
        return self._leader_car.velocity

    @l_velocity.setter
    def l_velocity(self, value):
        self._leader_car.velocity = value

    def get_status(self):
        return np.array([self.l_velocity, self._agent.distance])

    def update(self, parameters, dt):
        # the environment updates according to the parameters
        pedal = float(parameters[0])
        self._leader_car.update(pedal * self._max_acceleration, dt)


class Agent:
    def __init__(self):
        self._car = Car()
        self._max_acceleration = 10.0

    def set_environment(self, environment):
        self._environment = environment

    @property
    def position(self):
        return self._car.position

    @position.setter
    def position(self, value):
        self._car.position = value

    @property
    def velocity(self):
        return self._car.velocity

    @velocity.setter
    def velocity(self, value):
        self._car.velocity = value

    @property
    def distance(self):
        return self._environment.l_position - self._car.position

    def get_status(self):
        return np.array([self.velocity, self.distance])

    def update(self, parameters, dt):
        # the action take place and updates the variables
        pedal = float(parameters[0])
        self._car.update(pedal * self._max_acceleration, dt)


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

        # status of the system, env and agent params should be recorded
        self._time += dt
        self.traces['time'].append(self._time)
        self.traces['dist'].append(self.agent.distance)

        status = (self._time, self.environment.l_position, \
                self.agent.position, self.agent.distance)
        self._records.append(status)

    def initialize_random(self):
        self._time = 0.0
        self.traces = {
            'time': [self._time],
            'dist': [self.agent.distance],
        }
        self._records = []

        self.agent.position = torch.rand(1) * 25
        self.agent.velocity = torch.rand(1) * 20
        self.environment.l_position = 28 + torch.rand(1) * 20
        self.environment.l_velocity = torch.rand(1) * 20

    def get_status(self):
        return (self.environment.l_velocity,
                self.agent.velocity,
                self.agent.distance)

    def get_records(self):
        return np.array(self._records).T


class RobustnessComputer:
    def __init__(self, formula):
        self.qs = convlogic.QuantitativeSemantic(formula)

    def compute(self, model, start_index=0):
        t = np.array(model.traces['time'][start_index:])
        d = np.array(model.traces['dist'][start_index:])

        return self.qs.check(t, dist=d)
        
