import json
import torch

from diffquantitative import DiffQuantitativeSemantic

class Car:
    def __init__(self):
        self._max_acceleration = 5.0
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = 10.0
        self._min_velocity = -self._max_velocity
        self.gravity = 9.81
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = 0.01

    def update(self, in_acceleration, angle, dt):
        self.acceleration = torch.clamp(in_acceleration, self._min_acceleration, self._max_acceleration)
        self.acceleration -= self.gravity * torch.sin(angle)
        if self.velocity != 0:
            self.acceleration -= self.friction_coefficient * self.gravity * torch.cos(angle)
        self.velocity = torch.clamp(self.velocity + self.acceleration * dt, self._min_velocity, self._max_velocity)
        self.position += self.velocity * dt


class Environment:
    def __init__(self):
        self._fn = lambda x: torch.tensor(0.)

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 5
        self.sensors = 0

    @property
    def status(self):
        return ()

    def get_steepness(self, x):
        return torch.clamp(self._fn(x), -0.35, 0.35)

    def update(self, parameters, dt):
        def linear_combination(x):
            basis = [x**i for i in range(len(parameters))]
            basis = torch.tensor(basis, dtype=torch.get_default_dtype())
            return parameters.dot(basis)

        self._fn = linear_combination


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
    def angle(self):
        return self._environment.get_steepness(float(self.position))

    @property
    def status(self):
        return (self.velocity,
                self.angle)

    def update(self, parameters, dt):
        # the action take place and updates the variables
        acceleration = parameters
        self._car.update(acceleration, self.angle, dt)


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

        self.traces['velo'].append(self.agent.velocity)

    def initialize_random(self):
        agent_position, agent_velocity = next(self._param_generator)

        self._last_init = (agent_position, agent_velocity)

        self.reinitialize(agent_position, agent_velocity)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, agent_position, agent_velocity):
        self.agent.position = torch.tensor(agent_position).reshape(1)
        self.agent.velocity = torch.tensor(agent_velocity).reshape(1)
        
        self.traces = {
            'velo': []
        }

class RobustnessComputer:
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        v = model.traces['velo']

        return self.dqs.compute(v=torch.cat(v))
        
