from abc import ABC, abstractmethod

class Environment(ABC):

    def set_agent(self, agent):
        self._agent = agent

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self, parameters):
        """ The enviroment applies the adversarial conditions.
        """
        pass

class Agent(ABC):

    def set_environment(self, environment):
        self._environment = environment

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def update(self):
        """ The agent reacts to the conditions of the environment.
        """
        pass
