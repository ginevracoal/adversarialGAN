import json
import torch
from utils.diffquantitative import DiffQuantitativeSemantic


DEBUG=False
FRICTION=True
ALPHA=0.4
K=10

class CartPole():

    def __init__(self):

        self.gravity = 9.81 # acceleration due to gravity, positive is downward, m/sec^2
        self.mcart = 1. # cart mass in kg
        self.mpole = .1 # pole mass in kg
        self.lpole = 1. # pole length in meters

        self.x, self.theta = (torch.tensor(0.0), torch.tensor(0.0))
        self.dot_theta, self.dot_x = (torch.tensor(0.0), torch.tensor(0.0))
        self.ddot_x, self.ddot_theta = (torch.tensor(0.0), torch.tensor(0.))
        self.x_target = torch.tensor(0.0)
        self.dist = torch.tensor(0.0)
        self.mu = torch.tensor(0.0)

        self._max_x = 30.
        self._max_theta = 1.5
        self._max_dot_x = 10
        self._max_dot_theta = 10
        self._max_dot_eps=5.
        self._max_mu=1.
        self._max_action=30

    def update(self, dt, action, mu, dot_eps):    

        g = self.gravity
        mp = self.mpole
        mc = self.mcart
        l = self.lpole/2
            
        f = torch.clamp(action, -self._max_action, self._max_action)

        # update_mu = np.random.binomial(n=1, p=0.1)
        mu = torch.clamp(mu, 0., self._max_mu) #if update_mu==1 else None

        dot_eps = torch.clamp(dot_eps, -self._max_dot_eps, self._max_dot_eps)
        eps = dot_eps * dt
        x_target = self.x + eps
        self.x_target = torch.clamp(x_target, -self._max_x, self._max_x)
        self.dist = torch.abs(self.x-self.x_target)
        
        if FRICTION:
            ddot_x = f - mu*self.dot_x  \
                       + mp*l*self.dot_theta**2* torch.sin(self.theta) \
                       - mp*g*torch.cos(self.theta) * torch.sin(self.theta)
            ddot_x = ddot_x / ( mc+mp-mp* torch.cos(self.theta)**2 )
            ddot_theta = (g*torch.sin(self.theta) - torch.cos(self.theta)*ddot_x ) / l
        
        else:
            temp = (f+mp*l*self.dot_theta**2*torch.sin(self.theta))/(mp+mc)
            denom = l*(4/3-(mp*torch.cos(self.theta)**2)/(mp+mc))
            ddot_theta = (g*torch.sin(self.theta)-torch.cos(self.theta)*temp)/denom
            ddot_x = temp - (mp*l*ddot_theta*torch.cos(self.theta))/(mp+mc)
        
        dot_x = self.dot_x + dt * ddot_x
        dot_theta = self.dot_theta + dt * ddot_theta
        x = self.x + dt * dot_x
        theta = self.theta + dt * dot_theta
        
        ## print(f"\neq diff out:\t {ddot_x.item():.4f} {ddot_theta.item():.4f}")
        # print(f"x update:\t{self.x.item():.4f} -> {x.item():.4f}")
        # print(f"theta update:\t{self.theta.item():.4f} -> {theta.item():.4f}\n")

        self.x = torch.clamp(x, -self._max_x, self._max_x)
        self.theta = torch.clamp(theta, -self._max_theta, self._max_theta)
        self.dot_x = torch.clamp(dot_x, -self._max_dot_x, self._max_dot_x)
        self.dot_theta = torch.clamp(dot_theta, -self._max_dot_theta, self._max_dot_theta)
        # print(eps, self.x, self.x_target, self.dist)

        if DEBUG:
            print(f"dist={self.dist.item():.4f}  mu={mu.item():.4f}", end="\t")
            print(f"x={self.x.item():4f}, theta={self.theta.item():4}")


class Environment:
    def __init__(self, cartpole):
        self._cartpole = cartpole

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 2
        self.sensors = len(self.status)

    @property
    def x(self):
        return self._cartpole.x.clone()

    @x.setter
    def x(self, value):
        self._cartpole.x = value

    @property
    def theta(self):
        return self._cartpole.theta.clone()

    @theta.setter
    def theta(self, value):
        self._cartpole.theta = value

    @property
    def x_target(self):
        return self._cartpole.x_target.clone()

    @x_target.setter
    def x_target(self, value):
        self._cartpole.x_target = value

    @property
    def dist(self):
        return self._cartpole.dist.clone()

    @dist.setter
    def dist(self, value):
        self._cartpole.dist = value

    @property
    def status(self):
        return (self._agent.x, self._agent.theta, self._cartpole.dist, self.x_target)


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
    def ddot_x(self):
        return self._cartpole.ddot_x.clone()

    @ddot_x.setter
    def ddot_x(self, value):
        self._cartpole.ddot_x = value

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
    def x_target(self):
        return self._cartpole.x_target.clone()

    @x_target.setter
    def x_target(self, value):
        self._cartpole.x_target = value

    @property
    def status(self):
        return (self.x, self.dot_x, self.theta, self.dot_theta, self.x_target)


class Model:
    
    def __init__(self, param_generator):
        # setting of the initial conditions
        self.cartpole = CartPole()

        self.agent = Agent(self.cartpole)
        self.environment = Environment(self.cartpole)

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

        self._param_generator = param_generator
        self.traces = None

    def step(self, env_input, agent_input, dt):
        dot_eps, mu = env_input
        action = agent_input

        self.cartpole.update(dt=dt, action=action, mu=mu, dot_eps=dot_eps)

        self.traces['dist'].append(self.cartpole.dist)
        self.traces['theta'].append(self.agent.theta)

    def initialize_random(self):
        cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target = next(self._param_generator)

        self._last_init = (cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target)

        self.reinitialize(cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target)

    def initialize_rewind(self):
        self.reinitialize(*self._last_init)

    def reinitialize(self, cart_position, cart_velocity, pole_angle, pole_ang_velocity, x_target):

        self.agent.x = torch.tensor(cart_position).reshape(1)
        self.agent.dot_x = torch.tensor(cart_velocity).reshape(1)
        self.agent.theta = torch.tensor(pole_angle).reshape(1)
        self.agent.dot_theta = torch.tensor(pole_ang_velocity).reshape(1)
        self.agent.x_target = torch.tensor(x_target).reshape(1)

        self.traces = {'theta': [], 'dist':[]}

class RobustnessComputer:
    def __init__(self, formula_theta, formula_dist):
        self.dqs_theta = DiffQuantitativeSemantic(formula_theta)
        self.dqs_dist = DiffQuantitativeSemantic(formula_dist)

    def compute(self, model):
        """ Computes robustness for the given trace """
        theta = model.traces['theta'][-K:]
        dist = model.traces['dist'][-K:]
        rob_theta = self.dqs_theta.compute(theta=torch.cat(theta))
        rob_dist = self.dqs_dist.compute(dist=torch.cat(dist))

        return ALPHA*rob_dist+(1-ALPHA)*rob_theta
