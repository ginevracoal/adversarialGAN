import json
import torch
import numpy as np
from diffquantitative import DiffQuantitativeSemantic
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy


class ElMotor:
    def __init__(self):
        self.EM_w_list=np.array([0,95,190,285,380,475,570,665,760,855,950,1045,1140])
        self.EM_T_list=np.array([0,11.25,22.5,33.75,45,56.25,67.5,78.75,90,101.25,112.5,123.75,135,146.25,157.5,168.75,180])
       
        x2d, y2d = np.meshgrid(self.EM_w_list, self.EM_T_list)
       
        self.x2d = x2d
        self.y2d = y2d
       
        self.x_speed_flat = x2d.flatten()
        self.y_torque_flat = y2d.flatten()
       
        self.EM_T_max_list   = np.array([179.1,179,180.05,180,174.76,174.76,165.13,147.78,147.78,109.68,109.68,84.46,84.46])

        self.efficiency = np.array([
        [.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50,.50],
        [.68,.70,.71,.71,.71,.71,.70,.70,.69,.69,.69,.68,.67,.67,.67,.67,.67],
        [.68,.75,.80,.81,.81,.81,.81,.81,.81,.81,.80,.80,.79,.78,.77,.76,.76],
        [.68,.77,.81,.85,.85,.85,.85,.85,.85,.84,.84,.83,.83,.82,.82,.80,.79],
        [.68,.78,.82,.87,.88,.88,.88,.88,.88,.87,.87,.86,.86,.85,.84,.83,.83],
        [.68,.78,.82,.88,.88,.89,.89,.89,.88,.88,.87,.85,.85,.84,.84,.84,.83],
        [.69,.78,.83,.87,.88,.89,.89,.88,.87,.85,.85,.84,.84,.84,.84,.84,.83],
        [.69,.73,.82,.86,.87,.88,.87,.86,.85,.84,.84,.84,.84,.84,.84,.84,.83],
        [.69,.71,.80,.83,.85,.86,.85,.85,.84,.84,.84,.84,.84,.84,.83,.83,.83],
        [.69,.69,.79,.82,.84,.84,.84,.84,.83,.83,.83,.83,.83,.83,.83,.82,.82],
        [.69,.68,.75,.81,.82,.81,.81,.81,.81,.81,.81,.80,.80,.80,.80,.80,.80],
        [.69,.68,.73,.80,.81,.80,.76,.76,.76,.76,.76,.76,.76,.76,.75,.75,.75],
        [.69,.68,.71,.75,.75,.75,.75,.75,.75,.75,.75,.75,.74,.74,.74,.74,.74] ]).T
       
        self.efficiency[self.EM_T_list[:,np.newaxis] > self.EM_T_max_list] = np.nan
       
        self.efficiency_flat = self.efficiency.flatten()
       
    def getEfficiency(self, speed, torque): 
        points = (self.x_speed_flat, self.y_torque_flat)
        pair = (speed.item(), torque.item())
        grid = griddata(points, self.efficiency_flat, pair, method = "cubic")
        # todo: debug
        # print(grid)
        return grid
   
    def getMinMaxTorque(self, speed):
        max_tq = numpy.interp(speed.cpu().detach().numpy(), self.EM_w_list, self.EM_T_max_list)
        return -max_tq[0], max_tq[0]

    def plotEffMap(self):

        fig = plt.figure()
        ax1 = plt.contourf(self.x2d, self.y2d, self.efficiency, cmap = 'jet')
        plt.colorbar(ax1)
        plt.show()


class Car:
    """ Describes the physical behaviour of the vehicle """
    def __init__(self, device):
        self.device=device

        self._max_acceleration = 3.0   #m/s^2
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = 20.0 #m/s
        self._min_velocity = 0.0
        self.gravity = 9.81 #m/s^2
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = 0.01 # will be ignored

        self.mass = 800       #kg
        self.rho =  1.22          #the air density, 
        self.aer_coeff = 0.4     #the aerodynamic coefficient
        self.veh_surface  = 2  #equivalent vehicle surface
        self.rr_coeff =  8*10^-3     #rolling resistance coefficient
        self.gear_ratio = 10
        self.wheel_radius = 0.3  #effective wheel radius
        self._max_whl_brk_torque = 2000  #Nm
        
        self.e_motor = ElMotor()
        
        self.max_e_tq = np.max(self.e_motor.EM_T_max_list)
        self.min_e_tq = - self.max_e_tq
        self.e_motor_speed = torch.tensor(0.0)
        self.e_torque= torch.tensor(0.0)
        self.br_torque= torch.tensor(0.0)
        self.e_power = torch.tensor(0.0)

    def motor_efficiency(self):
        eff = self.e_motor.getEfficiency(self.e_motor_speed,self.e_torque)
        return eff**(-torch.sign(self.e_torque))
    
    def calculate_wheels_torque(self, e_torque, br_torque):
        self.br_torque = torch.clamp(br_torque, 0, self._max_whl_brk_torque)
        self.e_torque = torch.clamp(e_torque, self.min_e_tq, self.max_e_tq)
        return self.e_torque*self.gear_ratio - self.br_torque

    def resistance_force(self):
        F_loss = 0.5*self.rho*self.veh_surface*self.aer_coeff*(self.velocity**2) + \
            self.rr_coeff*self.mass*self.gravity*self.velocity
        return F_loss


    def update(self, dt, e_torque, br_torque, dist_force=0):
        #Differential equation for updating the state of the car

        in_wheels_torque = self.calculate_wheels_torque(e_torque, br_torque)

        acceleration = (in_wheels_torque/self.wheel_radius - self.resistance_force() + dist_force) / self.mass
           
        self.acceleration = torch.clamp(acceleration, self._min_acceleration, self._max_acceleration)
        
        # self.velocity = torch.clamp(self.velocity + self.acceleration * dt, self._min_velocity, self._max_velocity)
        self.velocity = self.velocity + self.acceleration * dt
        self.e_motor_speed = self.velocity*self.gear_ratio/self.wheel_radius
        
        #update min/max e-torque based on new motor speed
        self.min_e_tq, self.max_e_tq = self.e_motor.getMinMaxTorque(self.e_motor_speed)
        # update power consumed
        self.e_power = self.e_motor_speed*self.e_torque*self.motor_efficiency().item()        
        self.position += self.velocity * dt

        print(f"pos={self.position.item()}\tpower={self.e_power.item()}")

class Environment:
    def __init__(self, device):
        self._leader_car = Car(device)

    def set_agent(self, agent):
        self._agent = agent
        self.initialized()

    def initialized(self):
        self.actuators = 2
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
        """ Representation of the state """
        return (self.l_velocity,
                self._agent.velocity,
                self._agent.distance)

    def update(self, parameters, dt):
        """ Updates the physical state with the parameters
            generated by the NN.
        """
        e_torque, br_torque = parameters
        self._leader_car.update(e_torque=e_torque, br_torque=br_torque, dt=dt)


class Agent:
    def __init__(self, device):
        self._car = Car(device)

    def set_environment(self, environment):
        self._environment = environment
        self.initialized()

    def initialized(self):
        self.actuators = 2
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
    def e_power(self):
        return self._car.e_power.clone()

    @e_power.setter
    def e_power(self, value):
        self._car.e_power = value

    @property
    def distance(self):
        return self._environment.l_position - self._car.position

    @property
    def status(self):
        """ Representation of the state """
        return (self._environment.l_velocity,
                self.velocity,
                self.distance, 
                self.e_power)

    def update(self, parameters, dt):
        """ Updates the physical state with the parameters
            generated by the NN.
        """
        e_torque, br_torque = parameters
        self._car.update(e_torque=e_torque, br_torque=br_torque, dt=dt)

class Model:
    """ The model of the whole world.
        It includes both the attacker and the defender.
    """

    def __init__(self, param_generator, device="cuda"):
        self.agent = Agent(device)
        self.environment = Environment(device)

        self.agent.set_environment(self.environment)
        self.environment.set_agent(self.agent)

        self._param_generator = param_generator

        self.traces = None

    def step(self, env_input, agent_input, dt):
        """ Updates the physical world with the evolution of
            a single instant of time.
        """
        self.environment.update(env_input, dt)
        self.agent.update(agent_input, dt)

        self.traces['dist'].append(self.agent.distance)
        self.traces['e_power'].append(self.agent.e_power)

    def initialize_random(self):
        """ Sample a random initial state """
        agent_position, agent_velocity, leader_position, leader_velocity = next(self._param_generator)

        self._last_init = (agent_position, agent_velocity, leader_position, leader_velocity)

        self.reinitialize(agent_position, agent_velocity, leader_position, leader_velocity)

    def initialize_rewind(self):
        """ Restore the world's state to the last initialization """
        self.reinitialize(*self._last_init)

    def reinitialize(self, agent_position, agent_velocity, leader_position, leader_velocity):
        """ Sets the world's state as specified """
        self.agent.position = torch.tensor(agent_position).reshape(1)
        self.agent.velocity = torch.tensor(agent_velocity).reshape(1)
        self.environment.l_position = torch.tensor(leader_position).reshape(1)
        self.environment.l_velocity = torch.tensor(leader_velocity).reshape(1)

        self.traces = {
            'dist': [],
            'e_power': []
        }

class RobustnessComputer:
    """ Used to compute the robustness value (rho) """
    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def compute(self, model):
        """ Computes rho for the given trace """
        d = model.traces['dist']
        e_power = model.traces['e_power'][-1].item()

        return self.dqs.compute(dist=torch.cat(d))-e_power 
