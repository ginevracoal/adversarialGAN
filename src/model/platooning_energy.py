import os
import torch
from model.electric_motor import ElMotor, ElMotor_torch
from utils.diffquantitative import DiffQuantitativeSemantic

K=10
ALPHA=0.7
USE_TORCH_EFF_MAP=True


class Car():
    """ Describes the physical behaviour of the vehicle """
    def __init__(self, device):
        self.device=device

        self.gravity = 9.81 # m/s^2
        self.mass = 800       #kg
        self.rho =  1.22          #the air density, 
        self.aer_coeff = 0.4     #the aerodynamic coefficient
        self.veh_surface  = 2  #equivalent vehicle surface
        self.rr_coeff =  8e-3     #rolling resistance coefficient
        self.gear_ratio = 10
        self.wheel_radius = 0.3  #effective wheel radius
        self._max_whl_brk_torque = 2000  #Nm

        if USE_TORCH_EFF_MAP:
            self.e_motor = ElMotor_torch(device = self.device, net_name = 'Net_10_15_15_5')
        else:
            self.e_motor = ElMotor() 

        self._max_acceleration = 5.0 
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = 0.95 * self.e_motor.max_speed / self.gear_ratio * self.wheel_radius
        self._min_velocity = 0.0
        self.max_br_tq = 2000
        self.max_e_tq = torch.max(torch.tensor(self.e_motor.max_torque)).item()
        self.min_e_tq = - self.max_e_tq

        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.e_power = torch.tensor(0.0)
        self.timestep_power = torch.tensor(0.0)

    def motor_efficiency(self, e_motor_speed, e_torque):

        if not e_torque.shape:
            eff = self.e_motor.getEfficiency(e_motor_speed, e_torque.unsqueeze(0))
        else:
            eff = self.e_motor.getEfficiency(e_motor_speed, e_torque)
            
        if not torch.is_tensor(eff):
            eff = torch.tensor(eff)

        return eff
    
    def calculate_wheels_torque(self, e_torque, br_torque):
        br_torque = - br_torque*self.max_br_tq
        e_torque = e_torque*self.max_e_tq
        return e_torque, br_torque, e_torque*self.gear_ratio + br_torque

    def resistance_force(self, current_velocity):
        F_loss = 0.5*self.rho*self.veh_surface*self.aer_coeff*(current_velocity**2) + \
                    self.rr_coeff*self.mass*self.gravity*current_velocity
        return F_loss

    def compute_power_consumption(self, e_torque):
    
        e_motor_speed = self.velocity*self.gear_ratio/self.wheel_radius
        eff = self.motor_efficiency(e_motor_speed, e_torque)
        
        reduction_count = 0
        if torch.isnan(eff):
            while torch.isnan(eff):
                e_torque = 0.9*e_torque
                eff = self.motor_efficiency(e_motor_speed, e_torque)
                reduction_count +=1
                if reduction_count > 5:
                    raise('Eff calculation issue')

        if not torch.is_tensor(eff):
            eff = torch.tensor(eff)
        effective_efficiency = eff**(-torch.sign(e_torque))
        
        e_power = e_motor_speed*e_torque*effective_efficiency

        self.timestep_power = torch.abs(e_power-self.e_power)
        self.e_power = e_power      

        # update min/max e-torque based on new motor speed
        self.max_e_tq = self.e_motor.getMaxTorque(e_motor_speed)
        self.min_e_tq = -self.max_e_tq

        return self.e_power

    def update(self, dt, norm_e_torque, norm_br_torque, compute_power=True):
        """ Differential equations for updating the state of the car
        """
        e_torque, _, in_wheels_torque = self.calculate_wheels_torque(norm_e_torque, norm_br_torque)

        resistance_force = self.resistance_force(self.velocity)
        acceleration = (in_wheels_torque/self.wheel_radius - resistance_force) / self.mass
        acceleration = torch.clamp(acceleration, self._min_acceleration, self._max_acceleration)
        self.velocity = torch.clamp(self.velocity + acceleration * dt, self._min_velocity, self._max_velocity)
        self.position = self.position + self.velocity * dt
        
        # if compute_power:
        self.compute_power_consumption(e_torque)



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
        return (self.l_velocity, self._agent.velocity, self._agent.distance)

    def update(self, parameters, dt):
        """ Updates the physical state with the parameters
            generated by the NN.
        """
        norm_e_torque, norm_br_torque = parameters        
        self._leader_car.update(norm_e_torque=norm_e_torque, norm_br_torque=norm_br_torque, dt=dt)


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
    def distance(self):
        return torch.abs(self._environment.l_position - self._car.position).clone()

    @distance.setter    
    def distance(self, value):
        self._car.distance = value

    # @property
    # def timestep_power(self):
    #     return self._car.timestep_power.clone()

    # @timestep_power.setter
    # def timestep_power(self, value):
    #     self._car.timestep_power = value

    @property
    def e_power(self):
        return self._car.e_power.clone()

    @e_power.setter
    def e_power(self, value):
        self._car.e_power = value

    @property
    def status(self):
        """ Representation of the state """
        return (self._environment.l_velocity, self.velocity, self.distance)

    def update(self, parameters, dt):
        """ Updates the physical state with the parameters
            generated by the NN.
        """
        norm_e_torque, norm_br_torque = parameters
        self._car.update(norm_e_torque=norm_e_torque, norm_br_torque=norm_br_torque, dt=dt, compute_power=True)

class Model:
    """ The model of the whole world.
        It includes both the attacker and the defender.
    """

    def __init__(self, param_generator, device="cpu"):        
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
        self.traces['power'].append(self.agent.e_power)

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

        self.traces = {'dist': [], 'power': []}

class RobustnessComputer:
    """ Used to compute the robustness value (rho) """
    def __init__(self, formula_dist, formula_power):
        self.dqs_dist = DiffQuantitativeSemantic(formula_dist)
        self.dqs_power = DiffQuantitativeSemantic(formula_power)

    def compute(self, model):
        """ Computes robustness for the given trace """
        dist = model.traces['dist'][-K:]
        power = model.traces['power'][-K:]

        rob_dist = self.dqs_dist.compute(dist=torch.cat(dist))/10
        rob_power = self.dqs_power.compute(power=torch.cat(power))/4000

        return ALPHA*rob_dist+(1-ALPHA)*rob_power
