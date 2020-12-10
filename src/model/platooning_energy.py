import json
import torch
import numpy
import numpy as np
from scipy.interpolate import griddata, interpolate 
import matplotlib.pyplot as plt
from utils.diffquantitative import DiffQuantitativeSemantic

from utils.Linear_NNs import LinearModel
import os

DEBUG=True
K=10
ALPHA=0.9
USE_TORCH_EFF_MAP = True

torch.autograd.set_detect_anomaly(True)

class ElMotor():
    
    def __init__(self):
        
        self.max_speed = 1140
        self.max_torque = 180
        
        self.tq_margin = 5
        
        self.EM_w_list=np.array([0,95,190,285,380,475,570,665,760,855,950,1045,1140])
        self.EM_T_list=np.array([0,11.25,22.5,33.75,45,56.25,67.5,78.75,90,101.25,112.5,123.75,135,146.25,157.5,168.75,180])
       
        x2d, y2d = np.meshgrid(self.EM_w_list, self.EM_T_list)
       
        self.x2d = x2d
        self.y2d = y2d
       
        self.x_speed_flat = x2d.flatten()
        self.y_torque_flat = y2d.flatten()
       
        #self.EM_T_max_list   = np.array([179.1,179,180.05,180,174.76,174.76,165.13,147.78,147.78,109.68,109.68,84.46,84.46])
        self.EM_T_max_list   = np.array([180,180,180,180,174.76,170,165.13,150,137.78,115.68,105.68,94.46,84.46])
        
        self.f_max_rq = interpolate.interp1d(self.EM_w_list, self.EM_T_max_list, kind =  "cubic", fill_value="extrapolate")

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
       
        self.efficiency_flat = self.efficiency.flatten()
        self.get_eff_matrix()
       
    def getEfficiency(self, speed, torque): 
        if torch.is_tensor(torque):
            torque = torque.item()
        if torch.is_tensor(speed):
            speed = speed.item()
        
        points = (self.x_speed_flat, self.y_torque_flat)
        pair = (np.abs(speed), np.abs(torque))
        grid = griddata(points, self.efficiency_flat, pair, method = "cubic")
        # todo: debug
        grid[np.abs(torque) > self.f_max_rq(np.abs(speed)) + self.tq_margin ] = np.nan
        
        # print(grid)
        return grid
   
    def getMaxTorque(self, speed):
        #max_tq = numpy.interp(np.abs(speed.cpu().detach().numpy()), self.EM_w_list, self.EM_T_max_list)
        max_tq = self.f_max_rq(np.abs(speed.cpu().detach().numpy()))
        if isinstance(max_tq, np.ndarray) and max_tq.shape:
            return torch.tensor(max_tq[0])
        else:
            return torch.tensor(max_tq)

    def get_eff_matrix(self):
        
        self.speed_vect = np.linspace(0,self.max_speed,201)
        self.torque_vect = np.linspace(0,self.max_torque,151)
        xx, yy = np.meshgrid(self.speed_vect, self.torque_vect)

        self.eff_matrix = self.getEfficiency(xx, yy) #.reshape((emot.speed_vect.shape[0],emot.torque_vect.shape[0]))
                
        self.eff_matrix[yy >  self.f_max_rq(xx) ] = np.nan

    def plotEffMap(self, scatter_array = None):
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_xlim([0,self.max_speed])
        ax1.set_ylim([0,self.max_torque])
       
        levels = np.linspace(0.5, 0.9, 25)
        
        ax1 = plt.contourf(self.speed_vect, self.torque_vect, self.eff_matrix,levels = levels ,cmap = 'jet')

        if scatter_array is not None:
            plt.scatter(scatter_array[:,0],scatter_array[:,1])

        plt.plot(self.speed_vect, self.f_max_rq(self.speed_vect) , 'k')
        #plt.plot(self.EM_w_list,self.EM_T_max_list , 'k')
        cbar =plt.colorbar(ax1)
        cbar.ax.locator_params(nbins=5)
        #plt.show()
        return fig1

    def save_tq_limit(self):
        torque_limit = torch.tensor(np.concatenate( (self.speed_vect[:,np.newaxis], self.f_max_rq(self.speed_vect)[:,np.newaxis]), axis = 1 ))
        
        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'torque_limit.pt')
        torch.save(torque_limit, file_name)


class Car():
    """ Describes the physical behaviour of the vehicle """
    def __init__(self, device, initial_speed = 0.0):
        self.device=device

        self.gravity = 9.81 #m/s^2
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(initial_speed)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = 0.01 # will be ignored

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

        self._max_acceleration = 3.0   #m/s^2
        self._min_acceleration = -self._max_acceleration
        self._max_velocity =  0.95 * self.e_motor.max_speed / self.gear_ratio * self.wheel_radius
        self._min_velocity = 0.0

        self.max_br_tq = 2000
        self.max_e_tq = torch.max(torch.tensor(self.e_motor.max_torque))
        self.min_e_tq = - self.max_e_tq
        self.e_motor_speed = torch.tensor(0.0)
        self.e_torque= torch.tensor(0.0)
        self.br_torque= torch.tensor(0.0)
        self.e_power = torch.tensor(0.0)
        self.timestep_power = torch.tensor(0.0)
        self.eff = torch.tensor(0.0)

    def motor_efficiency(self, e_motor_speed, e_torque):

        if not e_torque.shape:
            eff = self.e_motor.getEfficiency(e_motor_speed, e_torque.unsqueeze(0))
        else:
            eff = self.e_motor.getEfficiency(e_motor_speed, e_torque)
        if not torch.is_tensor(eff):
            #print('eff not tensor')
            eff = torch.tensor(eff).to(self.device)
        # self.eff = eff
        # print(self.eff)
        return eff.squeeze(0)
    
    def calculate_wheels_torque(self, e_torque, br_torque):
        self.e_torque = torch.clamp(e_torque*self.max_e_tq, self.min_e_tq, self.max_e_tq)
        self.br_torque = torch.clamp(br_torque*self.max_br_tq, 0, self._max_whl_brk_torque)
        return self.e_torque*self.gear_ratio - self.br_torque

    def resistance_force(self, old_velocity):
        F_loss = 0.5*self.rho*self.veh_surface*self.aer_coeff*(old_velocity**2) + \
                    self.rr_coeff*self.mass*self.gravity*old_velocity
        return F_loss

    def update(self, dt, norm_e_torque, norm_br_torque, dist_force=0):
        """ Differential equations for updating the state of the car
        """

        in_wheels_torque = self.calculate_wheels_torque(norm_e_torque, norm_br_torque)

        resistance_force = self.resistance_force(self.velocity.clone())
        acceleration = (in_wheels_torque/self.wheel_radius - resistance_force + dist_force) / self.mass
           
        self.acceleration = acceleration #torch.clamp(acceleration, self._min_acceleration, self._max_acceleration)
        # self.velocity = torch.clamp(self.velocity + self.acceleration * dt, 0, self._max_velocity)
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        e_motor_speed = (self.velocity*self.gear_ratio/self.wheel_radius)
        
        eff = self.motor_efficiency(e_motor_speed, self.e_torque)
        effective_efficiency = eff**(-torch.sign(self.e_torque)).to(self.device)
        e_power = e_motor_speed*self.e_torque*effective_efficiency

        self.eff = eff
        self.timestep_power = e_power-self.e_power
        self.e_power = e_power

        #update min/max e-torque based on new motor speed
        self.max_e_tq = self.e_motor.getMaxTorque(self.e_motor_speed)
        self.min_e_tq = -self.max_e_tq
        # print(f"pos={self.position.item()}\tpower={self.e_power.item()}")
        # print(self.timestep_power.item())


class ElMotor_torch():
    
    def __init__(self, device,net_name, path_log = os.path.abspath(os.path.dirname(__file__))):
        self.device = device
        self.net = LinearModel('LinearModel',0.0002, 1, 2 )
        self.net.load_net_params( path_log, net_name, self.device)
        
        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'torque_limit.pt')
        self.tq_limit = torch.load(file_name)
        
        self.max_speed = 1140
        self.max_torque = 180

    def getEfficiency(self, speed, torque):
        data = torch.stack(( speed/self.max_speed , torque/self.max_torque ) , dim = 1).to(self.device)
        with torch.no_grad():
            eff = self.net(data)
        
        return eff
        
    def getMaxTorque(self, speed):
        idx = torch.argmax(self.tq_limit[:,0]-speed)
        return self.tq_limit[idx,1]
    
    def plotEffMap(self):
        
        speed_vect = np.linspace(0,self.max_speed,201)
        torque_vect = np.linspace(0,self.max_torque,151)        
        
        xx,yy = np.meshgrid(speed_vect, torque_vect)
        xx_norm = xx/self.max_speed
        yy_norm = yy/self.max_torque
        
        test_data = torch.stack((torch.tensor(xx_norm),torch.tensor(yy_norm)),dim = 2).to(self.device)
        with torch.no_grad():
            test_y = self.net(test_data.float()).squeeze(2).cpu()
        
        test_y[torch.tensor(yy) >  self.tq_limit[:,1].unsqueeze(0) ] = np.nan
        test_y = np.minimum(0.9,test_y.detach().numpy())
        
        fig1 = plt.figure()
        
        ax1 = fig1.add_subplot(111)
        ax1.set_xlim([0,self.max_speed])
        ax1.set_ylim([0,self.max_torque])
        
        levels = np.linspace(0.5, 0.9, 28)
        ax1 = plt.contourf(speed_vect, torque_vect, test_y, levels=levels ,cmap = 'jet')
        plt.plot(self.tq_limit[:,0],self.tq_limit[:,1] , 'k')
        

        plt.colorbar(ax1)
        plt.show()
        
        return fig1


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
        self._leader_car.update(norm_e_torque=e_torque, norm_br_torque=br_torque, dt=dt)


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
    def timestep_power(self):
        return self._car.timestep_power.clone()

    @timestep_power.setter
    def timestep_power(self, value):
        self._car.timestep_power = value

    @property
    def e_power(self):
        return self._car.e_power.clone()

    @timestep_power.setter
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
                self.distance)

    def update(self, parameters, dt):
        """ Updates the physical state with the parameters
            generated by the NN.
        """
        e_torque, br_torque = parameters
        self._car.update(norm_e_torque=e_torque, norm_br_torque=br_torque, dt=dt)

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
        self.traces['power'].append(self.agent.timestep_power)

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
        """ Computes rho for the given trace """
        dist = model.traces['dist'][-K:]
        power = model.traces['power'][-K:]
        rob_dist = self.dqs_dist.compute(dist=torch.cat(dist))
        rob_power = self.dqs_power.compute(power=torch.cat(power))
        return ALPHA*rob_dist+(1-ALPHA)*rob_power