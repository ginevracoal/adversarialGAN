import os
import json
import numpy
import numpy as np
from scipy.interpolate import griddata, interpolate 
import matplotlib.pyplot as plt
import torch

from utils.Linear_NNs import LinearModel


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
        
        cmap = plt.cm.get_cmap('Spectral')
        fig1 = plt.figure(figsize=(6, 5))
        fig1.tight_layout(pad=3.0)

        ax1 = fig1.add_subplot(111)
        ax1.set_xlim([0,self.max_speed+2])
        ax1.set_ylim([0,self.max_torque+2])

        print("max_speed = ", self.max_speed, " max_torque = ", self.max_torque)
 
        ax1.set_xlabel(r'$n_m (rpm)$')
        ax1.set_ylabel(r'$T_m (N\,m)$')
       
        levels = np.linspace(0.5, 0.9, 25)
        
        ax1 = plt.contourf(self.speed_vect, self.torque_vect, self.eff_matrix,levels = levels, cmap=cmap)

        if scatter_array is not None:
            plt.scatter(scatter_array[:,0],scatter_array[:,1])

        plt.plot(self.speed_vect, self.f_max_rq(self.speed_vect) , 'k')
        #plt.plot(self.EM_w_list,self.EM_T_max_list , 'k')

        fig1.subplots_adjust(right=0.83)
        cbar_ax = fig1.add_axes([0.88, 0.12, 0.03, 0.75])
        cbar =plt.colorbar(ax1, cax=cbar_ax)
        cbar.ax.locator_params(nbins=5)
        #plt.show()
        return fig1

    def save_tq_limit(self):
        torque_limit = torch.tensor(np.concatenate( (self.speed_vect[:,np.newaxis], self.f_max_rq(self.speed_vect)[:,np.newaxis]), axis = 1 ))
        
        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'torque_limit.pt')
        torch.save(torque_limit, file_name)


class ElMotor_torch():
    
    def __init__(self, device,net_name, path_log = os.path.abspath(os.path.dirname(__file__))):
        self.device = device
        self.net = LinearModel('LinearModel', 0.0002, 1, 2)
        self.net.load_net_params(path_log, net_name, self.device)
        for params in self.net.parameters():
            params.requires_grad=False
        
        file_name = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'torque_limit.pt')
        self.tq_limit = torch.load(file_name)
        
        self.max_speed = 1140
        self.max_torque = 180

    def getEfficiency(self, speed, torque):
        data = torch.stack((speed/self.max_speed, torque/self.max_torque), dim=1)
        # with torch.no_grad():
        eff = self.net(data)[0]
        return eff
        
    def getMaxTorque(self, speed):
        idx = torch.argmax(self.tq_limit[:,0]-speed)
        return self.tq_limit[idx,1].item()
    
    def plotEffMap(self):
             
        speed_vect = np.linspace(0, self.max_speed,201)
        torque_vect = np.linspace(0, self.max_torque,151)        
         
        xx,yy = np.meshgrid(speed_vect, torque_vect)
        xx_norm = xx/self.max_speed
        yy_norm = yy/self.max_torque
        
        test_data = torch.stack((torch.tensor(xx_norm), torch.tensor(yy_norm)), dim = 2)
        with torch.no_grad():
            test_y = self.net(test_data.float()).squeeze(2).cpu()
        
        test_y[torch.tensor(yy) >  self.tq_limit[:,1].unsqueeze(0)]=np.nan
        test_y = np.minimum(0.9, test_y.detach().numpy())
        
        fig1 = plt.figure()
        
        ax1 = fig1.add_subplot(111)
        ax1.set_xlim([0, self.max_speed])
        ax1.set_ylim([0, self.max_torque])
        
        levels = np.linspace(0.5, 0.9, 28)
        ax1 = plt.contourf(speed_vect, torque_vect, test_y, levels=levels, cmap='jet')
        plt.plot(self.tq_limit[:,0], self.tq_limit[:,1], 'k')
        
        plt.colorbar(ax1)
        plt.show()
        
        return fig1
