#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 10:44:17 2020

@author: rodpod21
"""

from model.platooning_energy import Car
import torch



class Car_classic(Car):

    def __init__(self, device, velocity = torch.tensor(0.0), reference_distance = torch.tensor(40.0), initial_distance = torch.tensor(40.0)):
        super(Car_classic, self).__init__(device)
        
        self.velocity = velocity
        self.tc_lp = 0.6
        self.dist_thds = [12,15]
        self.K_ctrl = [.15, .001, .01]
        self.cum_error = self.previous_def_e_tq = torch.tensor(0)
        
        self.ref_distance = reference_distance 
        self.previous_distance = initial_distance
        
        self.max_norm_val = 0.999

    def update(self, dt, current_distance): # norm_e_torque, norm_br_torque not used!

        classic_norm_e_torque, classic_norm_br_torque = self.get_controller_input(current_distance)
        super().update(dt, classic_norm_e_torque, classic_norm_br_torque)
        
        return classic_norm_e_torque, classic_norm_br_torque
        
    def get_controller_input(self, current_distance):
        
        error = self.ref_distance - current_distance
        self.cum_error = self.cum_error + error
        
        vel_error = (self.previous_distance - current_distance ) / dt 
        
        ctrl_law =  ( self.K_ctrl[0]*error +self.K_ctrl[1]* self.cum_error \
                      + self.K_ctrl[2]*vel_error )
        
        norm_e_torque = (1-self.tc_lp)*ctrl_law + \
            self.tc_lp*self.previous_def_e_tq
        
        norm_e_torque = torch.clamp(-norm_e_torque, -self.max_norm_val , self.max_norm_val)
       
        norm_br_torque = 0
        if error > self.dist_thds[1] and vel_error > 0:
            norm_br_torque = self.max_norm_val
            norm_e_torque = -self.max_norm_val
        elif error > self.dist_thds[0] and vel_error > 0:
            norm_br_torque = (error - self.dist_thds[0]) / \
                (self.dist_thds[1]- self.dist_thds[0])
            norm_e_torque = -self.max_norm_val
 
        if not torch.is_tensor(norm_e_torque):
            norm_e_torque = torch.tensor(norm_e_torque)
            
        if not torch.is_tensor(norm_br_torque):
            norm_br_torque = torch.tensor(norm_br_torque)
            
        self.previous_def_e_tq = norm_e_torque
        self.previous_distance = current_distance
 
        return norm_e_torque, norm_br_torque
    
#%%
    
if __name__ == "__main__":
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    device = torch.device("cpu")
    
    journey_length = 200
    
    initial_speed = 10
    ref_vehicle_speed = 25 #m/s
    target_distance = 40
    ref_vehicle_position = 40 #m ahead of controlled vehicle
    dt = 1 #s
    
    current_distance = ref_vehicle_position

    follower = Car_classic(device, velocity = torch.tensor(initial_speed))
    
        
    # car position, car speed, vehicle position, vehicle speed
    state_storage = np.array([[0, initial_speed, ref_vehicle_position,ref_vehicle_speed]])
    # e torque, br torque
    ctrl_storage = np.array([[0, 0, 0, 0, 0]])
    cum_energy = 0
    
    for i in range(journey_length):
                
        classic_norm_e_torque, classic_norm_br_torque = follower.update(dt, torch.tensor(current_distance))
    
        acc = 2*np.random.randn()
        ref_vehicle_speed += acc * dt
        ref_vehicle_speed = np.clip(ref_vehicle_speed, 5, 30)
        ref_vehicle_position += ref_vehicle_speed* dt
        
        
        
        power = follower.e_power.item()
        cum_energy += power*dt
        
        current_distance = ref_vehicle_position - follower.position.item()
        error = target_distance - current_distance
        
        state_storage = np.append(state_storage, np.array([[follower.position.item(), \
                            follower.velocity.item(),ref_vehicle_position, ref_vehicle_speed ]]), axis = 0)
        ctrl_storage = np.append(ctrl_storage, np.array([[classic_norm_e_torque.item(), classic_norm_br_torque.item() , error, power, cum_energy]]), axis = 0)
        
        #print(f'iteration {i}')
    
    
    fig1 = plt.figure()
    ax_0 = fig1.add_subplot(3,1,1)
    ax_1 = fig1.add_subplot(3,1,2)
    ax_2 = fig1.add_subplot(3,1,3)
    
    
    ax_0.plot(ctrl_storage[:,2])
    ax_0.plot(0*np.ones((journey_length,1)), 'k',linewidth=0.5)
    ax_0.plot(35*np.ones((journey_length,1)), 'r')
    #ax_0.plot(-20*np.ones((journey_length,1)))
    ax_0.legend(['distancing error','reference','crash line' ])
    
    ax_1.plot(state_storage[:,2])
    ax_1.plot(state_storage[:,0])
    ax_1.legend(['leader pos','car position'])
    
    
    ax_2.plot(state_storage[:,3])
    ax_2.plot(state_storage[:,1])
    ax_2.legend(['leader vel','car vel'])
    
    fig1.savefig('state_signals.png')
    
    
    plt.show()
    
    fig2 = plt.figure()
    ax0 = fig2.add_subplot(3,1,1)
    ax1 = fig2.add_subplot(3,1,2)
    ax2 = fig2.add_subplot(3,1,3)
    
    ax0.plot(ctrl_storage[:,3])
    ax0.legend(['power'])
    
    ax1.plot(ctrl_storage[:,4])
    ax1.legend(['cum energy'])
    
    ax2.plot(ctrl_storage[:,0])
    ax2.plot(ctrl_storage[:,1])
    ax2.legend(['electric tq','brake tq'])
    
    fig2.savefig('ctrl_signals.png')
    
    plt.show()
    
    
    
    
    
    