# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:22:12 2020

@author: Enrico Regolin
"""

# This module contains the NN Class used (variable number of linear ReLU layers)
# 2 additional funcitons used for the definition of the loss inside the Net() Class

import numpy as np
#import pandas as pd
import pickle
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

   
    
#%%
#
# NN class
class LinearModel(nn.Module):
    def __init__(self,net_type='LinearModel0',lr =0.001, n_outputs=1, n_inputs=10,*argv,**kwargs):
        super().__init__()
    
    # NNs of this class can have a maximum depth of 5 layers
    ## HERE WE PARSE THE ARGUMENTS
    
        self.net_type = net_type
    
        self.net_depth = np.maximum(1,len(argv))
        self.n_inputs = n_inputs
        
        if not argv:
            self.n_layer_1 = 1
        else:
            for i,arg in zip(range(1,self.net_depth+1),argv):
                setattr(self, 'n_layer_'+str(i), arg)
        self.n_outputs = n_outputs
    
        self.lr = lr  
        #self.log_df = pd.DataFrame(columns=['Epoch','Loss Train','Loss Test'])
        self.reshuffle_in_epoch = True
    
        self.update_NN_structure()    
    
    ## let's make the net dynamic by using the "property" decorator
        @property
        def net_depth(self):
            return self._net_depth
        @net_depth.setter
        def net_depth(self,value):
            if value > 5:
                raise AttributeError(f'NN is too deep')
            else:
                self._net_depth = value     
        
        @property
        def n_inputs(self):
            return self._n_inputs
        @n_inputs.setter
        def n_inputs(self,value):
            self._n_inputs = value
            
        @property
        def n_layer_1(self):
            return self._n_layer_1
        @n_layer_1.setter
        def n_layer_1(self,value):
            self._n_layer_1 = value
             
        @property
        def n_layer_2(self):
            return self._n_layer_2
        @n_layer_2.setter
        def n_layer_2(self,value):
            self._n_layer_2 = value     
        
        @property
        def n_layer_3(self):
            return self._n_layer_3
        @n_layer_3.setter
        def n_layer_3(self,value):
            self._n_layer_3 = value     

        @property
        def n_layer_4(self):
            return self._n_layer_4
        @n_layer_4.setter
        def n_layer_4(self,value):
            self._n_layer_4 = value     

        @property
        def n_layer_5(self):
            return self._n_layer_5
        @n_layer_5.setter
        def n_layer_5(self,value):
            self._n_layer_5 = value     

        @property
        def n_outputs(self):
            return self._n_outputs
        @n_outputs.setter
        def n_outputs(self,value):
            self._n_outputs = value     

        @property
        def fc1(self):
            return self._fc1
        @fc1.setter
        def fc1(self,value):
            self._fc1 = value
        
        @property
        def fc2(self):
            return self._fc2
        @fc2.setter
        def fc2(self,value):
            self._fc2 = value
            
        @property
        def fc3(self):
            return self._fc3
        @fc3.setter
        def fc3(self,value):
            self._fc3 = value
        
        @property
        def fc4(self):
            return self._fc4
        @fc4.setter
        def fc4(self,value):
            self._fc4 = value
        
        @property
        def fc5(self):
            return self._fc5
        @fc5.setter
        def fc5(self,value):
            self._fc5 = value
        
        @property
        def fc_output(self):
            return self._fc_output
        @fc_output.setter
        def fc_output(self,value):
            self._fc_output = value

        """
    ## HERE WE DEAL WITH ADDITIONAL (OPTIONAL) ARGUMENTS
        # lr = optimizer learning rate
        default_optimizer = optim.Adam(self.parameters(),self.lr)
        default_loss_function = nn.MSELoss()  #this might be made changable in the future 
        default_weights_dict = {new_list: .1 for new_list in range(1,11)} 
        
        allowed_keys = {'EPOCHS':6,'BATCH_SIZE':200,'device': torch.device("cpu"), \
                        'optimizer' : default_optimizer , 'loss_function' : default_loss_function, \
                        'weights_dict' : default_weights_dict }#, 'VAL_PCT':0.25 , \
                        
        # initialize all allowed keys
        self.__dict__.update((key, allowed_keys[key]) for key in allowed_keys)
        # and update the given keys by their given values
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        """
        
        
        # imported from Conv Net
        # define Adam optimizer
        self.initialize_optimizer()
        
        # initialize mean squared error loss
        self.criterion = nn.MSELoss()
        #self.criterion = weighted_mse_loss()
        
        
    ##########################################################################
    def initialize_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr= self.lr)

    ##########################################################################
    def update_learning_rate(self, new_lr):
        self.lr = new_lr  
        self.initialize_optimizer()

    ##########################################################################
    ## HERE WE re-DEFINE THE LAYERS when needed        
    def update_NN_structure(self):
        # defining first layer
        self.fc1 = nn.Linear(self.n_inputs, self.n_layer_1)
        # defining layers 2,3,etc.
        for i in range(1,self.net_depth):
            setattr(self, 'fc'+str(i+1), nn.Linear(getattr(self, 'n_layer_'+str(i)),getattr(self, 'n_layer_'+str(i+1) ) ) )
        # define last layer
        last_layer_width = getattr(self, 'n_layer_'+str(self.net_depth)  )
        self.fc_output = nn.Linear(last_layer_width, self.n_outputs)

    ##########################################################################        
    ## HERE WE DEFINE THE PATH
    # inside the forward function we can create CONDITIONAL PATHS FOR CERTAIN LAYERS!!
    def forward(self,x):
        iterate_idx = 1
        for attr in self._modules:
            if 'fc'+str(iterate_idx) in attr:
                x = F.relu(self._modules[attr](x))
                iterate_idx += 1
                if iterate_idx > self.net_depth:
                    break
        x = self.fc_output(x)
        return x #F.log_softmax(x,dim=1)  # we return the log softmax (sum of probabilities across the classes = 1)
    
    
    
    ##########################################################################    
    def updateWeights(self,y_batch,q_value ):
        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.optimizer.zero_grad()
        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()
        # calculate loss
        loss = self.criterion(q_value, y_batch)
        # do backward pass
        loss.backward()
        self.optimizer.step()  
        
        return loss


    ##########################################################################    
    def updateWeightsFromLoss(self, loss_in):
        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        self.optimizer.zero_grad()
        loss = self.criterion(loss_in.unsqueeze(0).float(),torch.zeros((1,loss_in.shape[0]), dtype = torch.float32))
        
        a = list(self.parameters())[0].clone()
        loss.backward()
        self.optimizer.step()
        b = list(self.parameters())[0].clone()
        print(torch.equal(a.data, b.data))
                
        return loss
    
    
    ##########################################################################
    # save the network parameters and the training history
    def save_net_params(self,path_log = None,net_name = 'my_net', epoch = None, loss = None):
        
        if path_log is None:
            path_log = os.path.dirname(os.path.abspath(__file__))
        
        filename_pt = os.path.join(path_log, net_name + '.pt')
        #opt_filename_pt = os.path.join(path_log, net_name + '_opt.pt')

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            },filename_pt)
        
        #torch.save(self.optimizer.state_dict(),opt_filename_pt)


    ##########################################################################
    # load the network parameters and the training history
    def load_net_params(self,path_log,net_name, device):    
        filename_pt = os.path.join(path_log, net_name + '.pt')
        checkpoint = torch.load(filename_pt, map_location=device)
        
        epoch = checkpoint['epoch']
        model_state = checkpoint['model_state_dict']
        opt_state = checkpoint['optimizer_state_dict']
        loss = checkpoint['loss']
        
        # update net parameters based on state_dict() (which is loaded afterwards)
        self.n_inputs =  (model_state['fc1.weight']).shape[1]
        self.net_depth = int(len(model_state)/2)-1
        for i in range(1,self.net_depth+1):
            setattr(self, 'n_layer_'+str(i), (model_state['fc' + str(i) + '.weight']).shape[0])
        self.n_outputs =  (model_state['fc_output.weight']).shape[0]
        self.update_NN_structure()        
        self.load_state_dict(model_state)
        self.to(device)  # required step when loading a model on a GPU (even if it was also trained on a GPU)
        
        self.initialize_optimizer()
        self.optimizer.load_state_dict(opt_state)
        
        self.eval()

