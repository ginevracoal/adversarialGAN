'''
File name: plot-pytorch-autograd-graph.py
Author: Ludovic Trottier
Date created: November 8, 2017.
Date last modified: November 8, 2017
Credits: moskomule (https://discuss.pytorch.org/t/print-autograd-graph/692/15)
'''
from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_dot(var, params, path):
    """ Produces Graphviz representation of PyTorch autograd graph.
    
    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.
    
    Args:
        var: output Variable
        params: list of (name, Parameters)
    """
    
    param_map = {id(v): k for k, v in params}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(
        filename='network', 
        format='pdf',
        node_attr=node_attr, 
        graph_attr=dict(size="12,12"))
    seen = set()
    
    def add_nodes(var):
        if var not in seen:
            
            node_id = str(id(var))
             
            if torch.is_tensor(var):
                node_label = "saved tensor\n{}".format(tuple(var.size()))
                dot.node(node_id, node_label, fillcolor='orange')
                
            elif hasattr(var, 'variable'):
                variable_name = param_map.get(id(var.variable))
                variable_size = tuple(var.variable.size())
                node_name = "{}\n{}".format(variable_name, variable_size)
                dot.node(node_id, node_name, fillcolor='lightblue')
                
            else:
                node_label = type(var).__name__.replace('Backward', '')
                dot.node(node_id, node_label)
                
            seen.add(var)
            
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
                        
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)

    dot.render(filename='computational_graph.dot', directory=path, view=False)