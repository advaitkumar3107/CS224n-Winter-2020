#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, shape):
    	super(Highway, self).__init__()
    	self.projection = nn.Linear(shape, shape, bias = True)
    	self.gate = nn.Linear(shape, shape, bias = True)
    	self.relu = nn.ReLU(inplace = True)
    	self.sigmoid = nn.Sigmoid()


    def forward(self, x):
    	projection = self.projection(x)
    	projection = self.relu(projection)

    	gate = self.gate(x)
    	gate = self.sigmoid(gate)

    	highway = gate * projection + (1 - gate) * x

    	return highway

    ### END YOUR CODE