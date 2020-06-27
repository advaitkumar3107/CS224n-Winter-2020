#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, char_embed_size, word_embed_size):
    	super(CNN, self).__init__()
    	self.conv = nn.Conv1d(char_embed_size, word_embed_size, kernel_size = 5, padding = 1, bias = True)
 
    def forward(self,x):
    	x = self.conv(x)
    	x, _ = torch.max(x, dim = 2)

    	return x
    ### END YOUR CODE

