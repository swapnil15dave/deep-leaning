# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:18:26 2020

@author: DAVE SWAPNIL
"""


import numpy as np

from perceptron import Perceptron

# To predict whether the it will be LBW or not??

#The input sequence is 

#   PITCH IN LINE ,   IMPACT    ,  MISSING STUMPS    


training_inputs = []
training_inputs.append(np.array([1,0,0]))
training_inputs.append(np.array([0,1,1]))
training_inputs.append(np.array([1,1,1]))

# The Output of the given training data

labels = np.array([0,0,1])

percept = Perceptron(3)
percept.train(training_inputs, labels)

inputs = np.array([0,1,0])
print(percept.predict(inputs))

inputs = np.array([0,0,1])
print(percept.predict(inputs))

inputs = np.array([1,1,0])
print(percept.predict(inputs))

inputs = np.array([1,0,1])
print(percept.predict(inputs))
