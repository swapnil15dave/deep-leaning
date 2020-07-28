# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 22:18:26 2020

@author: DAVE SWAPNIL
"""


import numpy as np

from perceptron import Perceptron


training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

labels = np.array([1,1,1,0])

percept = Perceptron(2)
percept.train(training_inputs, labels)

inputs = np.array([1,1])
print(percept.predict(inputs))

inputs = np.array([0,0])
print(percept.predict(inputs))

inputs = np.array([0,1])
print(percept.predict(inputs))

inputs = np.array([1,0])
print(percept.predict(inputs))
