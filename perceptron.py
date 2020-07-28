# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""


import numpy as np

class Perceptron(object):
    
    def __init__(self,no_of_inputs,threshold=8,learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate=learning_rate
        self.weights = np.zeros(no_of_inputs+1)
        self.counter=1    
        
    def predict(self,inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation
            
    def train(self, training_inputs, labels):
            
                for _ in range(self.threshold):
                    for inputs,label in zip(training_inputs, labels):
                        prediction = self.predict(inputs)
                        print("weights:",self.weights[1:])
                        print("bias",self.weights[0])
                        self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                        self.weights[0] +=self.learning_rate * (label - prediction)
                                
                    
