# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 13:08:02 2020

@author: DAVE SWAPNIL
"""
import numpy as np

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def propagate(w,b,X_train,Y_train):
    # forward propagation
    m = X_train.shape[1]
    Z = np.dot(w.T,X_train)+b
    A = sigmoid(Z)
    print("-----A-----",A)
    cost = -np.sum(Y_train*np.log(A)+(1-Y_train)*np.log(1-A))*(1/m)
    
    #backward propagation
    
    dz = A-Y_train
    print("---dz----",dz)
    dw = np.dot(X_train,dz.T)*(1/m)
    print("----dw----",dw)
    db = np.sum(dz)*(1/m)
    print("----db----",db)
    
    grads = {
            "dw":dw,
            "db":db
        }
    
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost = np.squeeze(cost)
    assert(cost.shape==())
    
    return grads,cost

def optimize(w,b,X_train,Y_train,num_iterations=10,learning_rate=0.009,print_cost=False):
    costs = []
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X_train,Y_train)
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - (learning_rate*dw)
        print("----w----",w)
        b = b - (learning_rate*db)
        print("----b----",b)
        if(i%100==0):
            costs.append(cost)
    grads={
             "dw":dw,
             "db":db
             
         }       
    params={"w":w,
            "b":b
        }
    return params,grads,costs

def predict(w,b,X):

    Z = np.dot(w.T,X)+b
    A = sigmoid(Z)
    
    Y_predictions = np.zeros((1,X.shape[1]))
    for i in range(A.shape[1]):
        if(A[0,i]>0.5):
            Y_predictions[0,i] = 1
        else:
            Y_predictions[0,i] = 0
    return Y_predictions
        
            
def model(w,b,X_train,Y_train,X_test,Y_test,num_iterations=100,learning_rate=0.001,print_cost=False):
    
    params,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate)
    
    dw = grads["dw"]
    db = grads["db"]
    w = params["w"]
    b =  params["b"]
    Y_train_predict = predict(w,b,X_train)
    Y_test_predict = predict(w,b,X_test)
    
    print("dw===>",dw)
    print("w===>",w)
    print("db===>",db)
    print("b===>",b)
    print("y_train_predict",Y_train_predict)
    print("y_test_predict",Y_test_predict)
    


w,b = np.zeros((2,1)),0
X_train,Y_train = np.array([[0,1,0,1],[0,0,1,1]]),np.array([[0,0,0,1]])
X_test,Y_test = np.array([[1,1],[0,1]]),np.array([[1,0]])

print(X_train.shape)
print(Y_train.shape)

optimize(w,b,X_train,Y_train,num_iterations=10,learning_rate=0.001)
print(predict(w,b,X_train))
        
        
 
    
    
    
    
    
    