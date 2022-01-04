## Welcome 

This repository contains code to build N to N artificial Neural Network , N to N means N number of layers and units . 
We are going to use the concept of oop e.g class , instance . This project require knowledge about oop concept , for loop , function , numpy , dictionary . 
We will build our own model and  we will implement MNIST dataset to our model . 

MNIST dataset 
<img src="https://user-images.githubusercontent.com/76767487/148059145-c2b13ff0-ac67-4f79-b170-11de64a3d7a6.png" width=600 height=400 />

## LETS GET STARTED 

First we will create class which will help to get instance , and inside that first we will initialize list and dictionary , dictionary is main storage to store weights , units .
```
 class ANN : 
    def __init__(self):
        self.no_of_units_in_layers=[]
        self.activation_fun=[]
        self.weights={}
        self.units={}
        self.bias={}
```
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Below  ,  we have made instance of ANN class which is x , here we have made a add function which will create a layer , it require no of units and an activation function associated to that layer . we have added the value of units and activation to list . here we have add two time add function means we have two layers one with 3 units with sigmoid activation and second is 2 units . last time we call add function is our output layer , we have 2 units in output layer means we have 2 class (:. like cat,dog ).
```
class ANN:
    def __init__(self):
        self.no_of_units_in_layers=[]
        self.activation_fun=[]
        self.weights={}
        self.units={}
        self.bias={}
    
    def add(self,unit=0,activation=0):  # default value of activation is 0
        self.no_of_units_in_layers.append(int(unit))
        self.activation_fun.append(activation)

x=ANN()
x.add(3,"sigmoid")
x.add(2)
```        
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Now its time to create activation function , it takes two arguments first is array and second is a number it tells what layer need to use which activation function . 
sigmoid function : converts +ive integer to a  number in between 0 to 1 . formula is 1 / (1 + e**(-x) ) , we have use math function to use value of e . 
relu function    : if value less than 0 than it will return 0 otherwise it will return the original value . 
```
class ANN:
    import math
    def __init__(self):
        self.no_of_units_in_layers=[]
        self.activation_fun=[]
        self.weights={}
        self.units={}
        self.bias={}
    
    def add(self,unit=0,activation=0):  # default value of activation is 0
        self.no_of_units_in_layers.append(int(unit))
        self.activation_fun.append(activation)
    def activation_function(self,x,i):
        if self.activation_fun[i]=="sigmoid":
            return 1/(1 + self.math.e**(-x))
        elif self.activation_fun[i]=="relu":
            return x*(x>0)    
x=ANN()
x.add(3,"sigmoid")
x.add(2)
```     
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Now its time to take input , output , learning rate , epochs to our model .to do that we have function called ann. all the things happen inside the ann function . we use numpy library for creation of array , getting random values . We have call ann function at last in code. input is an array for exaple MNIST dataset contains 
1) creation of weights : initially the value of weights is random and get change by backpropagation . to store weights we use dictionary 

```
input=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]
output=[0.01,0.99]

class ANN:
    import math,numpy as np
    def __init__(self):
        self.no_of_units_in_layers=[]
        self.activation_fun=[]
        self.weights={}
        self.units={}
        self.bias={}
    
    def add(self,unit=0,activation=0):  # default value of activation is 0
        self.no_of_units_in_layers.append(int(unit))
        self.activation_fun.append(activation)
    def activation_function(self,x,i):
        if self.activation_fun[i]=="sigmoid":
            return 1/(1 + self.math.e**(-x))
        elif self.activation_fun[i]=="relu":
            return x*(x>0)    
    def ann(self,inputs,outputs,epochs=10,learning_rate=0.1):  # default value  of learning rate is 0.1 and epochs is 10 
       ###### creation of weights 
       self.weights['0']=np.random.uniform(-0.5,0.5,(len(inputs[0]),self.no_of_units_in_layers[0]))
       for i in range(len(self.no_of_units_in_layers)-1):
           self.weights['{}'.format(i+1)]=np.random.uniform(-0.5,0.5,(self.no_of_units_in_layers[i],self.no_of_units_in_layers[i+1]))
       print("weights : ",self.weights)    
       ###### initializing the units 
       for i in range(len(self.no_of_units_in_layers)):
           self.units['{}'.format(i+1)] = np.zeros(self.no_of_units_in_layers[i])
       print("units : ",self.units)        
       for i in range(len(self.units)):
           self.bias['{}'.format(i+1)]=np.zeros(len(self.units['{}'.format(i+1)]))
       print("bias : ",self.bias)    
       
x=ANN()
x.add(3,"sigmoid")
x.add(2)
x.ann(input,output,500,learning_rate=0.01)
```  
