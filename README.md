## Welcome 

This repository contains code to build N to N artificial Neural Network , N to N means N number of layers and units . 
We are going to use the concept of oop e.g class , instance . This project require knowledge about oop concept , for loop , function , numpy , dictionary . 
We will build our own model and in upcoming repository we will implement MNIST dataset to our model

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


```
class ANN:
    def __init__(self):
        self.no_of_units_in_layers=[]
        self.activation_fun=[]
        self.weights={}
        self.units={}
        self.bias={}
    
    def add(self,unit=0,activation=0):
        self.no_of_units_in_layers.append(int(unit))
        self.activation_fun.append(activation)
```        
