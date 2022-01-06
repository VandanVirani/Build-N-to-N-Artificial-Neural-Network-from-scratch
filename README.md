## Welcome 

This repository contains code to build N to N artificial Neural Network , N to N means N number of layers and units . 
We are going to use the concept of oop e.g class , instance . This project require knowledge about oop concept , for loop , arrays , matrix , function , numpy , dictionary . 
We will build our own model and  we will implement MNIST dataset to our model . 

MNIST dataset 
<img src="https://user-images.githubusercontent.com/76767487/148059145-c2b13ff0-ac67-4f79-b170-11de64a3d7a6.png" width=600 height=400 />

## LETS GET STARTED 

First  we will use this function  to get the input and ouput from mnist data . The MNIST database (Modified National Institute of Standards and Technology database[1]) is a large database of handwritten digits that is commonly used for training various image processing systems . The MNIST database contains 60,000 training images . 
Because there are total 10 digits 0,1,2,3,4,5,6,7,8,9 there will be 10 class in last layer . 
```
import  numpy as np

def get_mnist():
    with np.load(f"D:/python programing/test/ann/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels
input,output= get_mnist()
print(input,output)
print(input.shape,output.shape)
```
<img src="https://user-images.githubusercontent.com/76767487/148325194-287d827c-a3dd-4fda-9038-cb1349feba94.jpeg" width=900 height=600 />

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
we will create class which will help to get instance , and inside that first we will initialize list and dictionary , dictionary is main storage to store weights , units .
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
    
    def add(self,unit,activation=0):                 # default value of activation is 0
        self.no_of_units_in_layers.append(int(unit))   # append the information in no_of_units_in_layers list
        self.activation_fun.append(activation)         # append the information in activation_fun list

x=ANN()
x.add(3,"sigmoid")
x.add(10)   # because 10 class are there 
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
x.add(10)
x.ann(input,output,500,learning_rate=0.01)    # input and output are taken from mnist dataset . 
```  

<img src="https://user-images.githubusercontent.com/76767487/148326714-1543568b-bd7d-4b36-8e71-06b12e3107c9.jpg" width=900 height=230 />

Now we will be thinking how i have created weights . see below image


![WhatsApp Image 2022-01-06 at 9 59 03 AM](https://user-images.githubusercontent.com/76767487/148328318-cb6342a4-db48-4b8a-9962-e4ba904fb66e.jpeg)


