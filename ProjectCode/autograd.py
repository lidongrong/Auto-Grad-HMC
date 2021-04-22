# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 11:32:12 2021

@author: a
"""

from collections import defaultdict
import math
import numpy as np


#Node object, contain value of the node, reference to its children
class Node():
    def __init__(self,value,local_grad=()):
        self.value=value
        self.local_grad=local_grad
    def __add__(self,other):
        return add(self,other)
    def __mul__(self,other):
        return mul(self,other)
    def __sub__(self,other):
        return add(self,neg(other))
    def __truediv__(self,other):
        return mul(self,inv(other))
    def __pow__(self,other):
        return power(self,other)
    
# common useful functions   
def add(a,b):
    value=a.value+b.value
    local_grad=((a,1),(b,1))
    return Node(value,local_grad)

def mul(a,b):
    value=a.value*b.value
    local_grad=((a,b.value),(b,a.value))
    return Node(value,local_grad)

def scale(scalar,a):
    value=a.value*scalar
    local_grad=((a,scalar),)
    return Node(value,local_grad)


def neg(a):
    value=-1*a.value
    local_grad=((a,-1),)
    return Node(value,local_grad)

def inv(a):
    value=1.0/a.value
    local_grad=((a,-1.0/a.value**2),)
    return Node(value,local_grad)

def sin(a):
    value=math.sin(a.value)
    local_grad=((a,math.cos(a.value)),)
    return Node(value,local_grad)

def cos(a):
    value=math.cos(a.value)
    local_grad=((a,-1*math.sin(a.value)),)
    return Node(value,local_grad)

def exp(a):
    value=math.exp(a.value)
    local_grad=((a,math.exp(a.value)),)
    return Node(value,local_grad)

def log(a):
    value=math.log(a.value)*(a.value>0)+0.01*(a.value<=0)
    local_grad=((a,1/a.value),)
    return Node(value,local_grad)

def power(a,n):
    value=a.value**n
    local_grad=((a,n*(a.value**(n-1))),)
    return Node(value,local_grad)



# Compute gradient using computational graph
def get_grad(node):
    # construct a default dictionary and add grad into it
    grad=defaultdict(lambda:0)
    # The recursive part
    # path value stands for the value w.r.t the node, which shall be multiplied
    # to the grad of the child under chain rule
    def compute_grad(current_node,path_value):
        for child_node, child_grad in current_node.local_grad:
            value_to_child=path_value*child_grad
            grad[child_node]=grad[child_node]+value_to_child
            compute_grad(child_node,value_to_child)
    # df/df=1
    compute_grad(node,1)
    return grad

node_vector=np.vectorize(lambda x: Node(x))
value_vector=np.vectorize(lambda nodes: nodes.value)