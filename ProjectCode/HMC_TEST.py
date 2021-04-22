# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:28:49 2021

@author: a
"""
import matplotlib.pyplot as plt
import autograd as ag
import HMC
import numpy as np

##########################################################
'''
TEST GAMMA
'''
##########################################################

'''
x=ag.node_vector([4])

model=HMC.Sampler(HMC.gam,x)
draw=model.HMC(6000,20,0.05)

draw=np.array(draw)
draw=draw[3000:]

fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
_=ax1.hist(draw,bins=100,color='k')
'''


########################################################
'''
TEST NORMAL
'''
########################################################
# test code, test the performance of the model

x=ag.node_vector([1,1])
model=Sampler(std_normal,x)
model.set_M(model.M)
draw=model.HMC(6000,10,0.1)


import matplotlib.pyplot as plt
fig=plt.figure()


draw=np.array(draw)
draw=draw[3000:,:]
x=draw[:,0]
y=draw[:,1]
ax1=fig.add_subplot(1,1,1)
ax1.scatter(x,y)
fig




