# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:03:22 2021

@author: a
"""

import math
import autograd as ag
import numpy as np
import scipy.stats as stats
import copy

# HMC takes in the pdf of a distribution, and then 
# sample based on the gradient of log pdf.
# momentum phi follows N(0,M), where N is usually set up to be a diagonal matrix
# In our code, we first implement an algorithm to sample from any distribution,
# not specifically prior

# When defining pdf and sampler, theta must be a vector of nodes!
# if you only have one theta, let theta=ag.node_vector(x) then

# To define a Sampler, we have to initialize a Sampler object
# with coefficients pdf and theta
# theta shall be a node list, representing initial value
# of input theta
# pdf shall be a function taking a node list as input and return a node, to
# simulate the pdf
# like this: pdf(theta): return theta[0]+theta[1]
class Sampler():
    def __init__(self,pdf,theta):
        self.pdf=pdf
        self.theta=theta
        self.dim=len(theta)
        self.M=np.eye(self.dim)
        self.M_inv=np.linalg.inv(self.M)
        # # of parameters
        self.para_size=len(theta)
    
    # set the covariance of the momentum, identity matrix as default
    def set_M(self,M):
        self.M=M
        self.M_inv=np.linalg.inv(M)
    
    # iteration stands for the # of total samples we draw
    # Leapfrog stands for the step of L for each leapfrog process
    # eps -> the stepsize in each leapfrog jump
    def HMC(self,iteration, Leapfrog,eps):
        # initialize the draw
        post_draw=[]
        # save class variables separately to accelerate the code
        M_inv=2*self.M_inv
        theta=self.theta
        # sample from mvn by linear tranform on std mvn
        M2=np.linalg.cholesky(self.M)
        #phi=np.dot(np.random.randn(self.para_size),M2)
        
        # start sampling
        for t in range(0,iteration):
            # Leapfrog process
            phi=np.dot(np.random.randn(self.para_size),M2)
            for i in range(0,Leapfrog):
                new_theta=copy.deepcopy(theta)
                # compute log pdf
                log_pdf=ag.log(self.pdf(new_theta))
                # auto differentiation
                diff=ag.get_grad(log_pdf)
                
                # start leap frog process, renew phi
                new_phi=[phi[i]+eps*0.5*(diff[new_theta[i]]) for i in range(0,self.para_size)]
                # leap frog process: renew theta
                tmp_theta_value=[new_theta[i].value+eps*M_inv[i,i]*new_phi[i] for i in range(0,self.para_size)]
                new_theta=ag.node_vector(tmp_theta_value)
                # compute the new gradient to update phi
                new_log_pdf=ag.log(self.pdf(new_theta))
                new_diff=ag.get_grad(new_log_pdf)
                # renew phi again
                new_phi=[new_phi[i]+eps*0.5*(new_diff[theta[i]]) for i in range(0,self.para_size)]
            # Metropolis step
            theta=self.Metropolis_step(new_theta,theta,new_phi,phi)
            # test code: print('cool')
            #phi=new_phi
            post_draw.append(ag.value_vector(theta))
            print(ag.value_vector(theta))
        return post_draw

    # Metropolis step in sampling, defined seperately for scalability
    # new/old_theta shall be a node list and phi a real number
    def Metropolis_step(self,new_theta,old_theta,new_phi,old_phi):
        p_old=self.pdf(old_theta)
        p_new=self.pdf(new_theta)
        #print(p_new.value)
        numerator=p_new.value*normal_pdf(new_phi,np.zeros(len(new_phi)),self.M)
        dominator=p_old.value*normal_pdf(old_phi,np.zeros(len(old_phi)),self.M)
        r=numerator/dominator
        prob=min(1.0,r)
        draw=np.random.binomial(1,prob)
        if draw==1:
            return new_theta
        else:
            return old_theta
        
        


#compute the pdf of normal distribution given vector x, mean u and covariance M
def normal_pdf(x,u,M):
    x=np.array(x)
    u=np.array(u)
    p=len(x)
    det=np.linalg.det(M)
    exponential=np.dot((x-u).T,np.dot(np.linalg.inv(M),x-u))
    exponential=np.exp(exponential)
    pdf=1.0/(((2*np.pi)**(p/2.0))*math.sqrt(det))*exponential
    return pdf

#
#
# test code
#
''' 
def gam(x):
    alpha=2
    beta=3
    c1=ag.Node(math.gamma(alpha))
    c0=ag.Node(1)
    c2=ag.Node(beta)
    c3=c2**alpha
    c4=c1*c3
    d1=c0/(c4)
    #x=ag.Node(x)
    d2=x[0]**(alpha-1)
    d3=ag.exp(ag.neg(x[0])/c2)
    f1=d2*d3
    f=d1*f1
    
    return f
'''

x=ag.node_vector([1,1])
def std_normal(x):
    u1=ag.Node(1)
    u2=ag.Node(2*math.pi)
    v1=u1/u2
    u3=x[0]*x[0]+x[1]*x[1]
    u4=ag.Node(0.5)*u3
    v2=ag.exp(ag.neg(u4))
    f=v1*v2
    return f


model=Sampler(std_normal,x)
draw=model.HMC(8000,10,0.1)


import matplotlib.pyplot as plt
fig=plt.figure()


draw=np.array(draw)
draw=draw[4000:,:]
x=draw[:,0]
y=draw[:,1]
ax1=fig.add_subplot(1,1,1)
ax1.scatter(x,y)
fig

'''
draw=np.array(draw)
ax1=fig.add_subplot(1,1,1)
_=ax1.hist(draw,bins=50,color='k')
'''