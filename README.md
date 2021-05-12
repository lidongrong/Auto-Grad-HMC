# Auto-Grad-HMC

## Background and Description

HMC is the abbreviation of Hamiltonian Monte-Carlo algorithm, which is a widely-used sampling algorithm in Bayesian statistics. 

When running HMC, we need to compute the gradient of log posterior function, which is the main difficulty for this algorithm. In our implementation, gradient is computed via auto differentiation on computational graph. By using auto differentiation, the sampling code would be easy to implement for users, since the only thing they need to do is to define the computational graph.

This package is based on numpy and scipy

## HMC And Bayesian Statistics

As we've just discussed above, Markov Chain Monte Carlo (MCMC) algorithms play an important role in Bayesian inference. In general cases, we can not solve out the posterior analytically. To tackle this problem, numerical simulation techniques such as MCMC are adopted. HMC is a gradient-based posterior sampler, that means HMC only take the gradient of posterior into account and hence avoid difficult high-dimensional integration that one may come across in Gibbs sampler.

A key advantange of HMC is that it does not need to compute the marginal distribution p(X). This is because when taking gradient w.r.t. the parameters in posterior, p(X) will simply vanish because it has nothing to do with the parameters and can then be treated as a constant. Hence, to simulate the posterior, prior & likelihood are the only thing we need to compute in HMC algorithm. **Once we've acquired prior & likelihood, HMC will automatically sample from posterior**, which is very convenient and has a little bit style of Probablistic Programming.

## Sample From a Normal Model

In this section, a toy example will be given to demonstrate the inference procedure. 

We consider a normal model with known variance (set to be 1) and an unknow mean parameter $\theta$. The parameter $\theta$ itself follows a normal distribution with known mean and variance. The hierarchical structure of the model can be written as:

$\theta \sim N(\mu, b^2)$

$y|\theta \sim N(\theta, 1)$

To make the prior rather non-informative, we let $\mu=0, b^2=6$.

To perform the estimate, we generate $y$ from distribution $N(1,1)$. That is, we set $\theta$ to be 1. In this example, we generate 100 samples:

```py
y=np.random.randn(100)
y=y+1
```

Let $\tau_0=b^2=6$ and set the initial point of $\theta$ to be 0, we further define the prior density using computational graph:

```py
mu0=0
tau0=6
theta=node_vector([0])

# theta must be a node vector!
def prior(theta):
    t=theta[0]
    dom=ag.Node(math.sqrt(2*math.pi*tau0))
    num=(theta[0]+ag.Node(-1*mu0))*(theta[0]+ag.Node(-1*mu0))
    num=ag.Node(-1)*num/ag.Node(2*tau0)
    f=ag.exp(num)/dom
    return f

```

Note that here the variable theta must be a Node vector, even if it is 1-dimensional. 

Then we construct the likelihood:

```py
def likelihood(theta):
    num=Node(-1)*np.dot(node_vector(y)+theta[0]*Node(-1),node_vector(y)+theta[0]*Node(-1))
    # HMC and autograd does not support scalar multiplication, so num/Node(2) instead of num/2
    num=num/Node(2)
    num=exp(num)
    dom=Node((2*math.pi)**5)
    f=num/dom
    return f
```

There are 2 things worth mentioning here. **First**, our package does not support the multiplication operation between a node and a scalar. Hence, to accomplish this operation, you must write like node * Node(scalar). That is, a scalar shall be transformed into a node with constant value. For instance, in the code above, num/2 is written as num/Node(2).

**Second**, all probability densities, including likelihood and prior, shall be set to be function with only $\theta$ as its parameters. Of course, likelihood is interpreted as a fundtion of $y$ given $\theta$. However, this may cause the program to be confused when calculateing posterior. So a recommended manner of writing likelihood is that you define data $y$ outside the likelihood function, and then involve the data into the likelihood function as global variables and don't pass them as parameters of a function.

After we've defined prior + likelihood, we can now immediately sample from posterior. We just pass the prior, likelihood and initial point to the sampler, then the sampler will automatically sample from posterior.

The whole sampler and the Bayes model is encapsulated in the **BayesModel** class. An object of **BayesModel** Class is initialized with parameters prior, likelihood and theta: 

```py
model=BayesModel(prior,likelihood,theta)
```

Then the model could immediately sample from the posterior by calling the **HMC()** function:

```py
#2000 -> number of simulated samples
#10 -> # of iterations in leapfrog step
#0.1 -> learning rate of per leapfrog step
# # of iterations times learning rate is recommended to be 1 (from numerical PDE)

draw=model.HMC(2000,10,0.1)
```

Discarding the first 1,000 samples as burn-in samples, we can acquire a posterior mean that is very close to theoretical result derived from Bayesian statistical theory:

```py
sample=draw[1000:]
sum(sample)/1000

# theoretical posterior mean from Bayesian statistical theory:
sum(y)/(1/6+100)
```

## Limitations

#### 2021.5.12
The sampler cannot deal with intractable likelihoods. Perhaps ABC (Approximation Bayesian Computation) will be adopted in the future.

The sampler may also have difficulty representing highly complex models such as PGM.


## Reference

Some of the codes and algorithms are inspired by:

1. Reverse-mode automatic differentiation from scratch, in Python: https://sidsite.com/posts/autodiff/

2. Radford M. Neal, MCMC using Hamiltonian dynamics
