# Auto-Grad-HMC

## Background and Description

HMC is the abbreviation of Hamiltonian Monte-Carlo algorithm, which is a widely-used sampling algorithm in Bayesian statistics. 

When running HMC, we need to compute the gradient of log posterior function, which is the main difficulty for this algorithm. In our implementation, gradient is computed via auto differentiation on computational graph. By using auto differentiation, the sampling code would be easy to implement for users, since the only thing they need to do is to define the computational graph.

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



## Reference

Some of the codes and algorithms are inspired by:

1. Reverse-mode automatic differentiation from scratch, in Python: https://sidsite.com/posts/autodiff/

2. Radford M. Neal, MCMC using Hamiltonian dynamics
