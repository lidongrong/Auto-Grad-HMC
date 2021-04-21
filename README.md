# Auto-Grad-HMC

## Background and Description

HMC is the abbreviation of Hamiltonian Monte-Carlo algorithm, which is a widely-used sampling algorithm in Bayesian statistics. 

When running HMC, we need to compute the gradient of log posterior function, which is the main difficulty for this algorithm. In our implementation, gradient is computed via auto differentiation on computational graph. By using auto differentiation, the sampling code would be easy to implement for users, since the only thing they need to do is to define the computational graph.

Let me update this later, because I want to find a way to insert LaTex in GitHub.

## Reference

Some of the codes and algorithms are inspired by:

1.Reverse-mode automatic differentiation from scratch, in Python: https://sidsite.com/posts/autodiff/
2.Radford M. Neal, MCMC using Hamiltonian dynamics
