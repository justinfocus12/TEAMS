Tour of TEAMS software using Lorenz96 as an example

I. lorenz96.py: definition of the dynamical system
    A. Note the difference between Lorenz96SDE and Lorenz96ODE: the former uses continuous-time stochastic forcing, the latter uses instantaneous kicks. Every instantiation of Lorenz96SDE has an encapsulated object of type Lorenz96ODE, mirroring the structure in the abstract classes ODESystem and SDESystem.  
    B. Lorenz96ODE
        1. Requires a dictionary "config" to instantiate; the method "default_config()" shows how it should be structured. 
    C. 
II. DNS (specific kind of ensemble) -- 
