*Basic instructions on running the TEAMS algorithm, with particular focus on the Frierson GCM with stochastic parameterization*

This repository contains code for running the TEAMS algorithm, as well as more general ensemble-generating procedures for dynamical systems. Among the included examples are the Frierson GCM featured in Finkel & O'Gorman (2025), as well as the Lorenz-96 system featured in Finkel & O'Gorman (2024). The summary below is meant to convey the general design of the codebase, which aims to enable users to extend it to implement new algorithms, and applied to different systems. But the implementation is not professional, and will certainly need modification. Interested readers are highly encouraged to contact Justin Finkel (justinfocus12@gmail.com) for guidance on using it. Still, we hope that the core ideas conveyed here, in the publications, and in the code structure will help to facilitate community uptake of our methods. 

I. The driver code hierarchy: dynamical systems and algorithms for running them in ensemble mode
    A. DynamicalSystem class
    B. Ensemble class
    C. Algorithm class
II. The Lorenz96 system
III. The Frierson GCM 
IV.  TEAMS on the Frierson GCM
V.   PeriodicBranching on the Frierson GCM
