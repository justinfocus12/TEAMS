import numpy as np


class EnsembleMember:
    def __init__(self, forcing):
        self.forcing = forcing # includes the initial condition, the random number seeds, and/or any deterministic forcing that happens over the course of the trajectory
        return

