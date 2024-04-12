import numpy as np
from numpy.random import default_rng
import pickle
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import sys
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
sys.path.append('../..')
from fitzhugh_nagumo import FitzhughNagumoODE,FitzhughNagumoSDE
from ensemble import Ensemble
import forcing
import algorithms
import utils

class FitzhughNagumoODEDirectNumericalSimulation(algorithms.ODEDirectNumericalSimulation):
    def obs_dict_names(self):
        return ['obs_x','obs_y']
    def obs_fun(self, t, x):
        obs_dict = dict({
            name: getattr(self.ens.dynsys, name)(t,x)
            for name in self.obs_dict_names()
            })
        return obs_dict


