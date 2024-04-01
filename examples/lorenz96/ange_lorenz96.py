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
import lorenz96; reload(lorenz96)
import ensemble; reload(ensemble)
import forcing; reload(forcing)
import algorithms; reload(algorithms)
import algorithms_lorenz96; reload(algorithms_lorenz96)
import utils; reload(utils)

def ange_lorenz96_paramset(i_expt):
    # Physical
    F4s = [3,1,0.5,0.25]
    config_sde = lorenz96.Lorenz96SDE.default_config()
    config_sde['frc']['white']['wavenumber_magnitudes'][0] = F4s[i_F4]
    # Algorithmic
    deltas_phys = [2,1,0.5,0.0]
    config_algo = dict({

        })
    







