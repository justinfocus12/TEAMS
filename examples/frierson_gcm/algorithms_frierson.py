# Instantiation of EnsembleMember class on Frierson GCM

import numpy as np
from numpy.random import default_rng
from scipy.special import softmax
import xarray as xr
import dask
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
from os.path import join, exists, basename
from os import mkdir, makedirs
import sys
import shutil
import glob
import subprocess
import resource
import pickle
import copy as copylib

import sys
sys.path.append("../..")
from ensemble import Ensemble
from dynamicalsystem import DynamicalSystem
import forcing
import algorithms
import frierson_observables as frobs
from frierson_gcm import FriersonGCM

class FriersonGCMPeriodicBranching(algorithms.PeriodicBranching):
    def obs_dict_names(self):
        return ['total_rain','column_water_vapor']
    def obs_fun(self, t, ds):
        obs = dict()
        for key in self.obs_dict_names():
            obs[key] = getattr(self.ens.dynsys, key)(ds)
        return obs
    def generate_next_icandf(self):
        

