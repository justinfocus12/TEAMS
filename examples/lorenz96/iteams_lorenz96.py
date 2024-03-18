i = 0
print(f'--------------Beginning imports-------------')
import numpy as np
print(f'{i = }'); i += 1
from numpy.random import default_rng
print(f'{i = }'); i += 1
import xarray as xr
print(f'{i = }'); i += 1
from matplotlib import pyplot as plt, rcParams 
print(f'{i = }'); i += 1
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
print(f'{i = }'); i += 1
from os.path import join, exists, basename, relpath
print(f'{i = }'); i += 1
from os import mkdir, makedirs
print(f'{i = }'); i += 1
import sys
print(f'{i = }'); i += 1
import shutil
print(f'{i = }'); i += 1
import glob
print(f'{i = }'); i += 1
import subprocess
print(f'{i = }'); i += 1
import resource
print(f'{i = }'); i += 1
import pickle
print(f'{i = }'); i += 1
import copy as copylib
print(f'{i = }'); i += 1
from importlib import reload

sys.path.append("../..")
print(f'Now starting to import my own modules')
import utils; reload(utils)
print(f'{i = }'); i += 1
import ensemble; reload(ensemble)
from ensemble import Ensemble
print(f'{i = }'); i += 1
import forcing; reload(forcing)
print(f'{i = }'); i += 1
import algorithms; reload(algorithms)
print(f'{i = }'); i += 1
import lorenz96; reload(lorenz96)
from lorenz96 import Lorenz96SDE
print(f'{i = }'); i += 1


class Lorenz96SDEITeams(algorithms.ITEAMS):
    def derive_parameters(self, config):
        sc = config['score']
        self.score_params = dict({
            'ks2avg': sc['ks'], # List of sites of interest to sum over
            'kweights': sc['kweights'],
            'tavg': sc['tavg'],
            })
        super().derive_parameters(config)
        return
    def score_components(self, t, x):
        scores = list((x[:,self.score_params['ks2avg']]**2).T)
        return scores
    def score_combined(self, sccomps):
        score = np.zeros(sccomps[0].size)
        return np.mean(np.array([sccomps[i]*self.score_params['kweights'][i] for i in range(len(sccomps))]), axis=0)
    @staticmethod
    def label_from_config(config):
        abbrv_population,label_population = algorithms.ITEAMS.label_from_config(config)
        abbrv_k = '_'.join([
            r'%gx%g'%(
                self.score_params['kweights'][i],
                self.score_params['ks2avg'][i]) 
                for i in range(len(self.score_params['ks2avg']))
            ])
        abbrv_t = r'tavg%g'%(self.score_params['tavg'])
        abbrv = r'ITEAMS_%s_%s_%s'%(abbrv_population,abbrv_k,abbrv_t)
        return abbrv,label_population

def iteams(i_param,seed_inc):
    tododict = dict({
        'run_iteams':             1,
        'plot_spaghetti':         1,
        })

