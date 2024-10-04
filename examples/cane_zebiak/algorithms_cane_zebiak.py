import numpy as np
from numpy.random import default_rng
from scipy.special import logsumexp,softmax
import xarray as xr
from matplotlib import pyplot as plt, rcParams 
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
import os
from os.path import join, exists, basename, relpath
from os import mkdir, makedirs
import sys
import shutil
import glob
import subprocess
import resource
import pickle
import copy as copylib
from importlib import reload

sys.path.append("../..")
print(f'Now starting to import my own modules')
import utils; #reload(utils)
import ensemble; #reload(ensemble)
from ensemble import Ensemble
import forcing; #reload(forcing)
import algorithms; #reload(forcing)
import cane_zebiak_sp; #reload(frierson_gcm)
from cane_zebiak_sp import CaneZebiak

class CaneZebiakTEAMS(algorithms.TEAMS)
    @classmethod
    def initialize_from_dns(cls, dns, config, ens):
        # Optionally, 
        pass
    @classmethod
    def initialize_from_base(cls, base_dir, config, ens):
        """
        Inputs:
        - cls: together with the @classmethod signature, this is implied to be CaneZebiakTEAMS (just like "self" in an instance method is implied to be an instance, e.g., czt where czt = CaneZebiakTEAMS(...))
        - base_dir: wherever the long-run restart files exist
        - config: a dictionary of TEAMS parameters which must contain the following (see "derive_parameters" method of algorithms.TEAMS to see what all keys get used)
            * population_size: the number of initial ancestors to generate (start with a small number, like 8, for debugging, but ultimately increase to 64 or 128. Doesn't have to be a power of two but why not)
            * num2drop: how many members to kill with each level-raising. 
            * num_active_families_min: how many ancestors must still be at play for the algorithm to continue. Once genetic diversity dwindles too much, the whole thing could go off the rails. 
            * time_horizon_phys: how long each ancestor runs after its starting point (something longer than the decorrelation time; probably 2-4 ENSO cycles)
            * buffer_time_phys: time to wait between the end-time of one ancestor and the start-time of the next one (ordered by initialization time from DNS); probably 0 is fine.
            * advance_split_time_phys: how far to split in advance. 
            * advance_split_time_max_phys: the upper bound on advance_split_time_phys (in practice this is not used in the algorithm right now; it's intended to be used when adaptively adjusting the advance split time)
            * split_landmark: either 'max' (meaning split at (peak timing) - advance_split_time) or 'thx' (meaning split at (threshold-crossing) - advance_split_time). The latter is currently standard, but feel free to play with this. 
            * inherit_perts_after_split (a boolean): Suppose advance_split_time = 10, and a family lineage is x0 -> x1 -> x2; x0 clears the threshold at time 24, x1 splits from x0 at time 14 (=24-10) and clears the next threshold at time 22. So then x2 splits from x1 at time 12 (=22-10). The question is, should x2 get perturbed only once at time 12, or both at time 12 and at time 14 (the latter with the same perturbation as x1 received at time 14)? There are arguments for both, so inherit_perts_after_split is another parameter to play with. If you want to set True (but also in any case), consult lines 224-237 of algorithms_frierson.py (within "generate_icandf_from_parent" within class TEAMS definition)
        - ens: Ensemble instance with zero members 



        """
        pass
    @abstractmethod
    def score_components(self, t, x):
        # Something directly computable from the system state. Return a dictionary
        # TODO 
        pass
    @abstractmethod
    def score_combined(self, t, sccomps):
        # sccomps is the output of score_components
        # Scalar score used for splitting, which is derived from sccomp; e.g., a time average
        # TODO 
        pass
    def merge_score_components(self, comps0, comps1, nsteps2prepend):
        pass
    def merge_score_components_simple(self, mem_leaf, score_components_leaf):
        # JF: this is my guess as to how to implement merge_score_components
        # The child always starts from the same restart as the ancestor, so no merging necessary
        return score_components_leaf
    @abstractmethod
    def generate_icandf_from_parent(self, parent, branch_time):
        pass

def run_teams(dirdict,filedict,config_model,config_algo):
    root_dir = dirdict['data']
    angel = pickle.load(open(filedict['angel'], 'rb'))
    if exists(filedict['alg']):
        alg = pickle.load(open(filedict['alg'], 'rb'))
        alg.set_capacity(config_algo['num_levels_max'], config_algo['num_members_max'])
    else:
        cz = CaneZebiak(config_gcm, recompile=recompile)
        ens = ensemble.Ensemble(gcm, root_dir=root_dir)
        #alg = algorithms_frierson.FriersonGCMTEAMS.initialize_from_ancestorgenerator(angel, config_algo, ens)
        alg = algorithms_frierson.FriersonGCMTEAMS.initialize_from_dns(angel, config_algo, ens)

    alg.ens.dynsys.set_nproc(nproc)
    alg.ens.set_root_dir(root_dir)
    while not alg.terminate:
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict({
            'temp_dir': f'mem{mem}_temp',
            'final_dir': f'mem{mem}',
            })
        saveinfo.update(dict({
            'filename_traj': join(saveinfo['final_dir'],f'history_mem{mem}.nc'),
            'filename_restart': join(saveinfo['final_dir'],f'restart_mem{mem}.cpio'),
            }))
        alg.take_next_step(saveinfo)
        if exists(filedict['alg']):
            os.rename(filedict['alg'], filedict['alg_backup'])
        pickle.dump(alg, open(filedict['alg'], 'wb'))
    return
