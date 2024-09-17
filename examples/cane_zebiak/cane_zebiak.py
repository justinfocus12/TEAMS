# Instantiation of EnsembleMember class on Cane-Zebiak model 

import numpy as np
import xarray as xr
import f90nml
from matplotlib import pyplot as plt, rcParams 
rcParams.update({
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
import pprint

import sys
sys.path.append("../..")
import utils 
from dynamicalsystem import DynamicalSystem
import forcing

class CaneZebiak(DynamicalSystem):
    # TODO for SP: implement concrete methods implementing the abstract methods of DynamicalSystem, as stipulated in "../../dynamicalsystem.py". They are listed below. 
    def generate_default_icandf(self,init_time,fin_time,seed=None): 
        """
        Inputs
        - init_time: time corresponding to an initial condition (should be an integer, specifying the number of time units since time 0)
        - fin_time: time (integer) when the integration should finish
        - seed: integer for random-number generator to create the random force

        Outputs
        - icandf of the format specified in the inputs to run_trajectory

        Side effects: none (unless needed to calculate outputs). 
        """
        pass
    @staticmethod
    def get_timespan(metadata):
        pass
    @staticmethod
    def observable_props():
        # Should return a dictionary whose keys correspond to class methods and whose values correspond to plotting arguments
        pass
    def compute_pairwise_observables(self, pair_funs, md0, md1list, root_dir):
        """
        Not needed now (helpful for comparison of two trajectories, e.g., Euclidean distance, mean)
        """
        return 
    def compute_observables(self, obs_funs, metadata, root_dir):
        # obs_names must correspond to class methods
        """
        Not needed now (helpful for batch calculation of multiple low-dimensional observables without reading the output files repeatedly)
        """
        return
    def run_trajectory(self, icandf, obs_fun, saveinfo, root_dir):
        """
        Inputs
        - icandf (for 'initial conditions and forcing'): dictionary with following items
            a. init_cond: filename containing the initial condition, e.g., a restart leftover from a spinup run, BEFORE perturbation (because later ensemble members might want to launch from the same file with their own perturbation).
            b. frc: an object of type CaneZebiakForcing (see the class below). 
        - obs_fun: an "observable function" that the external calling function might want to use immediately for deciding what to do next, without having to read the cumbersome output files produced by the model. If not needed, just set obs_fun = lambda model_output: None
        - saveinfo: output filenames, temporary directories, etc.; whatever is needed to specify where to store the output. In case you want to move the whole output directory elsewhere later, I suggest making these paths relative to the final argument, root_dir.
        - root_dir: the path relative to which saveinfo is specified. 

        Outputs: 
        - metadata: all info needed to recreate the output (suggested format below). 
        - observables: result of evaluating obs_fun on the integration.

        Side effects: model gets integrated and saved to the file system. 

        """
        metadata = dict({
            'icandf': icandf, 
            'filename_traj': join(root_dir,saveinfo['filename_traj']),
            'filename_restart': join(root_dir,saveinfo['filename_restart']),
            })
        return metadata, observables

class CaneZebiakForcing(forcing.Forcing):
    # TODO for SP: implement concrete methods implementing the abstract methods of Forcing, as stipulated in "../../forcing.py". (There's only one: get_forcing_times()). Beyond that, devise a compact data structure that fully specifies the times and shapes of perturbations, which can then be used in CaneZebiak.run_trajectory(). 
    def get_forcing_times():
        pass
