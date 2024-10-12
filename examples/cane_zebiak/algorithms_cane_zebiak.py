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
    def initialize_from_dns_files(cls, base_dir, config, ens):
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
            * inherit_perts_after_split (a boolean): Suppose advance_split_time = 10, and a family lineage is x0 -> x1 -> x2; x0 clears the threshold at time 24, x1 splits from x0 at time 14 (=24-10) and clears the next threshold at time 22. So then x2 splits from x1 at time 12 (=22-10). The question is, should x2 get perturbed only once at time 12, or both at time 12 and at time 14 (the latter with the same perturbation as x1 received at time 14)? There are arguments for both, so inherit_perts_after_split is another parameter to play with. If you want to set True (but also in any case), consult lines 224-237 of algorithms_frierson.py (within "generate_icandf_from_parent" within class TEAMS definition). Also 
        - ens: Ensemble instance with zero members 



        """
        pass
    """
    The three "score_..." methods below handle the following weird situation. 

    level = 16
    delta = 3
    time =               [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    parent score R1(t) = [ 13,  14,  13,  14,  15,  16,  17,  16,  15,  14] (threshold crossed at time 106, so split child at time 106-3=103)
    child score  R2(t) = [  -,   -,   -,  12,  13,  12,  11,  10,   9,   8]  (- denotes implicit shared history with the parent, whereas the numbers represent what is written out in data files from the integration of the child. Notice the child's own output history starts at the time step AFTER the split, since we don't need to encode the redundant information that R2(101,102,103) = [13,14,13].))

    The parent's maximum score is S1 = 17. What is the child's maximum score? Naively, we might say 13, since that's the maximum achieved after the split. But we should really think of the child as sharing the same start time as the parent (which is time 100, not 101; see the comment above after R2(t)), and the same history up to and including 103, the split time, after which the child score diverges. Therefore, the child's maximum score is 14. This value is not available from the child's output history file(s)! Instead, we need to splice together scores from both parent and child. Hence the need for "merge_score_components", which prepends the child's score (what was just computed by the integration) with the parent's score up until the timing of the split. I did not have to do this for the Frierson GCM, because I only stored a restart file at time 100, and so the child integration explicitly replicates the integration done through time 103 (wastes CPU time, but saves storage and is conceptually simpler, as this example demonstrates). Since Cane-Zebiak can save out restarts every time step, you'll have to implement something similar to what I did in algorithms_lorenz96.py, circa lines 289-298. 

    You might still wonder, why "components" and not just "score"? Suppose we've chosen some weird score function such as 
    
    R(X(t)) = \max\{
                    \sum_{s=t-2}^t nino3(s),
                    upwelling(t)
                   \}
    }

    In this case, R(X(104)) is not computable from R(X(103)) alone. We need both the full timeseries nino3(t) and upwelling(t) --- the intended output of "score_components"---since time 98. Fortunately, the advance split time delta=3 is longer than the "memory" of 2, so before scoring the parent or child we replace R(X(101,102)) with NaN (see the latter parts of "take_next_step" in ../../algorithms.py, circa line 1177). I haven't yet thought about generalizing this condition, but I don't think you'll want to either ;) 

    Rather than writing a detailed input/output requirement for these functions, I'll let you try to fill them in considering how they are used by ../../algorithms.py, in "take_next_step". Refer also to the examples in ../lorenz96.py/algorithms_lorenz96 and ../frierson_gcm/algorithms_frierson.py. 

    """
    def score_components(self, trajectory_output_in_whatever_format):
        """
        TL;DR: return nino3_from_dat_file(trajectory_output_in_whatever_format)


        To understand how this is used in TEAMS, we need to follow this function as it's passed through a few layers of abstraction. Bear with me, and by the end you'll see it's very simple what to do.
        1. Look c. line 1159 in ../../algorithms.py.n the class definition for TEAMS. The function "self.score_components" (where "self" is the algorithm object) is passed as an argument to "self.ens.branch_or_plant", which is implemented by the Ensemble class in ../../ensemble.py.
        2. In ../../ensemble.py, The function "branch_or_plant" has the corresponding argument "obs_fun". Circa line 29, obs_fun is passed as an argument to "self.dynsys.run_trajectory" (where "self" now refers to the Ensemble object--remember we're looking at the Ensemble class definition now). Meanwhile, "self.dynsys" is the DynamicalSystem object, which is none other than CaneZebiak! And "run_trajectory" is a function you already implemented, so let's climb back down the ladder and look at that.
        3. In cane_zebiak_sp.py, c. lines 132-160 currently implement "run_trajectory", with the corresponding argument "obs_fun". Right now, you don't use obs_fun, because its purpose was unclear. Hopefully it becomes clearer now: obs_fun is the function that computes the score (more specifically, the components of the score to be spliced together with those of the parent before computing a new fused score). See the additional comments in run_trajectory. As suggested by the last argument of this function, it is up to you how to format the inputs and outputs to self.score_components; but it must be consistent between here and in CaneZebiak.run_trajectory(...,obs_fun,...).

        """
        pass
    def merge_score_components(self, child, nino3_child): 
        """
        This is a modified version of merge_score_components from ../lorenz96/algorithms_lorenz96.py c. line 289. You will probably have to modify a bit to match your data format, but this hopefully gets you halfway. 

        The purpose is toe splice together the child score with the parent score at the appropriate t_branch
        """
        init_time,fin_time = self.ens.get_member_timespan(child)
        parent = next(self.ens.memgraph.predecessors(child))
        init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        while init_time_parent > init_time:
            parent = next(self.ens.memgraph.predecessors(parent))
            init_time_parent,fin_time_parent = self.ens.get_member_timespan(parent)
        nino3_parent = self.branching_state['score_components_tdep'][parent]
        nsteps2prepend = len(nino3_parent) - len(nino3_child) # how old the parent was when the child was born 
        return np.concatenate((nino3_parent[:nsteps2prepend], nino3_child))


    def score_combined(self, nino3):
        """
        In Lorenz96, the last argument is called "sccomps" for "score components", note the plural, which allows us to use both velocity and energy (for example). In your case, there is is really just one component: nino3, which was computed by self.score_components. So there's no combining to be done. Note that Lorenz96 returns a list (with one element); you have the freedom to simplify and return a single timeseries. Remember, you just need to make sure the score function is called in a consistent way in CaneZebiak and here. Think of algorithms_cane_zebiak.py and cane_zebiak_sp.py as people connected by cell phone; the abstract stuff in ../../algorithms.py and ../../ensemble.py is just the satellite, which transmits the message while staying as agnostic as possible to the details of the speech.
        """
        return nino3
    def generate_icandf_from_parent(self, parent, branch_time):
        # TODO
        """
        This is the crux of implementing TEAMS for a specific model. The returned icandf dictionary should look similar to what you made with generate_default_icandf(), but with some TEAMS-specific tweaks. "branch_time" is determined externally by the algorithms.TEAMS abstract class (see ../algorithms.TEAMS.take_next_step), and is model-independent. But how to implement that with restart files and seeds is model-specific. For example, consider the pathological case described above where I moteivated the parameter "inherit_perts_after_split". In that case, you don't actually branch from the parent, but rather from the grandparent! I recommend referring to ../../algorithms.SDETEAMS.generate_icandf_from_parent, a specific implementation of the method for stochastic differential equations (which is used for Lorenz-96). Of course Cane-Zebiak is not an SDE, but the feature of restarts-every-timestep makes the SDE case a good analog in this specific aspect. 

        Note that depending on the choice of inherit_perts_after_timestep, you may need to get more involved with specifying random number generation seeds within python, and encoding this information into icandf. 
        """
        pass

def run_teams(dirdict,filedict,config_model,config_algo):
    """
    See ../lorenz96/teams_lorenz96
    """
    # TODO: root_dir = ...
    # TODO: base_dir = ...
    # TODO: alg_file = ... (pickle file where to store the algorithm object---including the encapsulated Ensemble and CaneZebiak objects---between steps. I suggest making this somewhere within ase_dir)
    # 
    if exists(alg_file):
        alg = pickle.load(open(alg_file, 'rb'))
        alg.set_capacity(config_algo['num_levels_max'], config_algo['num_members_max']) # in case we decide to extend a TEAMS run for more rounds of splitting after one go
    else:
        cz = CaneZebiak(config_gcm, recompile=recompile)
        ens = ensemble.Ensemble(cz, root_dir=root_dir)
        alg = algorithms_cane_zebiak.CaneZebiakTEAMS.initialize_from_dns_files(base_dir, config_algo, ens)

    alg.ens.set_root_dir(root_dir)
    while not alg.terminate:
        mem = alg.ens.get_nmem()
        print(f'----------- Starting member {mem} ----------------')
        saveinfo = dict({
            # TODO
            })
        alg.take_next_step(saveinfo)
        pickle.dump(alg, open(alg_file, 'wb'))
    return
