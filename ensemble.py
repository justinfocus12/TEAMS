from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
import pickle
from os.path import join, exists
from os import makedirs
from matplotlib import pyplot as plt, rcParams
rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)


class Ensemble(ABC): 
    # TODO decide how to handle trajectory reloading, etc. Should this be implemented in inherited Ensemble objects, or in Manager objects?
    def __init__(self, dynsys, root_dir):
        self.dynsys = dynsys
        self.memgraph = nx.DiGraph() 
        self.traj_metadata = [] 
        self.root_dir = root_dir
        return
    def set_root_dir(self, root_dir):
        # For example, after we've moved the whole ensemble to somewhere else 
        self.root_dir = root_dir
        return
    def branch_or_plant(self, icandf, obs_fun, saveinfo, parent=None):
        # TODO in case icandf has no explicit initial condition, pull the state from the corresponding parent at the initial time given in f
        metadata,observables = self.dynsys.run_trajectory(icandf, obs_fun, saveinfo, self.root_dir)
        self.traj_metadata.append(metadata)
        newmem = self.memgraph.number_of_nodes()
        self.memgraph.add_node(newmem)
        if parent is not None:
            self.memgraph.add_edge(parent, newmem)
        return observables
    def get_nmem(self):
        return self.memgraph.number_of_nodes()
    def get_member_timespan(self, member):
        return self.dynsys.get_timespan(self.traj_metadata[member])
    def get_all_timespans(self,mems=None):
        if mems is None:
            mems = np.arange(self.get_nmem())
        all_timespans = np.array([np.array(self.get_member_timespan(mem)) for mem in mems])
        return all_timespans[:,0],all_timespans[:,1]
    def get_first_forcing_times(self,mems=None):
        # Assume all mems have at least one forcing time 
        if mems is None:
            mems = np.arange(self.get_nmem())
        return np.array([self.traj_metadata[mem]['icandf']['frc'].get_forcing_times()[0] for mem in mems])
    def compute_pairwise_observables(self, pair_funs, mem0, mem1list):
        md0 = self.traj_metadata[mem0]
        md1list = [self.traj_metadata[mem1] for mem1 in mem1list]
        return self.dynsys.compute_pairwise_observables(pair_funs, md0, md1list, self.root_dir)
        
    def compute_observables(self, obs_funs, mems):
        # args_dict and kwargs_dict should have a different value for each obs_name
        obs_names = list(obs_funs.keys())
        obs_dict = dict({obs_name: [] for obs_name in obs_names})
        for i_mem,mem in enumerate(mems):
            metadata = self.traj_metadata[mem]
            obs_dict_mem = self.dynsys.compute_observables(obs_funs, metadata, self.root_dir)
            for obs_name in obs_names:
                obs_dict[obs_name].append(obs_dict_mem[obs_name])
        return obs_dict

    def compute_observables_along_lineage(self, obs_names, mem_leaf):
        mems = sorted(nx.ancestors(self.memgraph, mem_leaf) | {mem_leaf})
        obs_dict = self.compute_observables(obs_names, mems)
        return obs_dict # leave the explicit concatenation to later

    # --------------- Plotting methods ---------------
    def plot_observables(self, mems, ts, obsvals):
        # TODO also make an alternative method to compute observable functions
        tu = self.dynsys.dt_save
        colors = plt.cm.rainbow(np.arange(len(mems))/max(1,len(mems)-1))
        fig,ax = plt.subplots(figsize=(12,5))
        handles = []
        for i_mem,mem in enumerate(mems):
            t = ts[i_mem]
            obs = obsvals[i_mem]
            h, = ax.plot(t*tu, obs, label=f"Member {i_mem}", color=colors[i_mem])
            handles.append(h)
            # Plot origin time
            ax.scatter(t[0]*tu, obs[0], marker='o', color=colors[i_mem])
            # plot forcing times (plus one, since we didn't save initial conditions; add this as possible modification)
            tf = np.array(self.traj_metadata[mem]['icandf']['frc'].get_forcing_times()) + 1 
            print(f'{tf = }')
            i_tf = tf - t[0]
            print(f'{i_tf = }, {obs[i_tf] = }')
            ax.scatter(tf*tu, obs[i_tf], marker='x', color=colors[i_mem])
            ax.set_xlabel("Time")
        ax.legend(handles=handles, loc=(1,0))
        return fig,ax
