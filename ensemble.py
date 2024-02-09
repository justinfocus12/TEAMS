from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
import pickle
from os.path import join, exists
from os import makedirs
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)


class Ensemble(ABC): 
    # TODO decide how to handle trajectory reloading, etc. Should this be implemented in inherited Ensemble objects, or in Manager objects?
    def __init__(self, dynsys):
        self.dynsys = dynsys
        self.memgraph = nx.DiGraph() 
        self.traj_metadata = [] 
        return
    def branch_or_plant(self, icandf, obs_fun, saveinfo, parent=None):
        # TODO in case icandf has no explicit initial condition, pull the state from the corresponding parent at the initial time given in f
        metadata,observables = self.dynsys.run_trajectory(icandf, obs_fun, saveinfo)
        self.traj_metadata.append(metadata)
        newmem = self.memgraph.number_of_nodes()
        self.memgraph.add_node(newmem)
        if parent is not None:
            self.memgraph.add_edge(parent, newmem)
        return observables
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
            # plot forcing times
            tf = np.array(self.traj_metadata[mem]['icandf']['frc'].get_forcing_times())
            i_tf = tf - t[0]
            ax.scatter(tf*tu, obs[i_tf], marker='x', color=colors[i_mem])
            ax.set_xlabel("Time")
        ax.legend(handles=handles, loc=(1,0))
        return fig,ax
