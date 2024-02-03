from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
import pickle
from os.path import join, exists
from os import makedirs


class Ensemble(ABC): 
    # TODO decide how to handle trajectory reloading, etc. Should this be implemented in inherited Ensemble objects, or in Manager objects?
    def __init__(self, savedir, dynsys, *args, **kwargs):
        self.savedir = savedir # This is for ensemble-level metadata only; trajectory-level data may have to go elsewhere.
        self.dynsys = dynsys
        makedirs(self.savedir, exist_ok=True)
        self.memgraph = nx.DiGraph() 
        self.traj_metadata = [] 
        return
    def branch_or_plant(self, icandf, obs_fun, saveinfo, parent=None):
        metadata,observables = self.dynsys.run_trajectory(icandf, obs_fun, saveinfo)
        self.traj_metadata.append(metadata)
        newmem = self.memgraph.number_of_nodes()
        self.memgraph.add_node(newmem)
        if parent is not None:
            self.memgraph.add_edge(parent, newmem)
        self.save_state()
        return observables
    def save_state(self):
        pickle.dump(self, open(join(self.savedir, 'ens.pickle'),'wb'))
        return
