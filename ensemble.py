from abc import ABC, abstractmethod
import numpy as np
import networkx as nx


class Ensemble(ABC):
    def __init__(self, savedir, dynsys, *args, **kwargs):
        self.savedir = savedir # This is for ensemble-level metadata only; trajectory-level data may have to go elsewhere.
        self.dynsys = dynsys
        os.makedirs(self.savedir, exist_ok=False)
        self.memgraph = nx.DiGraph() 
        self.traj_metadata = [] 
        return
    def branch_or_plant(self, forcing, parent=None):
        metadata,observables = self.dynsys.run_trajectory(forcing)
        self.traj_metadata.append(metadata)
        newmem = self.memgraph.number_of_nodes()
        self.memgraph.add_node(newmem)
        if parent is not None:
            self.memgraph.add_edge(parent, newmem)
        return observables
