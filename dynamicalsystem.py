from abc import ABC, abstractmethod
import numpy as np
import forcing as fc

class DynamicalSystem(ABC):
    def __init__(self):
        return
    @abstractmethod
    def run_trajectory(self, forcing, observables):
        # return some metadata sufficient to reconstruct the output, for example (1) a filename, (2) full numpy array of the output of an ODE solver. 
        # Optionally, return some observables passed as a dictionary of function handles
        pass 


class ODESystem(DynamicalSystem):
    def __init__(self, state_dim, config, *args, **kwargs):
        self.state_dim = state_dim
        self.config = config
        self.params = self.derive_parameters(config) # This includes both physical and simulation parameters
        return
    @abstractmethod
    def derive_parameters(self, config):
        # convert raw configuration into class attributes for efficient integration of dynamics
        pass
    @abstractmethod
    def tendency(self, t, x):
        pass
    def run_trajectory_unperturbed(self, init_time, fin_time, observables, method):
        tarr_solve = np.arange(init_time, term_time+2, self.dt_step)
        tarr_save = np.arange(int(np.ceil(init_time)), int(fin_time)+1)
        Nt_solve = len(tarr_solve)
        Nt_save = len(tarr_save)
        # calculate weights from tarr_solve to tarr_save 
        weights = np.zeros((Nt_save,2))
        idx_lower = np.where(np.diff(np.floor(tarr_solve)) == 1)[0]
        if Nt_save-1 in idx_lower:
            idx_lower = idx_lower[:-1]
        weights[:-1,1] = (tarr_save - tarr_solve[idx_lower[:-1] 
        Nt = len(tarr)
        idx2save_lower = np.where(np.diff(np.floor(tarr) == 1))[0]
        idx2save_upper = idx2save_upper + 1
        indicator_idx2save_upper = np.zeros(Nt)
        indicator_idx2save_upper[idx2save_upper] = 1
        tarr_save = np.floor(idx_save_upper)
        lower_frac = (tarr_save - tarr[idx2save_lower]) / (tarr[idx2save_upper] - tarr[idx2save_lower])
        upper_frac = 1 - lower_frac

        Nt = len(tarr)
        xarr = np.zeros((Nt, self.state_dim))
        x = init_cond
        t = init_time
        x[0] = x
        if method == 'rk4':
            for i_t in range(1,Nt):
                k1 = self.dt * self.tendency(t,x)
                k2 = self.dt * self.tendency(t+self.dt/2, x+k1/2)
                k3 = self.dt * self.tendency(t+self.dt/2, x+k2/2)
                k4 = self.dt * self.tendency(t+self.dt, x+k3)
                xnew = x + (k1 + 2*(k2 + k3) + k4)/6
                if indicator_idx2save_upper[i_t]:
                    x[



    def run_trajectory(self, force, observables, method):
        # Special cases 
        t = np.arange(force.init_time, force.term_time+1, 1)
        Nt = len(t)
        x = np.zeros((Nt, self.state_dim))
        x[0] = force.init_cond
        if isinstance(force, forcing.ImpulsiveForcing):
            # Integrate piecewise from one input forcing to the next
            if 
            

class Lorenz96(ODESystem):
    def derive_parameters(self, config):
        self.K = config['K']
        self.F = config['F']
        self.dt_step = config['dt_step']
        self.dt_save = config['dt_save']
        self.time_unit = config['time_unit']
        return
    def tendency(self, x):
        return np.roll(x,1) * (np.roll(x, -1) - np.roll(x,2)) - x + self.F
    




