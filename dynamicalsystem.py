from abc import ABC, abstractmethod
import numpy as np
import forcing as fc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)

class DynamicalSystem(ABC):
    def __init__(self):
        return
    #@abstractmethod
    #def run_trajectory(self, forcing, observables):
    #    # return some metadata sufficient to reconstruct the output, for example (1) a filename, (2) full numpy array of the output of an ODE solver. 
    #    # Optionally, return some observables passed as a dictionary of function handles
    #    pass 


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
    def run_trajectory_unperturbed(self, init_cond, init_time, fin_time, method):
        t_save = np.arange(int(np.ceil(init_time)), int(np.floor(fin_time))+1, 1) # unitless
        tp_save = t_save * self.dt_save # physical (unitful)
        Nt_save = len(t_save)
        # Initialize the solution array
        x_save = np.zeros((Nt_save, self.state_dim))
        # special cases: endpoints
        if init_time % 1 == 0:
            x_save[0] = init_cond
            i_save = 1
        else:
            i_save = 0
        tp_save_next = tp_save[i_save]
        x = init_cond
        tp = init_time * self.dt_save # physical units
        if method == 'rk4':
            while tp < tp_save[-1]:
                k1 = self.dt_step * self.tendency(tp,x)
                k2 = self.dt_step * self.tendency(tp+self.dt_step/2, x+k1/2)
                k3 = self.dt_step * self.tendency(tp+self.dt_step/2, x+k2/2)
                k4 = self.dt_step * self.tendency(tp+self.dt_step, x+k3)
                xnew = x + (k1 + 2*(k2 + k3) + k4)/6
                tpnew = tp + self.dt_step
                if tpnew > tp_save_next:
                    new_weight = (tp_save_next - tp)/self.dt_step 
                    x_save[i_save] = (1-new_weight)*x + new_weight*xnew 
                    i_save += 1
                    if i_save < Nt_save:
                        tp_save_next = tp_save[i_save]
                x = xnew
                tp = tpnew
        return t_save,x_save

class Lorenz96(ODESystem):
    def __init__(self, config):
        state_dim = config['K']
        super().__init__(state_dim, config)
    def derive_parameters(self, config):
        self.K = config['K']
        self.F = config['F']
        self.dt_step = config['dt_step'] # physical
        self.dt_save = config['dt_save'] # physical
        # all time arrays will have integers as entries, for unambiguous time alignment
        return
    def tendency(self, t, x):
        return np.roll(x,1) * (np.roll(x, -1) - np.roll(x,2)) - x + self.F
    


def test_Lorenz96():
    config = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05})
    ode = Lorenz96(config)
    tu = ode.dt_save
    init_cond = 0.001*np.random.randn(config['K'])
    init_time = -4.0/tu
    fin_time = 15.0/tu
    method = 'rk4'
    t_save,x_save = ode.run_trajectory_unperturbed(init_cond, init_time, fin_time, method)

    fig,axes = plt.subplots(nrows=2,figsize=(10,10), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(t_save*tu, x_save[:,0])
    ax.set_xlabel('time')
    ax.set_ylabel('x0')

    ax = axes[1]
    im = ax.pcolormesh(t_save*tu, np.arange(ode.K)[::-1], x_save.T, shading='nearest', cmap='BrBG')
    ax.set_xlabel('time')
    ax.set_ylabel('Longitude $k$')
    fig.savefig('test_Lorenz96', **pltkwargs)
    plt.close(fig)
    return t_save, x_save

if __name__ == "__main__":
    test_Lorenz96()



