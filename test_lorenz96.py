import numpy as np
from numpy.random import default_rng
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs
import copy as copylib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
from lorenz96 import Lorenz96
from ensemble import Ensemble
import forcing

def test_Lorenz96_white():
    config = dict({'K': 40, 'F': 8.0, 'dt_step': 0.001, 'dt_save': 0.05})
    config['frc'] = dict({
        'type': 'white',
        'white': dict({
            'wavenumbers': [4],
            'wavenumber_magnitudes': [0.25],
            'sites': [],
            'site_magnitudes': [],
            }),
        })
    ode = Lorenz96(config)
    tu = ode.dt_save

    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-02"
    sub_date_str = "0"
    param_abbrv,param_label = ode.label_from_config(config)
    ensdir = join(scratch_dir, date_str, sub_date_str, param_abbrv)

    ens = Ensemble(ensdir, ode)

    seed_times_phys = [-4.0, 5.0, 15.0]
    seed_times = [int(stp/tu) for stp in seed_times_phys]
    fin_time_phys = 30.0
    fin_time = int(fin_time_phys/tu)
    seeds = [8764, 3910, 6789]
    obs_fun = lambda t,x: (t,x[:,0]**2,np.sum(x**2, axis=1)) # local and global energy
    obs_vals = []

    # 0
    icandf = dict({
        'init_cond': 0.001*np.sin(2*np.pi*np.arange(config['K'])/config['K']), 
        'frc': forcing.WhiteNoiseForcing(seed_times[:1], seeds[:1], fin_time)
        })
    saveinfo = dict(filename = join(ensdir, 'mem0.npz'))
    obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo)
    obs_vals.append(copylib.copy(obs_val_new))

    # 1 
    parent = 0
    icandf = dict({
        'init_cond': 0.001*np.sin(2*np.pi*np.arange(config['K'])/config['K']), 
        'frc': forcing.WhiteNoiseForcing(seed_times[:2], seeds[:2], fin_time)
        })
    saveinfo = dict(filename = join(ensdir, 'mem1.npz'))
    obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
    obs_vals.append(copylib.copy(obs_val_new))

    # 2 
    # This time branch directly off the parent
    # TODO devise a more elegant solution to the problem of branching from a specific point
    parent = 0
    t_parent,x_parent = ode.load_trajectory(ens.traj_metadata[parent], tspan=[seed_times[1]]*2)
    print(f"{t_parent = }; {seed_times = }")
    icandf.update(init_cond=x_parent, frc=forcing.WhiteNoiseForcing(seed_times[1:3], seeds[1:3], fin_time))
    icandf = dict({
        'init_cond': x_parent,
        'frc': forcing.WhiteNoiseForcing(seed_times[1:3], seeds[1:3], fin_time)
        })
    saveinfo = dict(filename = join(ensdir, 'mem2.npz'))
    obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
    obs_vals.append(copylib.copy(obs_val_new))

    # Do all three trajectories match? 
    t0,x0 = ode.load_trajectory(ens.traj_metadata[0])
    t1,x1 = ode.load_trajectory(ens.traj_metadata[1])
    t2,x2 = ode.load_trajectory(ens.traj_metadata[2])

    x0_neq_x1 = np.where(np.any(x0 != x1, axis=1))[0]
    if len(x0_neq_x1) > 0:
        print(f"{t0[x0_neq_x1[0]]*tu = }")
    else:
        print(f"x1 never diverges from x0")

    print(f"{t0[[0,-1]] = }; {t1[[0,-1]] = }; {t2[[0,-1]] = };")
    overlap_12 = np.argmax(t1 == t2[0])
    x1_neq_x2 = np.where(np.any(x1[overlap_12:] != x2, axis=1))[0]
    if len(x1_neq_x2) > 0:
        print(f"{t2[x1_neq_x2[0]]*tu = }")
    else:
        print(f"x2 never diverges from x1")


    return

def test_Lorenz96_impulsive():

    tododict = dict({
        'run_ensemble':          1,
        'check_overlap':         1,
        'plot_ensemble':         1,
        })

    config = dict({'K': 40, 'F': 6.0, 'dt_step': 0.001, 'dt_save': 0.05})
    config['frc'] = dict({
        'type': 'impulsive',
        'impulsive': dict({
            'wavenumbers': [4],
            'wavenumber_magnitudes': [0.1,0.1],
            'sites': [20,30],
            'site_magnitudes': [0.5, 0.5],
            }),
        })
    scratch_dir = "/net/hstor001.ib/pog/001/ju26596/TEAMS_results/examples/lorenz96"
    date_str = "2024-02-02"
    sub_date_str = "0"
    param_abbrv,param_label = Lorenz96.label_from_config(config)
    ensdir = join(scratch_dir, date_str, sub_date_str, param_abbrv)

    if tododict['run_ensemble']: 

        ode = Lorenz96(config)
        tu = ode.dt_save

        ens = Ensemble(ensdir, ode)

        imp_times_phys = [-4.0, 5.0, 25.0]
        imp_times = [int(itp/tu) for itp in imp_times_phys]
        fin_time_phys = 40.0
        fin_time = int(fin_time_phys/tu)
        rng = default_rng(8764)
        impulses = rng.normal(size=(len(imp_times), ode.impulse_dim))
        # TODO structure the obs_fun more like a dataframe to be more descriptive on our observables. 
        obs_fun = lambda t,x: (t,x[:,0], x[:,0]**2,np.sum(x**2, axis=1)) # local and global energy
        obs_vals = []

        # 0
        icandf = dict({
            'init_cond': 0.001*np.sin(2*np.pi*np.arange(config['K'])/config['K']), 
            'frc': forcing.ImpulsiveForcing(imp_times[:1], impulses[:1], fin_time)
            })
        saveinfo = dict(filename = join(ensdir, 'mem0.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo)
        obs_vals.append(copylib.copy(obs_val_new))

        # 1 
        parent = 0
        icandf = dict({
            'init_cond': 0.001*np.sin(2*np.pi*np.arange(config['K'])/config['K']), 
            'frc': forcing.ImpulsiveForcing(imp_times[:2], impulses[:2], fin_time)
            })
        saveinfo = dict(filename = join(ensdir, 'mem1.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        obs_vals.append(copylib.copy(obs_val_new))

        # 2 
        # This time branch directly off the parent
        # TODO devise a more elegant solution to the problem of branching from a specific point
        parent = 0
        t_parent,x_parent = ode.load_trajectory(ens.traj_metadata[parent], tspan=[imp_times[1]]*2)
        print(f"{t_parent = }; {imp_times = }")
        icandf = dict({
            'init_cond': x_parent,
            'frc': forcing.ImpulsiveForcing(imp_times[1:3], impulses[1:3], fin_time)
            })
        saveinfo = dict(filename = join(ensdir, 'mem2.npz'))
        obs_val_new = ens.branch_or_plant(icandf, obs_fun, saveinfo, parent=parent)
        obs_vals.append(copylib.copy(obs_val_new))

    # Do all three trajectories match? 
    t0,x0 = ode.load_trajectory(ens.traj_metadata[0])
    t1,x1 = ode.load_trajectory(ens.traj_metadata[1])
    t2,x2 = ode.load_trajectory(ens.traj_metadata[2])

    x0_neq_x1 = np.where(np.any(x0 != x1, axis=1))[0]
    if len(x0_neq_x1) > 0:
        print(f"{t0[x0_neq_x1[0]]*tu = }")
    else:
        print(f"x1 never diverges from x0")

    print(f"{t0[[0,-1]] = }; {t1[[0,-1]] = }; {t2[[0,-1]] = };")
    overlap_12 = np.argmax(t1 == t2[0])
    x1_neq_x2 = np.where(np.any(x1[overlap_12:] != x2, axis=1))[0]
    if len(x1_neq_x2) > 0:
        print(f"{t2[x1_neq_x2[0]]*tu = }")
    else:
        print(f"x2 never diverges from x1")

    # Plot the observables
    mems = [0,1,2]
    ts = [ov[0] for ov in obs_vals]
    obsvals2plot = [ov[1] for ov in obs_vals]
    fig,axes = ens.plot_observables(mems, ts, obsvals2plot)
    fig.savefig(join(ensdir, "obs0.png"), **pltkwargs)
    plt.close(fig)
    return

if __name__ == "__main__":
    test_Lorenz96_impulsive()
