
import numpy as np
from numpy.random import default_rng
from numba import njit
from numba.typed import Dict
from numba.core import types 
from scipy import sparse as sps
from os.path import join, exists
from os import makedirs

import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.size": 15
})
pltkwargs = dict(bbox_inches="tight",pad_inches=0.2)
sys.path.append("../..")
from dynamicalsystem import ODESystem,SDESystem
import forcing
from ensemble import Ensemble
import utils


# --------------- JIT functions ---------
@njit
def integrate_monotone_external(
        # intent(out)
        state_save, 
        # intent(in)
        tp_save, dt_step_max,
        init_time_phys,
        init_cond,
        flowdim,Nparticles,Nx,Ny,dx,dy,Lx,Ly,b,
        forcing_term,linear_term,bilinear_term,
        gamma_t,gamma_tilde_t, # for simplicity, we will have no orography cycle 
        basis_u,basis_v,
        source_flag,source_mean,source_amplitude,source_period,sink,
        s_rk4_1,s_rk4_2,s_rk4_3,s_rk4_4,
        s_next_temp,
        s_tendency_total,
        u_eul,v_eul,iflat_nbs,outflows,inflows,
        flux_u_eul,flux_v_eul,
        flux_coefs_center,flux_coefs_right,flux_coefs_left,flux_coefs_up,flux_coefs_down,
        s_lag,u_lag,v_lag,death_flag,
        c1x_lag,s1x_lag,c1y_lag,s1y_lag,c2y_lag,s2y_lag,
        ):
    i_save = 0
    Nt_save = len(tp_save)
    tp_save_next = tp_save[i_save]
    state = np.zeros(len(init_cond))
    state[:] = init_cond
    state_next = np.zeros(len(init_cond))
    tp = init_time_phys 

    timestep_counter = 0
    
    while tp < tp_save[-1]:
        verbose = 0 #1*(i_save <= 2)
        #print(f'{tp = }')
        dt_step = timestep_monotone_external(
                state_next,
                tp, dt_step_max,
                state,
                flowdim,Nparticles,Nx,Ny,dx,dy,Lx,Ly,b,
                forcing_term,linear_term,bilinear_term,
                gamma_t,gamma_tilde_t,
                basis_u,basis_v,
                source_flag,source_mean,source_amplitude,source_period,sink,
                s_rk4_1,s_rk4_2,s_rk4_3,s_rk4_4,
                s_next_temp,
                s_tendency_total,
                u_eul,v_eul,iflat_nbs,outflows,inflows,
                flux_u_eul,flux_v_eul,
                flux_coefs_center,flux_coefs_right,flux_coefs_left,flux_coefs_up,flux_coefs_down,
                s_lag,u_lag,v_lag,death_flag,
                c1x_lag,s1x_lag,c1y_lag,s1y_lag,c2y_lag,s2y_lag,
                verbose,
                )
        #d_xlag = flowdim + Nx*Ny + np.arange(Nparticles, dtype=int)
        #if verbose == 1:
        #    print(np.max(np.abs(state_next[d_xlag] - state[d_xlag])))
        #print(f'{dt_step = }')
        tpnew = tp + dt_step
        timestep_counter += 1

        if tpnew > tp_save_next:
            #print(f'--------------------------- {i_save = }-----------------')
            #print(f'{timestep_counter = }')
            new_weight = (tp_save_next - tp)/dt_step 
            #print(f'{dt_step = }')
            # TODO: save out an observable instead of the full state? Depends on a specified frequency
            state_save[i_save,:] = (1-new_weight)*state + new_weight*state_next 
            #if i_save >= 1:
            #    print(np.max(np.abs(state_save[i_save,d_xlag] - state_save[i_save-1,d_xlag])))
            i_save += 1
            if i_save < Nt_save:
                tp_save_next = tp_save[i_save]
            timestep_counter = 0
        state[:] = state_next
        tp = tpnew
    return



@njit
def timestep_monotone_external(
    # intent(out)
    state_next, 
    # intent(in)
    t, dt_max,
    state,
    flowdim,Nparticles,Nx,Ny,dx,dy,Lx,Ly,aspect,
    forcing_term,linear_term,bilinear_term,
    gamma,gamma_tilde,
    basis_u, basis_v, 
    source_flag,source_mean,source_amplitude,source_period,sink,
    # intent(inout)
    # for streamfunction coefficient update
    s_rk4_1,s_rk4_2,s_rk4_3,s_rk4_4,
    s_next_temp,
    s_tendency,
    # for concentration update
    u_eul,v_eul,iflat_nbs,outflows,inflows,
    flux_u_eul,flux_v_eul,
    flux_coefs_center,flux_coefs_right,flux_coefs_left,flux_coefs_up,flux_coefs_down,
    # for Lagrangian advection update 
    s_lag,u_lag,v_lag,death_flag,
    c1x,s1x,c1y,s1y,c2y,s2y,
    verbose,
    ):

    # Make convenience aliases for the components of the state vector
    s = state[0:flowdim]
    s_next = state_next[0:flowdim]
    conc = state[flowdim:flowdim+Nx*Ny]
    conc_next = state_next[flowdim:flowdim+Nx*Ny]
    x_lag = state[(flowdim+Nx*Ny):(flowdim+Nx*Ny+Nparticles)]
    y_lag = state[(flowdim+Nx*Ny+Nparticles):(flowdim+Nx*Ny+2*Nparticles)]
    x_lag_next = state_next[(flowdim+Nx*Ny):(flowdim+Nx*Ny+Nparticles)]
    y_lag_next = state_next[(flowdim+Nx*Ny+Nparticles):(flowdim+Nx*Ny+2*Nparticles)]

    # Determine the CFL condition
    u_eul[:,:] = 0.0
    v_eul[:,:] = 0.0
    for i in range(flowdim):
        u_eul[:,:] += s[i] * basis_u[i,:,:]
        v_eul[:,:] += s[i] * basis_v[i,:,:]
    max_abs_u = np.max(np.abs(u_eul))
    max_abs_v = np.max(np.abs(v_eul))
    dt = min(dt_max, min(dx,dy)/max(max_abs_u,max_abs_v)) / 4 # ends up being 0.00625
    #print(dt) 


    # ------------- Update flow field with Runge-Kutta ------------
    const_args = (flowdim,forcing_term,linear_term,bilinear_term,gamma,gamma_tilde)
    compute_streamfunction_tendency_external(
            s_rk4_1, t, s, 
            *const_args
            )
    s_next_temp[:] = s + (dt/2)*s_rk4_1
    compute_streamfunction_tendency_external(
            s_rk4_2, t+dt/2, s_next_temp,
            *const_args
            )
    s_next_temp[:] = s + (dt/2)*s_rk4_2
    compute_streamfunction_tendency_external(
            s_rk4_3, t+dt/2, s_next_temp, 
            *const_args
            )
    s_next_temp[:] = s + dt*s_rk4_3
    compute_streamfunction_tendency_external(
            s_rk4_4, t+dt, s_next_temp,
            *const_args
            )
    s_next[:] = s + (dt/6)*(s_rk4_1 + 2*(s_rk4_2 + s_rk4_3) + s_rk4_4)
    # --------------- Update concentration field with finite-volume ---------
    conc_next[:] = 0.0
    u_eul[:,:] = 0.0
    v_eul[:,:] = 0.0
    for i in range(flowdim):
        u_eul[:,:] += 0.5*(s[i] + s_next[i]) * basis_u[i,:,:]
        v_eul[:,:] += 0.5*(s[i] + s_next[i]) * basis_v[i,:,:]
    # Compute zonal fluxes
    flux_coefs_center[:,:] = 0.0
    for ix in range(Nx+1):
        ixlo,ixhi = (ix-1)%Nx, ix%Nx
        for iy in range(Ny):
            iylo,iyhi = iy,iy
            iflat_lo = ixlo*Ny + iylo
            iflat_hi = ixhi*Ny + iyhi
            wlo = 1*(u_eul[ix,iy] > 0)
            whi = 1*(u_eul[ix,iy] <= 0)
            flo = wlo*u_eul[ix,iy]*dt/dx
            fhi = whi*u_eul[ix,iy]*dt/dx
            flux_u_eul[ix,iy] = flo*conc[iflat_lo] + fhi*conc[iflat_hi]
            #flux_coefs_center[ixlo,iylo] -= flo
            flux_coefs_right[ixlo,iylo] = -fhi
            flux_coefs_left[ixhi,iyhi] = flo
            flux_coefs_center[ixhi,iyhi] += fhi
    # Compute meridional fluxes
    for ix in range(Nx):
        ixlo,ixhi = ix,ix
        for iy in range(1,Ny):
            iylo,iyhi = iy-1,iy
            iflat_lo = ixlo*Ny + iylo
            iflat_hi = ixhi*Ny + iyhi
            wlo = 1*(v_eul[ix,iy] > 0)
            whi = 1*(v_eul[ix,iy] <= 0)
            flo = wlo * v_eul[ix,iy]*dt/dy 
            fhi = whi * v_eul[ix,iy]*dt/dy 
            flux_v_eul[ix,iy] = flo*conc[iflat_lo] + fhi*conc[iflat_hi]
            #flux_coefs_center[ixlo,iylo] -= flo
            flux_coefs_up[ixlo,iylo] = -fhi
            flux_coefs_down[ixhi,iyhi] = flo
            flux_coefs_center[ixhi,iyhi] += fhi

    for iflat in range(Nx*Ny):
        ix = iflat // Ny
        iy = iflat - Ny*ix
        ix_right, iy_right = (ix+1)%Nx, iy
        ix_left, iy_left = (ix-1)%Nx, iy
        ix_up, iy_up = ix, iy+1
        ix_down, iy_down = ix, iy-1
        iflat_right = ix_right*Ny + iy_right
        iflat_left = ix_left*Ny + iy_left
        iflat_up = ix_up*Ny + iy_up
        iflat_down = ix_down*Ny + iy_down
        conc_next[iflat] = conc[iflat]
        conc_next[iflat] += flux_coefs_right[ix,iy] * (conc[iflat_right] - conc[iflat])
        conc_next[iflat] += flux_coefs_left[ix,iy] * (conc[iflat_left] - conc[iflat])
        if iy > 0:
            conc_next[iflat] += flux_coefs_down[ix,iy] * (conc[iflat_down] - conc[iflat])
        if iy < Ny-1:
            conc_next[iflat] += flux_coefs_up[ix,iy] * (conc[iflat_up] - conc[iflat])
        # source and sink 
        conc_next[iflat] *= np.exp(-sink*dt)
        if source_flag[ix,iy]:
            conc_next[iflat] += (source_mean + source_amplitude*np.cos(2*np.pi*t/source_period))/sink*(-np.expm1(-sink*dt))
    # --------- Forward Euler for particles -------------
    compute_streamfunction_lagrangian_external(
            s_lag, u_lag, v_lag, 
            t, s, x_lag, y_lag,
            aspect,
            c1x,s1x,c1y,s1y,c2y,s2y
            )
    #print(f'{np.max(np.abs(dt*u_lag)) = }')
    #print(f'{np.max(np.abs(dt*v_lag)) = }')
    x_lag_next[:] = np.mod(x_lag + dt*u_lag, Lx)
    y_lag_next[:] = y_lag + dt*v_lag

    #if verbose == 1:
    #    print(np.max(np.abs(x_lag_next - x_lag)))

    # ---------------- DEBUGGING ------------------- 
    #ix_lag = np.mod((x_lag / dx).astype(int), Nx)
    #x_rem = np.mod(x_lag, dx)/dx
    #iy_lag = np.mod((y_lag / dy).astype(int), Ny)
    #y_rem = np.mod(y_lag, dy)/dy

    #u_lag_eul = (1-x_rem)*u_eul[ix_lag,iy_lag] + x_rem*u_eul[ix_lag+1,iy_lag]
    #v_lag_eul = (1-y_rem)*v_eul[ix_lag,iy_lag] + y_rem*v_eul[ix_lag,iy_lag+1]

    #ustacked = np.array([u_lag, u_lag_eul])
    #vstacked = np.array([v_lag, v_lag_eul])

    #print('ustacked = ')
    #print(ustacked)
    #print('vstacked = ')
    #print(vstacked)


    # ----------------------------------------------

    # Account for sources and sinks 
    death_flag[:] = False # TODO implement 
    #np.logical_or(y_lag_next < source_width, y_lag_next > Ly-source_width) 
    #if np.any(death_flag):
    #    death_idx = np.where(death_flag)[0]
    #    y_lag_next[death_idx] = source_width
    #    x_lag_next[death_idx] = Lx * np.random.rand(len(death_idx))
    return dt #t+dt_step, np.concatenate((strfn_next, conc_next, x_lag_next, y_lag_next))

@njit
def timestep_finvol_external(
    # intent(out)
    state_next, 
    # intent(in)
    t, dt_max,
    state,
    flowdim,Nparticles,Nx,Ny,dx,dy,Lx,Ly,aspect,
    forcing_term,linear_term,bilinear_term,
    gamma,gamma_tilde,
    basis_u, basis_v, source_flag, source_conc, source_width,
    # intent(inout)
    # for streamfunction coefficient update
    s_rk4_1,s_rk4_2,s_rk4_3,s_rk4_4,
    s_next_temp,
    s_tendency,
    # for concentration update
    u_eul,v_eul,iflat_nbs,outflows,inflows,
    # for Lagrangian advection update 
    s_lag,u_lag,v_lag,death_flag,
    c1x,s1x,c1y,s1y,c2y,s2y,
    ):

    # Make convenience aliases for the components of the state vector
    s = state[0:flowdim]
    s_next = state_next[0:flowdim]
    conc = state[flowdim:flowdim+Nx*Ny]
    conc_next = state_next[flowdim:flowdim+Nx*Ny]
    x_lag = state[(flowdim+Nx*Ny):(flowdim+Nx*Ny+Nparticles)]
    y_lag = state[(flowdim+Nx*Ny+Nparticles):(flowdim+Nx*Ny+2*Nparticles)]
    x_lag_next = state_next[(flowdim+Nx*Ny):(flowdim+Nx*Ny+Nparticles)]
    y_lag_next = state_next[(flowdim+Nx*Ny+Nparticles):(flowdim+Nx*Ny+2*Nparticles)]

    # Determine the CFL condition
    u_eul[:,:] = 0.0
    v_eul[:,:] = 0.0
    for i in range(flowdim):
        u_eul[:,:] += s[i] * basis_u[i,:,:]
        v_eul[:,:] += s[i] * basis_v[i,:,:]
    max_abs_u = np.max(np.abs(u_eul))
    max_abs_v = np.max(np.abs(v_eul))
    dt = min(dt_max, min(dx,dy)/max(max_abs_u,max_abs_v))


    # ------------- Update flow field with Runge-Kutta ------------
    const_args = (flowdim,forcing_term,linear_term,bilinear_term,gamma,gamma_tilde)
    compute_streamfunction_tendency_external(
            s_rk4_1, t, s, 
            *const_args
            )
    s_next_temp[:] = s + (dt/2)*s_rk4_1
    compute_streamfunction_tendency_external(
            s_rk4_2, t+dt/2, s_next_temp,
            *const_args
            )
    s_next_temp[:] = s + (dt/2)*s_rk4_2
    compute_streamfunction_tendency_external(
            s_rk4_3, t+dt/2, s_next_temp, 
            *const_args
            )
    s_next_temp[:] = s + dt*s_rk4_3
    compute_streamfunction_tendency_external(
            s_rk4_4, t+dt, s_next_temp,
            *const_args
            )
    s_next[:] = s + (dt/6)*(s_rk4_1 + 2*(s_rk4_2 + s_rk4_3) + s_rk4_4)
    # --------------- Update concentration field with finite-volume ---------
    conc_next[:] = conc
    u_eul[:,:] = 0.0
    v_eul[:,:] = 0.0
    for i in range(flowdim):
        u_eul[:,:] += 0.5*(s[i] + s_next[i]) * basis_u[i,:,:]
        v_eul[:,:] += 0.5*(s[i] + s_next[i]) * basis_v[i,:,:]
    for iflat in range(Nx*Ny):
        ix = iflat // Ny
        iy = iflat - ix*Ny 
        if source_flag[ix,iy]:
            conc_next[iflat] = source_conc[ix,iy]
            continue
        iflat_nbs[:] = 0
        outflows[:] = 0.0
        inflows[:] = 0.0
        # right
        i_nb = 0
        ix_nb = (ix+1) % Nx
        iy_nb = iy
        iflat_nb = ix_nb*Ny + iy_nb #np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
        iflat_nbs[i_nb] = iflat_nb
        velocity = u_eul[ix+1,iy]
        if velocity > 0:
            conc_mid = conc[iflat]
            outflows[i_nb] = conc_mid * dt*velocity/dx
        elif source_flag[ix_nb,iy_nb]:
            conc_mid = conc[iflat_nb]
            inflows[i_nb] = conc_mid * dt*(-velocity)/dx
        # left 
        i_nb = 1
        ix_nb = (ix-1) % Nx
        iy_nb = iy
        iflat_nb = ix_nb*Ny + iy_nb #np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
        iflat_nbs[i_nb] = iflat_nb
        velocity = u_eul[ix,iy]
        if velocity < 0:
            conc_mid = conc[iflat]
            outflows[i_nb] = conc_mid * dt*(-velocity)/dx
        elif source_flag[ix_nb,iy_nb]:
            conc_mid = conc[iflat_nb]
            inflows[i_nb] = conc_mid * dt*velocity/dx
        # top
        i_nb = 2
        ix_nb = ix
        iy_nb = iy+1
        iflat_nb = ix_nb*Ny + iy_nb #iflat_nb = np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
        iflat_nbs[i_nb] = iflat_nb
        velocity = v_eul[ix,iy+1]
        if velocity > 0:
            conc_mid = conc[iflat]
            outflows[i_nb] = conc_mid * dt*velocity/dy
        elif source_flag[ix_nb,iy_nb]:
            conc_mid = conc[iflat_nb]
            inflows[i_nb] = conc_mid * dt*(-velocity)/dy
        # bottom
        i_nb = 3
        ix_nb = ix
        iy_nb = iy-1
        iflat_nb = ix_nb*Ny + iy_nb #iflat_nb = np.ravel_multi_index((ix_nb,iy_nb),(Nx,Ny))
        iflat_nbs[i_nb] = iflat_nb
        velocity = v_eul[ix,iy]
        if velocity < 0:
            conc_mid = conc[iflat]
            outflows[i_nb] = conc_mid * dt*(-velocity)/dy
        elif source_flag[ix_nb,iy_nb]:
            conc_mid = conc[iflat_nb]
            inflows[i_nb] = conc_mid * dt*velocity/dy
        # Sum up all in- and out-flows
        conc_next[iflat] += np.sum(inflows)
        sum_outflows = np.sum(outflows)
        if sum_outflows > 0:
            if (sum_outflows > conc[iflat]):
                downweight = conc[iflat]/sum_outflows
                outflows *= downweight
                sum_outflows *= downweight
            conc_next[iflat] -= sum_outflows
            for i_nb in range(4):
                ix_nb = iflat_nbs[i_nb] // Ny
                iy_nb = iflat_nbs[i_nb] - Ny*ix_nb
                if not source_flag[ix_nb,iy_nb]:
                    conc_next[iflat_nbs[i_nb]] += outflows[i_nb]
    # Forward Euler for particles
    compute_streamfunction_lagrangian_external(
            s_lag, u_lag, v_lag, 
            t, s, x_lag, y_lag,
            aspect,
            c1x,s1x,c1y,s1y,c2y,s2y
            )
    x_lag_next[:] = np.mod(x_lag + dt*u_lag, Lx)
    y_lag_next[:] = y_lag + dt*v_lag

    # Account for sources and sinks 
    death_flag[:] = np.logical_or(y_lag_next < source_width, y_lag_next > Ly-source_width) 
    if np.any(death_flag):
        death_idx = np.where(death_flag)[0]
        y_lag_next[death_idx] = source_width
        x_lag_next[death_idx] = Lx * np.random.rand(len(death_idx))
    return dt #t+dt_step, np.concatenate((strfn_next, conc_next, x_lag_next, y_lag_next))

@njit
def compute_streamfunction_tendency_external(
        # intent(out)
        s_tendency, 
        # intent(in)
        t, s, 
        flowdim,
        forcing_term, linear_term, bilinear_term,
        gamma_t,gamma_tilde_t,
        ):
    # forcing
    s_tendency[:] = forcing_term 
    # dissipation
    s_tendency[:] += linear_term @ s
    s_tendency[0] += gamma_tilde_t[0]*s[2]
    s_tendency[2] -= gamma_t[0]*s[0]
    s_tendency[3] += gamma_tilde_t[1]*s[5]
    s_tendency[5] -= gamma_t[1]*s[3]
    # advection
    for j in range(flowdim):
        s_tendency[j] += np.sum(s * (bilinear_term[j] @ s))
    return
        

@njit
def compute_streamfunction_lagrangian_external(
        # intent(out)
        s_lag, u_lag, v_lag,
        # intent(in)
        t, s, x_lag, y_lag,
        aspect,
        # intent(inout)
        c1x,s1x,c1y,s1y,c2y,s2y,
        ):
    Nparticles = len(x_lag)
    sqrt2 = np.sqrt(2)
    b = aspect
    c1x[:] = np.cos(x_lag)
    s1x[:] = np.sin(x_lag)
    c1y[:] = np.cos(y_lag/b)
    s1y[:] = np.sin(y_lag/b)
    c2y[:] = np.cos(2*y_lag/b)
    s2y[:] = np.sin(2*y_lag/b)
    s_lag[:] = 0
    u_lag[:] = 0
    v_lag[:] = 0 

    # zonal-mean flow
    s_lag += b*sqrt2  *      (s[0]*c1y + s[3]*c2y)
    u_lag -= b*sqrt2 *       (s[0]*(-s1y/b) + s[3]*(-s2y*2/b))

    # Wave-1 in y direction
    s_lag += 2*b     *  s1y * (s[1]*c1x - s[2]*s1x)
    v_lag += 2*b   *  s1y * (s[1]*(-s1x) - s[2]*c1x)
    u_lag -= 2*b   *  c1y/b * (s[1]*c1x - s[2]*s1x)

    # Wave-2 in y direction
    s_lag += 2*b     *  s2y * (s[4]*c1x - s[5]*s1x)
    v_lag += 2*b   *  s2y * (s[4]*(-s1x) - s[5]*c1x)
    u_lag -= 2*b   *  c2y*(2/b) * (s[4]*c1x - s[5]*s1x)
    return 

        


class Crommelin2004TracerODE(ODESystem): 
    def __init__(self, cfg):
        self.state_dim = 6 + cfg["Nxfv"]*cfg["Nyfv"] + 2*cfg["Nparticles"]
        super().__init__(cfg)
    @staticmethod
    def default_config():
        cfg = dict({
            "b": 0.5, "beta": 1.25, "gamma_limits": [0.2, 0.2], 
            "C": 0.1, "x1star": 0.95, "r": -0.801, "year_length": 400.0,
            })
        cfg['t_burnin_phys'] = 10.0
        cfg['dt_step'] = 0.025
        cfg['dt_save'] = 0.1
        cfg["dt_plot"] = 2.5 #0.25
        cfg['timestepper'] = 'monotone'
        cfg['frc'] = dict({
            'type': 'impulsive',
            'impulsive': dict({
                'modes': [5],
                'magnitudes': [0.01],
                }),
            })

        # Specify tracer characteristics 
        Nxfv = 128
        cfg.update(dict(
            Nxfv = Nxfv,
            Nyfv = int(round(Nxfv*cfg['b']/2)),
            source_relative_width = 1/32,
            Nparticles = 128,
            # Vary concentration source periodically
            # TODO have a Dirichlet option 
            source_mean = 1.0,
            source_amplitude = 0.25,
            source_period = 50.0,
            glob_diss_rate = 1/40,
            ))
        return cfg
    @staticmethod
    def label_from_config(cfg):
        abbrv_x1star = (r"x1st%g_r%g_src%gpm%g_sink%g"%(cfg['x1star'],cfg['r'],cfg['source_mean'],cfg['source_amplitude'],cfg['glob_diss_rate'],)).replace(".","p")
        abbrv_gamma = (r'gam%gto%g'%(cfg['gamma_limits'][0],cfg['gamma_limits'][1])).replace('.','p')
        label_physpar = r"$x_1^*=%g,\ r=%g \n \gamma\in[%g,%g],\ \beta=%g$"%(
                cfg['x1star'],
                cfg['r'],
                cfg['gamma_limits'][0],cfg['gamma_limits'][1],
                cfg['beta']
                )
        if cfg['frc']['type'] == 'impulsive':
            abbrv_noise_type = "frcimp"
            label_noise_type = "Impulse: "
        w = cfg['frc'][cfg['frc']['type']]
        abbrv_noise_mode = ""
        label_noise_mode = ""
        if len(w['modes']) > 0:
            abbrv_noise_mode = "mode"
            abbrv_noise_mode += "-".join([f"{m:g}" for m in w['modes']]) 
        else:
            abbrv_noise_mode= "modenil"
        abbrv_fvgrid = r"fvgrid%dx%d"%(cfg['Nxfv'],cfg['Nyfv'])
        abbrv = "_".join([abbrv_x1star,abbrv_gamma,abbrv_noise_type,abbrv_noise_mode,abbrv_fvgrid]).replace('.','p')
        label = "\n".join([label_physpar,label_noise_type,label_noise_mode])

        return abbrv,label
    def derive_parameters(self, cfg):
        n_max = 1
        m_max = 2
        #self.dt_step = cfg['dt_step']
        self.dt_save = cfg['dt_save'] 
        self.dt_plot = cfg['dt_plot']
        self.dt_step = cfg['dt_step']  #min(q['dx']/max_speed, q['dy']/max_speed, self.dt_save)
        self.t_burnin = int(cfg['t_burnin_phys']/self.dt_save) # depends on whether to use a pre-seeded initial condition 
        q = dict()
        flowdim = 6
        Nparticles = cfg['Nparticles']
        q["flowdim"] = flowdim
        q["Nparticles"] = cfg['Nparticles']
        q["year_length"] = cfg["year_length"]
        q["epsilon"] = 16*np.sqrt(2)/(5*np.pi)
        q["C"] = cfg["C"]
        q["b"] = cfg["b"]
        q["gamma_limits_cfg"] = cfg["gamma_limits"]
        q["xstar"] = np.array([cfg["x1star"],0,0,cfg["r"]*cfg["x1star"],0,0])
        q["alpha"] = np.zeros(m_max)
        q["beta"] = np.zeros(m_max)
        q["gamma_limits"] = np.zeros((2,m_max))
        q["gamma_tilde_limits"] = np.zeros((2,m_max))
        q["delta"] = np.zeros(m_max)
        for i_m in range(m_max):
            m = i_m + 1
            q["alpha"][i_m] = 8*np.sqrt(2)/np.pi*m**2/(4*m**2 - 1) * (cfg["b"]**2 + m**2 - 1)/(cfg["b"]**2 + m**2)
            q["beta"][i_m] = cfg["beta"]*cfg["b"]**2/(cfg["b"]**2 + m**2)
            q["delta"][i_m] = 64*np.sqrt(2)/(15*np.pi) * (cfg["b"]**2 - m**2 + 1)/(cfg["b"]**2 + m**2)
            for j in range(2):
                q["gamma_tilde_limits"][j,i_m] = cfg["gamma_limits"][j]*4*m/(4*m**2 - 1)*np.sqrt(2)*cfg["b"]/np.pi
                q["gamma_limits"][j,i_m] = cfg["gamma_limits"][j]*4*m**3/(4*m**2 - 1)*np.sqrt(2)*cfg["b"]/(np.pi*(cfg["b"]**2 + m**2))
        # Construct the forcing, linear matrix, and quadratic matrix, but leaving out the time-dependent parts 
        # 1. Constant term
        F = q["C"]*q["xstar"]
        # 2. Matrix for linear term
        L = -q["C"]*np.eye(flowdim)
        #L[0,2] = q["gamma_tilde"][0]
        L[1,2] = q["beta"][0]
        #L[2,0] = -q["gamma"][0]
        L[2,1] = -q["beta"][0]
        #L[3,5] = q["gamma_tilde"][1]
        L[4,5] = q["beta"][1]
        L[5,4] = -q["beta"][1]
        #L[5,3] = -q["gamma"][1]
        # 3. Matrix for bilinear term
        B = np.zeros((flowdim,flowdim,flowdim))
        B[1,0,2] = -q["alpha"][0]
        B[1,3,5] = -q["delta"][0]
        B[2,0,1] = q["alpha"][0]
        B[2,3,4] = q["delta"][0]
        B[3,1,5] = q["epsilon"]
        B[3,2,4] = -q["epsilon"]
        B[4,0,5] = -q["alpha"][1]
        B[4,3,2] = -q["delta"][1]
        B[5,0,4] = q["alpha"][1]
        B[5,3,1] = q["delta"][1]
        q["forcing_term"] = F
        q["linear_term"] = L
        q["bilinear_term"] = B
        # 4. Grid for finite volume
        q['Lx'] = 2*np.pi
        q['Ly'] = np.pi*cfg['b']
        q['Nx'] = cfg['Nxfv']
        q['Ny'] = cfg['Nyfv']
        q['dx'] = q['Lx']/cfg['Nxfv']
        q['dy'] = q['Ly']/cfg['Nyfv']
        q['x_s'],q['y_s'],q['basis_s'],q['x_u'],q['y_u'],q['basis_u'],q['x_v'],q['y_v'],q['basis_v'],q['x_c'],q['y_c'] = self.basis_functions(q['Nx'],q['Ny'])

        # source and sink
        self.timestep_constants = q
        # Impulse matrix
        imp_modes = cfg['frc']['impulsive']['modes']
        imp_mags = cfg['frc']['impulsive']['magnitudes']
        self.impulse_dim = len(imp_modes)
        #print(f'{imp_mags = }')
        #print(f'{imp_modes = }')
        self.impulse_matrix = np.zeros((self.state_dim,self.impulse_dim))
        for i,mode in enumerate(imp_modes):
            self.impulse_matrix[mode,0] += imp_mags[i]

        # Intermediate arrays allocated to store computations 
        # Runge-Kutta for streamfunction coefficients
        self.source_flag = np.zeros((q['Nx'],q['Ny']), dtype=bool)
        source_width = cfg['source_relative_width']*q['Ly']
        num_edge_cells = max(1, int(round(source_width/q['dy'])))
        self.source_flag[:,:num_edge_cells] = True
        self.source_mean = cfg['source_mean']
        self.source_amplitude = cfg['source_amplitude']
        self.source_period = cfg['source_period']
        self.sink = cfg['glob_diss_rate']
        self.s_rk4_1,self.s_rk4_2,self.s_rk4_3,self.s_rk4_4 = (np.zeros(flowdim) for _ in range(4))
        self.s_tendency_total,self.s_tendency_advection,self.s_tendency_dissipation,self.s_tendency_forcing = (np.zeros(flowdim) for _ in range(4))
        (self.s_temp, self.s_next_temp) = (np.zeros(flowdim) for _ in range(2))
        # Finite-volume for concentration field 
        self.conc_next_temp = np.zeros(q['Nx']*q['Ny'])
        self.u_eul,self.flux_u_eul = (np.zeros((q['Nx']+1,q['Ny'])) for _ in range(2))
        self.v_eul,self.flux_v_eul = (np.zeros((q['Nx'],q['Ny']+1)) for _ in range(2))
        self.iflat_nbs = np.zeros(4, dtype=int)
        self.outflows,self.inflows = (np.zeros(4) for _ in range(2))
        (self.flux_coefs_center,self.flux_coefs_right,self.flux_coefs_left,self.flux_coefs_up,self.flux_coefs_down) = (np.zeros((q['Nx'], q['Ny'])) for _ in range(5)) # self, right, left, top, bottom in that order
        # Forward Euler for particle positions
        (self.c1x_lag,self.s1x_lag,self.c1y_lag,self.s1y_lag,self.c2y_lag,self.s2y_lag) = (np.zeros(Nparticles) for _ in range(6))
        (self.s_lag,self.u_lag,self.v_lag) = (np.zeros(Nparticles) for _ in range(3))
        self.death_flag = np.zeros(Nparticles, dtype=bool)
        return 
    def timestep_finvol(self, state_next, t, state):
        # This is the non-numba version
        q = self.timestep_constants

        gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t = self.orography_cycle(t)
        dt = timestep_finvol_external(
                state_next,
                t, self.dt_save, 
                state,
                *(q[key] for key in 'flowdim,Nparticles,Nx,Ny,dx,dy,Lx,Ly,b'.split(',')),
                *(q[key] for key in 'forcing_term,linear_term,bilinear_term'.split(',')),
                gamma_t,gamma_tilde_t,
                *(q[key] for key in 'basis_u,basis_v,source_flag,source_conc,source_width'.split(',')),
                *(getattr(self, f's_rk4_{i}') for i in [1,2,3,4]),
                self.s_next_temp,
                self.s_tendency_total,
                *(getattr(self, key) for key in 'u_eul,v_eul,iflat_nbs,outflows,inflows,s_lag,u_lag,v_lag,death_flag,c1x_lag,s1x_lag,c1y_lag,s1y_lag,c2y_lag,s2y_lag'.split(',')),
                )
        return dt #t+self.dt_step, np.concatenate((strfn_next, conc_next, x_lag_next, y_lag_next))


    def integrate_monotone(self, state_save, tp_save, init_time, init_cond):
        q = self.timestep_constants
        init_time_phys = init_time * self.dt_save
        gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t = self.orography_cycle(init_time_phys)
        integrate_monotone_external(
            # intent(out)
            state_save, 
            # intent(in)
            tp_save, self.dt_step,
            init_time_phys,
            init_cond,
            *(q[key] for key in 'flowdim,Nparticles,Nx,Ny,dx,dy,Lx,Ly,b'.split(',')),
            *(q[key] for key in 'forcing_term,linear_term,bilinear_term'.split(',')),
            gamma_t,gamma_tilde_t, # for simplicity, we will have no orography cycle 
            *(q[key] for key in 'basis_u,basis_v'.split(',')),
            *(getattr(self, f'source_{spec}') for spec in ('flag,mean,amplitude,period').split(',')), self.sink,
            *(getattr(self, f's_rk4_{i}') for i in [1,2,3,4]),
            self.s_next_temp,
            self.s_tendency_total,
            *(
                getattr(self, key) for key in (
                    'u_eul,v_eul,iflat_nbs,outflows,inflows,' + 
                    'flux_u_eul,flux_v_eul,' + 
                    'flux_coefs_center,flux_coefs_right,flux_coefs_left,flux_coefs_up,flux_coefs_down,' +
                    's_lag,u_lag,v_lag,death_flag,' + 
                    'c1x_lag,s1x_lag,c1y_lag,s1y_lag,c2y_lag,s2y_lag'
                    )
                .split(',')
                )
            )
        return tp_save, state_save

    # non-numba version 
    def timestep_monotone(self, state_next, t, state):
        q = self.timestep_constants

        gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t = self.orography_cycle(t)
        dt = timestep_monotone_external(
                state_next,
                t, self.dt_save, 
                state,
                *(q[key] for key in 'flowdim,Nparticles,Nx,Ny,dx,dy,Lx,Ly,b'.split(',')),
                *(q[key] for key in 'forcing_term,linear_term,bilinear_term'.split(',')),
                gamma_t,gamma_tilde_t,
                *(q[key] for key in 'basis_u,basis_v,source_flag,source_conc,source_width'.split(',')),
                *(getattr(self, f's_rk4_{i}') for i in [1,2,3,4]),
                self.s_next_temp,
                self.s_tendency_total,
                *(
                    getattr(self, key) for key in (
                        'u_eul,v_eul,iflat_nbs,outflows,inflows,' + 
                        'flux_u_eul,flux_v_eul,' + 
                        'flux_coefs_center,flux_coefs_right,flux_coefs_left,flux_coefs_up,flux_coefs_down,' +
                        's_lag,u_lag,v_lag,death_flag,' + 
                        'c1x_lag,s1x_lag,c1y_lag,s1y_lag,c2y_lag,s2y_lag'
                        )
                    .split(',')
                    )
                )
        return dt #t+self.dt_step, np.concatenate((strfn_next, conc_next, x_lag_next, y_lag_next))

    def basis_functions(self, Nx, Ny):
        # Compute streamfunction and derivatives at cell corners and face centers (C-grid)
        sqrt2 = np.sqrt(2)
        Lx = 2*np.pi
        Ly = np.pi*self.config['b']
        dx = Lx/Nx
        dy = Ly/Ny
        b = self.config['b']
        # vertical faces where zonal velocity is evaluated
        x_u = np.linspace(0,Lx,Nx+1)
        y_u = np.linspace(dy/2,Ly-dy/2,Ny)
        # horizontal faces where meridional velocity is evaluated
        x_v = np.linspace(dx/2,Lx-dx/2,Nx)
        y_v = np.linspace(0,Ly,Ny+1)
        # vertices where streamfunction is evaluated
        x_s = np.linspace(0,Lx,Nx+1)
        y_s = np.linspace(0,Ly,Ny+1)
        # cell centers where concentrations are evaluated
        x_c = np.linspace(dx/2,Lx-dx/2,Nx)
        y_c = np.linspace(dy/2,Ly-dy/2,Ny)
        # 
        c1x_u,c1x_v,c1x_s,c1x_c = (np.cos(x) for x in (x_u,x_v,x_s,x_c))
        s1x_u,s1x_v,s1x_s,s1x_c = (np.sin(x) for x in (x_u,x_v,x_s,x_c))
        c1y_u,c1y_v,c1y_s,c1y_c = (np.cos(y/b) for y in (y_u,y_v,y_s,y_c))
        s1y_u,s1y_v,s1y_s,s1y_c = (np.sin(y/b) for y in (y_u,y_v,y_s,y_c))
        c2y_u,c2y_v,c2y_s,c2y_c = (np.cos(2*y/b) for y in (y_u,y_v,y_s,y_c))
        s2y_u,s2y_v,s2y_s,s2y_c = (np.sin(2*y/b) for y in (y_u,y_v,y_s,y_c))
        # Streamfunction
        basis_s = np.zeros((6,Nx+1,Ny+1)) 
        basis_s[0,:,:] = b*sqrt2 * np.outer(np.ones_like(x_s), c1y_s)
        basis_s[3,:,:] = b*sqrt2 * np.outer(np.ones_like(x_s), c2y_s)
        basis_s[1,:,:] =  2*b     * np.outer(c1x_s, s1y_s)
        basis_s[2,:,:] = -2*b     * np.outer(s1x_s, s1y_s)
        basis_s[4,:,:] =  2*b     * np.outer(c1x_s, s2y_s)
        basis_s[5,:,:] = -2*b     * np.outer(s1x_s, s2y_s)
        # zonal velocity 
        basis_u = np.zeros((6,Nx+1,Ny)) 
        basis_u[0,:,:] = -b*sqrt2 * np.outer(np.ones_like(x_u), -s1y_u/b)
        basis_u[3,:,:] = -b*sqrt2 * np.outer(np.ones_like(x_u), -s2y_u*2/b)
        basis_u[1,:,:] = -2*b    * np.outer(c1x_u, c1y_u/b)
        basis_u[2,:,:] =  2*b    * np.outer(s1x_u, c1y_u/b)
        basis_u[4,:,:] = -2*b    * np.outer(c1x_u, c2y_u*2/b)
        basis_u[5,:,:] =  2*b    * np.outer(s1x_u, c2y_u*2/b)
        # meridional velocity 
        basis_v = np.zeros((6,Nx,Ny+1)) 
        basis_v[1,:,:] =  2*b     * np.outer(-s1x_v/b, s1y_v)
        basis_v[2,:,:] = -2*b     * np.outer(c1x_v/b, s1y_v)
        basis_v[4,:,:] =  2*b     * np.outer(-s1x_v/b, s2y_v)
        basis_v[5,:,:] = -2*b     * np.outer(c1x_v/b, s2y_v)
        return (
                x_s,y_s,basis_s,
                x_u,y_u,basis_u,
                x_v,y_v,basis_v,
                x_c,y_c,
                )
    def orography_cycle(self,t_abs):
        """
        Parameters
        ----------
        t_abs: numpy.ndarray
            The absolute time. Shape should be (Nx,) where Nx is the number of ensemble members running in parallel. 

        Returns 
        -------
        gamma_t: numpy.ndarray
            The gamma parameter corresponding to the given time of year. It varies sinusoidaly. Same is (Nx,m_max) where m_max is the maximum zonal wavenumber.
        gamma_tilde_t: numpy.ndarray
            The gamma_tilde parameter corresponding to the given time of year. Same shape as gamma_t.
        """
        q = self.timestep_constants
        cosine = np.cos(2*np.pi*t_abs/q["year_length"])
        sine = np.sin(2*np.pi*t_abs/q["year_length"])
        gamma_t = cosine * (q["gamma_limits"][1,:] - q["gamma_limits"][0,:])/2 + (q["gamma_limits"][0,:] + q["gamma_limits"][1,:])/2
        gammadot_t = -sine * (q["gamma_limits"][1,:] - q["gamma_limits"][0,:])/2
        gamma_tilde_t = cosine * (q["gamma_tilde_limits"][1,:] - q["gamma_tilde_limits"][0,:])/2 + (q["gamma_tilde_limits"][0,:] + q["gamma_tilde_limits"][1,:])/2
        gammadot_tilde_t = -sine * (q["gamma_tilde_limits"][1,:] - q["gamma_tilde_limits"][0,:])/2 + (q["gamma_tilde_limits"][0,:] + q["gamma_tilde_limits"][1,:])/2
        gamma_cfg_t = cosine * (q["gamma_limits_cfg"][1] - q["gamma_limits_cfg"][0])/2 
        gammadot_cfg_t = -sine * (q["gamma_limits_cfg"][1] - q["gamma_limits_cfg"][0])/2
        return gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t
    def compute_tendency_forcing(self,s_tendency_forcing,t,s):
        s_tendency_forcing[:] = self.timestep_constants["forcing_term"]
        return
    def compute_tendency_dissipation(self,s_tendency_dissipation,t,s):
        s_tendency_dissipation[:] = self.timestep_constants["linear_term"] @ s
        # Modify the time-dependent components
        gamma_t,gamma_tilde_t,gamma_cfg_t,gammadot_t,gammadot_tilde_t,gammadot_cfg_t = self.orography_cycle(t)
        s_tendency_dissipation[0] += gamma_tilde_t[0]*s[2]
        s_tendency_dissipation[2] -= gamma_t[0]*s[0]
        s_tendency_dissipation[3] += gamma_tilde_t[1]*s[5]
        s_tendency_dissipation[5] -= gamma_t[1]*s[3]
        return 
    def compute_tendency_advection(self,s_tendency_advection,t,s):
        """
        Compute the tendency according to only the nonlinear terms, in order to check conservation of energy and enstrophy.
        """
        flowdim = self.timestep_constants["flowdim"]
        s_tendency_advection[:] = 0
        for j in range(flowdim):
            s_tendency_advection[j] += np.sum(s * (self.timestep_constants["bilinear_term"][j] @ s))
        return 
    def compute_streamfunction_tendency(
            self, 
            s_tendency_total, s_tendency_advection, s_tendency_dissipation, s_tendency_forcing, 
            t, s
            ):
        self.compute_tendency_advection(s_tendency_advection,t,s)
        self.compute_tendency_dissipation(s_tendency_dissipation,t,s)
        self.compute_tendency_forcing(s_tendency_forcing,t,s)
        s_tendency_total[:] = s_tendency_advection + s_tendency_dissipation + s_tendency_forcing
        return
    def generate_default_init_cond(self, init_time):
        rng = default_rng(seed=49582)
        # Flow
        s_star = self.timestep_constants["xstar"]
        # Concentrations
        q = self.timestep_constants
        Nx,Ny,dx,dy,Nparticles = (q[key] for key in ['Nx','Ny','dx','dy','Nparticles'])
        conc = np.zeros(Nx*Ny) #0 * (q['source_flag'] * q['source_conc']).flatten()
        idx_part_x,idx_part_y = np.unravel_index(rng.choice(np.arange(Nx*Ny), size=Nparticles, ), (Nx,Ny)) #p=conc/np.sum(conc), replace=True), (Nx,Ny))
        #print(f'{idx_part_x = }')
        #print(f'{idx_part_y = }')
        # Tracer positions (in general, draw from the sources)
        x_part = np.zeros(Nparticles)
        y_part = np.zeros(Nparticles)
        for i_part in range(Nparticles):
            x_part[i_part] = dx * (idx_part_x[i_part] + rng.uniform())
            y_part[i_part] = dy * (idx_part_y[i_part] + rng.uniform())
        state_init = np.concatenate((s_star,conc,x_part,y_part))
        return state_init
    def compute_observables(self, obs_funs, metadata, root_dir):
        t,x = Crommelin2004TracerODE.load_trajectory(metadata, root_dir)
        obs = []
        for i_fun,fun in enumerate(obs_funs):
            obs.append(fun(t,x))
        return obs
    def compute_pairwise_observables(self, pairwise_funs, md0, md1list, root_dir):
        # Compute distances between one trajectory and many others
        pairwise_fun_vals = [[] for pwf in pairwise_funs] # List of lists
        t0,x0 = self.load_trajectory(md0, root_dir)
        for i_md1,md1 in enumerate(md1list):
            t1,x1 = self.load_trajectory(md1, root_dir)
            for i_pwf,pwf in enumerate(pairwise_funs):
                pairwise_fun_vals[i_pwf].append(pwf(t0,x0,t1,x1))
        return pairwise_fun_vals
    @staticmethod
    def observable_props():
        obslib = dict({
            'c1y': dict({
                'abbrv': 'c1y',
                'label': r'$\cos(y/b)$',
                'cmap': 'coolwarm',
                }),
            'c1xs1y': dict({
                'abbrv': 'c1xs1y',
                'label': r'$\cos(x)\sin(y/b)$',
                'cmap': 'coolwarm',
                }),
            's1xs1y': dict({
                'abbrv': 's1xs1y',
                'label': r'$\sin(x)\sin(y/b)$',
                'cmap': 'coolwarm',
                }),
            'c2y': dict({
                'abbrv': 'c2y',
                'label': r'$\cos(2y/b)$',
                'cmap': 'coolwarm',
                }),
            'c1xs2y': dict({
                'abbrv': 'c1xs2y',
                'label': r'$\cos(x)\sin(2y/b)$',
                'cmap': 'coolwarm',
                }),
            's1xs2y': dict({
                'abbrv': 's1xs2y',
                'label': r'$\sin(x)\sin(2y/b)$',
                'cmap': 'coolwarm',
                }),
            })
        return obslib
    @staticmethod
    def c1y(t, state):
        return state[:,0]
    @staticmethod
    def c2y(t, state):
        return state[:,3]
    @staticmethod
    def c1xs1y(t,state):
        return state[:,1]
    @staticmethod
    def s1xs1y(t,state):
        return state[:,2]
    @staticmethod
    def c1xs2y(t,state):
        return state[:,4]
    @staticmethod
    def s1xs2y(t,state):
        return state[:,5]
    # Observables related to local concentration
    def local_conc(self,t,state,x,y):
        q = self.timestep_constants
        flowdim,dx,dy,Nx,Ny = (q[key] for key in 'flowdim,dx,dy,Nx,Ny'.split(','))
        ix,iy = int(x/dx),int(y/dy)
        iflat = ix*Ny + iy
        return state[:,flowdim+iflat]

    # --------------- plotting functions -----------------
    def check_fig_ax(self, fig=None, ax=None):
        if fig is None:
            if ax is None:
                fig,ax = plt.subplots()
        elif ax is None:
            raise Exception("You can't just give me a fig without an axis")
        return fig,ax
    def plot_mode_energy_timeseries(self, t, x, k, linekw, fig=None, ax=None, ):
        fig,ax = self.check_fig_ax(fig,ax)
        h, = ax.plot(t*self.dt_save, x[:,k], **linekw)
        ax.set_xlabel("Time")
        return fig,ax,h


if __name__ == "__main__":
    cfg = Crommelin2004TracerODE.default_config()
    crom = Crommelin2004TracerODE(cfg)
    tp_init = 0.0
    tp_fin = 100.0
    tp_save = np.arange(tp_init, tp_fin, step=crom.dt_save)
    


