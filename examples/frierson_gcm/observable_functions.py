import numpy as np
from scipy.sparse import csr_matrix, csgraph
from scipy import ndimage
import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import os
from os import listdir as ls
from os import mkdir 
from os.path import exists, join
import dask
import glob
import precip_extremes_scaling

def preprocess(ds):
    # What to do with individual netcdf files before merging with with Dask
    # 1. Remove the dimension "latb" for seamless merging of the latitude-partitioned datasets
    ds = ds.drop_dims("latb")
    return ds

def selnear(ds, coordname, coordval):
    # Select the coordinate value closest to the requested one
    icoord = np.argmin(np.abs(ds.coords[coordname].values - coordval))
    return ds.isel({coordname: icoord}, drop=True)

def precompute_features_vertprof(ds, fields2comp, savefolder, latlonbounds, overwrite=False):
    # Precompute and store a bunch of features of interest to enable faster processing for composites. Assume each one is small enough to fit in memory, ergo don't use Dask to store the output (however, the computation can use Dask).
    obslib = observable_library()
    target_files = dict()
    if not exists(savefolder):
        raise Exception(f"The savefolder {savefolder} does not exist")
    latb = latlonbounds["lat"]
    lonb = latlonbounds["lon"]
    latlonsel = dict(lat=slice(latb[0],latb[1]),lon=slice(lonb[0],lonb[1]))
    latlonstr = f"{int(latb[0])}-{int(latb[1])}N_{int(lonb[0])}-{int(lonb[1])}E"
    for (field,_) in fields2comp:
        file2write = join(savefolder,f"{obslib[field]['abbrv']}_{latlonstr}.nc")
        target_files[field] = file2write
        if exists(file2write) and (not overwrite):
            print(f"{field} already exists")
        else:
            print(f"Starting computation of {field}")
            f = observable_from_name_vertprof(ds, field, latlonsel).compute()
            print(f"\tStarting to save {field} to netcdf")
            f.to_netcdf(file2write)
    return target_files

def precompute_features(ds, fields2comp, savefolder, overwrite=False):
    # Precompute and store a bunch of features of interest to enable faster processing for composites. Assume each one is small enough to fit in memory, ergo don't use Dask to store the output (however, the computation can use Dask).
    obslib = observable_library()
    target_files = dict()
    if not exists(savefolder):
        raise Exception(f"The savefolder {savefolder} does not exist")
    for (field,pfull) in fields2comp:
        pstr = "" if pfull is None else f"{int(pfull)}"
        file2write = join(savefolder,f"{obslib[field]['abbrv']}{pstr}.nc")
        target_files[field] = file2write
        if exists(file2write) and (not overwrite):
            print(f"{field} already exists")
        else:
            print(f"Starting computation of {field} at pfull {pfull}")
            f = observable_from_name(ds, field, pfull_target=pfull).compute()
            print(f"\tStarting to save {field} to netcdf")
            f.to_netcdf(file2write)
    return target_files

def lon_dist(lon1, lon2):
    dist_naive = np.abs(lon1 - lon2)
    dist_periodic = np.minimum(dist_naive, 360 - dist_naive)
    return dist_periodic

def lon_Linf_mean(lon_arr):
    # Given an array of longitudes, return the mean, taking periodicity into account. 
    mean_0 = (np.min(lon_arr) + np.max(lon_arr))/2
    mean_1 = np.mod(mean_0 + 180, 360)
    max_dist_0 = np.max(np.abs(lon_dist(lon_arr, mean_0)))
    max_dist_1 = np.max(np.abs(lon_dist(lon_arr, mean_1)))
    if max_dist_1 < max_dist_0:
        mean_periodic = mean_1
    else:
        mean_periodic = mean_0
    return mean_periodic

def rotate_to_central_lon(da, central_lon):
    # Rotate so that a fixed longitude is at the center
    i_lon = np.argmin(np.abs(da["lon"].data - central_lon))
    da_rot = da.roll(lon=-i_lon)
    return da_rot

def lon_Linf_mean_unit_test():
    # Test a case with wrapping
    lon_arr = np.array([350,351,352,353,10])
    linf_mean = lon_Linf_mean(lon_arr)
    print(f"Mean of \n{lon_arr}\n is {linf_mean}")
    lon_arr = np.array([20,23,25,45])
    linf_mean = lon_Linf_mean(lon_arr)
    print(f"Mean of \n{lon_arr}\n is {linf_mean}")
    return

def reduce_labels(labels0, labels1):
    Nlab = max(np.max(labels0), np.max(labels1)) + 1
    C = csr_matrix((np.ones(len(labels0)), (labels0, labels1)), shape=(Nlab,Nlab))
    n_components,graph_labels = csgraph.connected_components(csgraph=C,directed=False,return_labels=True)
    # Return a data structure informing how to relabel the clusters
    unique_labels = np.unique(np.concatenate((labels0,labels1)))
    old_new = -np.ones((2, len(unique_labels)), dtype=int)
    old_new[0] = unique_labels
    for j in range(n_components): # The graph labels start at 0
        idx_graph, = np.where(graph_labels==j) # between 0 and Nlab
        idx_old, = np.where(np.in1d(unique_labels, idx_graph))
        if len(idx_old) > 0:
            old_new[1,idx_old] = np.min(old_new[0,idx_old])
    return old_new

def conn_comp_periodic(cube,verbose=False):
    # Given a binary cube with coordinates (time, latitude, longitude), find the connected components accounting for periodicity
    connectivity = np.ones((3,3,3), dtype=int)
    connectivity[([0,0,0,0],[0,2,0,2],[0,0,2,2])] = 0
    connectivity[([2,2,2,2],[0,2,0,2],[0,0,2,2])] = 0
    labels, num_comp = ndimage.label(cube, structure=connectivity)
    print(f"Before fixing the boundaries, num_comp = {num_comp}")
    if num_comp == 0:
        return labels, np.array([])
    if verbose: 
        print(f"First labels = \n{labels}")
        print(f"cube = \n{cube}")
    # Stitch together the walls 
    # Make a list of pairs of clusters that touch across the walls
    overlap_idx = np.where(cube[:,:,-1] * cube[:,:,0])
    num_overlaps = len(overlap_idx[0])
    print(f"num overlaps = {num_overlaps}")
    if num_overlaps > 0:
        right_overlaps = labels[:,:,-1][overlap_idx]
        left_overlaps = labels[:,:,0][overlap_idx]
        old_new = reduce_labels(right_overlaps, left_overlaps)
        for j in range(old_new.shape[1]):
            idx = np.where(labels==old_new[0,j])
            labels[idx] = old_new[1,j]
        # Re-index
        removed_labels = np.setdiff1d(old_new[0], old_new[1])
        unique_labels = np.setdiff1d(np.arange(1,num_comp+1), removed_labels)
    else:
        unique_labels = np.arange(1,num_comp+1)
    return labels, unique_labels

def test_conn_comp_periodic():
    # Unit test for conn_comp_periodic
    cube = np.random.uniform(size=(2,9,5))
    cube[:,:,2] = 0
    cube[:,[1,3,5,7],:] = 0
    cube[([0,0,0],[1,3,7],[1,3,3])] = np.random.uniform(size=3)
    print(f"Rectified data:\n{cube*(cube>0)}")
    labels,unique_labels = conn_comp_periodic(cube>0, verbose=True)
    print(f"Number of clusters = {len(unique_labels)}")
    print(f"Unique labels = \n{unique_labels}")
    print(f"Final labels = \n{labels}")
    return

def pressure(ds):
    # Return pressure with "pfull" as a vertical coordinate. 
    p_edge = ds["bk"]*ds["ps"] # Pascals
    p_cent = 0.5*(p_edge + p_edge.shift(phalf=-1)).isel(phalf=slice(None,-1)).rename({"phalf": "pfull"}).assign_coords({"pfull": ds["pfull"]})
    dp_dpfull = (p_edge.shift(phalf=-1) - p_edge)/(ds["phalf"].shift(phalf=-1) - ds["phalf"])
    dp_dpfull = dp_dpfull.isel(phalf=slice(None,-1)).rename({"phalf": "pfull"}).assign_coords({"pfull": ds["pfull"]})
    return p_cent, dp_dpfull
    

def sat_spec_hum(ds):
    p,_ = pressure(ds)
    #p = ds["bk"] * ds["ps"] # Pascals
    #p = ds["pfull"] * 100 # it's in hectopascals
    temp = ds["temp"]
    lam = 1.0
    # Constants
    Rd = 287.04
    Rv = 461.5
    cpd = 1005.7
    cpv = 1870.0
    g = 9.80665
    p0 = 1e5
    kappa = Rd/cpd
    gc_ratio = Rd/Rv

    # Saturation vapor pressure
    Tc = temp - 273.15
    es = 611.2 * np.exp(17.67*Tc/(Tc + 243.5))
    
    # Latent heat of condensation
    L = (2.501-0.00237*Tc)*1e6
    
    # Saturation mixing ratio
    rs = gc_ratio*es/(p - es)

    # Saturation specific humidity
    qs = rs/(1 + rs)
    return qs

def eff_stat_stab(ds):
    # Compute effective static stability
    p,dp_dpfull = pressure(ds)
    temp = ds["temp"]
    lam = 1.0
    # Constants
    Rd = 287.04
    Rv = 461.5
    cpd = 1005.7
    cpv = 1870.0
    g = 9.80665
    p0 = 1e5
    kappa = Rd/cpd
    gc_ratio = Rd/Rv

    # Saturation vapor pressure
    Tc = temp - 273.15
    es = 611.2 * np.exp(17.67*Tc/(Tc + 243.5))
    
    # Latent heat of condensation
    L = (2.501-0.00237*Tc)*1e6
    
    # Saturation mixing ratio
    rs = gc_ratio*es/(p - es)

    # Saturation specific humidity
    qs = rs/(1 + rs)
    
    # Potential temperature
    exponent = kappa*(1 + rs/gc_ratio)/(1 + rs*cpv/cpd)
    theta = temp * (p0/p) ** exponent

    # derivative of potential temperature with respect to pressure
    dtheta_dp = theta.differentiate("pfull") / dp_dpfull

    # density
    temp_virtual = temp * (1 + rs/gc_ratio)/(1 + rs)
    rho = p/Rd/temp_virtual

    # moist adiabatic lapse rate
    malr = g/cpd * (1+rs) / (1+cpv/cpd*rs) * (1+L*rs/Rd/temp) / (1+L**2*rs*(1+rs/gc_ratio)/(Rv*temp**2*(cpd+rs*cpv)))

    # Derivative of potential temperature wrt pressure along a moist adiabat (neglects small contribution from vertical variations of exponent)
    dtemp_dp_ma = malr/g/rho
    dtheta_dp_ma = dtemp_dp_ma * theta/temp - exponent*theta/p

    # effective static stability (eq. 8 of O'Gorman JAS 2011)
    dtheta_dp_eff = dtheta_dp - lam*dtheta_dp_ma

    # Convert to buoyancy frequency
    dtheta_dz_eff = dtheta_dp_eff * (-rho*g) # K/m
    Nsq_eff = dtheta_dz_eff * g/theta # 1/s^2
    print(f"Nsq_eff coords = {Nsq_eff.coords}")
    return Nsq_eff 

def column_water_vapor(ds):
    g = 9.806 
    p,dp_dpfull = pressure(ds)
    p = ds["bk"] * ds["ps"] # Pascals
    cwv = (ds["sphum"] * dp_dpfull).integrate("pfull")/g
    return cwv

def water_vapor_convergence(ds):
    g = 9.806 
    p,dp_dpfull = pressure(ds)
    p = ds["bk"] * ds["ps"] # Pascals
    conv = -divergence(ds["ucomp"]*ds["sphum"], ds["vcomp"]*ds["sphum"])
    qcon = (conv * dp_dpfull).integrate("pfull")/g
    return qcon 

def column_relative_humidity(ds):
    # CWV / max possible CWV
    p,dp_dpfull = pressure(ds)
    cwv_xg = (ds["sphum"] * dp_dpfull).integrate("pfull") # / g, but this cancels 
    qs = sat_spec_hum(ds)
    cwv_max_xg = (qs * dp_dpfull).integrate("pfull") # / g, but this cancels
    return cwv_xg / cwv_max_xg

def vert_deriv_sat_sphum(ds):
    # Vertical derivative of saturation specific humidity at fixed saturation equivalent potential temperature
    pass

def divergence(u, v):
    # Divergence in spherical coordinates
    a = 6371.0e3 # radius of earth
    coslat = np.cos(np.deg2rad(u["lat"]))
    div = 1.0/(a*coslat)*((v*coslat).differentiate("lat") + u.differentiate("lon")) * 180/np.pi
    return div

def vorticity(ds):
    return curl(ds["ucomp"], ds["vcomp"])

def curl(u, v):
    # Divergence in spherical coordinates (in the vertical direction)
    a = 6371.0e3 # radius of earth
    coslat = np.cos(np.deg2rad(u["lat"]))
    uxv = 1.0/(a*coslat)*(v.differentiate("lon") - (u*coslat).differentiate("lat")) * 180/np.pi
    return uxv

def condensation_rain(ds):
    cond = ds["condensation_rain"] * 3600*24 # From kg/(m**2 * s) to kg/(m**2 * day) = mm/day
    cond.attrs["units"] = "mm/day"
    return cond

def convection_rain(ds):
    if "convection_rain" in list(ds.data_vars.keys()):
        conv = ds["convection_rain"] * 3600*24 # From kg/(m**2 * s) to kg/(m**2 * day) = mm/day
    else:
        conv = xr.zeros_like(ds["condensation_rain"])
    conv.attrs["units"] = "mm/day"
    return conv

def temperature(ds):
    return ds["temp"] 

def specific_humidity(ds):
    return ds["sphum"] 

def surface_pressure(ds):
    return ds["ps"]

def vertical_velocity(ds):
    return ds["omega"]

def zonal_velocity(ds):
    return ds["ucomp"]

def meridional_velocity(ds):
    return ds["vcomp"]


def coord_stepsize(ds):
    # Measure the timestep (assume it's uniformly spaced)
    stepsize = dict()
    for v in ["time","lat","lon"]:
        stepsize[v] = ds[v][1].item() - ds[v][0].item()
    return stepsize

def area_avg(da, lat_bounds, lon_bounds):
    latsel = dict(lat=slice(lat_bounds[0],lat_bounds[1]))
    lonsel = dict(lon=slice(lon_bounds[0],lon_bounds[1]))
    cos_weight = (
            np.cos(np.deg2rad(da["lat"].sel(latsel)))
            * xr.ones_like(da["lon"].sel(lonsel))
            )
    cos_weight *= 1.0/cos_weight.sum()
    aa = (da.sel(latsel).sel(lonsel) * cos_weight).sum(dim=["lat","lon"])
    return aa

def total_rain(ds):
    conv = convection_rain(ds)
    cond = condensation_rain(ds)
    total = conv + cond
    return total

def exprec_scaling_wrapper(ds):
    # Swap the order of pressure to be increasing on all variables
    omega = ds["omega"].reindex(pfull=ds["pfull"][::-1])
    temp = ds["temp"].reindex(pfull=ds["pfull"][::-1])
    ps = ds["ps"]
    p, dp_dpfull = pressure(ds)
    p = p.reindex(pfull=ds["pfull"][::-1])
    dp_dpfull = dp_dpfull.reindex(pfull=ds["pfull"][::-1])
    scaling = precip_extremes_scaling.scaling(omega, temp, p, dp_dpfull, ps)
    scaling *= 3600 * 24
    return scaling

def resample_to_daily(da):
    # old:
    #da_resamp = da.groupby(np.ceil(da["time"]).astype(int)).mean(dim="time")
    # new (trial):
    #dt = coord_stepsize(da)["time"]
    #steps_per_day = int(round(1.0/dt))
    #day_end_tidx = np.where(np.mod(da["time"].to_numpy(), 1.0) == 0)[0]
    #runavg = da.rolling({"time": steps_per_day}, min_periods=1).mean().isel(day_end_idx)
    # newer trial): 
    day_end_tidx = np.where(np.mod(da["time"].to_numpy(), 1.0) == 0)[0]
    steps_per_day = day_end_tidx[1] - day_end_tidx[0]
    runavg = da.isel(time=day_end_tidx)
    for i_delay in range(1,steps_per_day):
        runavg += da.shift(time=i_delay).isel(time=day_end_tidx)
    runavg *= 1.0/steps_per_day
    return runavg

def time_avg(da, num_days):
    dt = coord_stepsize(da)["time"]
    steps_per_day = int(round(1.0/dt))
    runavg = da.rolling({"time": num_days*steps_per_day}, min_periods=num_days).mean()
    return runavg

def histogram_over_days(da, fmin=None, fmax=None, nbins=20):
    # Given a dataArray with DAILY samples, compute a CDF over each grid cell.
    # TODO: remake this while avoiding loading the whole dataset
    # See xarray.apply_ufunc
    if fmin is None:
        fmin = da.min() - 1e-32
    if fmax is None:
        fmax = da.max() + 1e-32
    bin_edges = np.linspace(fmin, fmax, nbins+1)
    time_axis = da.dims.index("time")
    hist = np.apply_along_axis(
            lambda arr: np.histogram(arr, bins=bin_edges, density=False)[0],
            axis = time_axis,
            arr = da.values,
            )
    # Turn into a DataArray
    hist_coords = dict()
    for key in da.dims:
        if key == "time":
            hist_coords["bin_lower"] = bin_edges[:-1]
        else:
            hist_coords[key] = da.coords[key]
    hist_da = xr.DataArray(coords=hist_coords, data=hist)
    return hist_da

def observable_library():
    obslib = dict()
    obslib["effective_static_stability"] = dict({
        "fun": eff_stat_stab,
        "abbrv": "ESS",
        "unit_symbol": "s$^{-2}$",
        "label": "Effective static stability",
        "cmap": "coolwarm",
        "cmin": None,
        "cmax": None,
        "clo": "gray",
        "chi": "yellow",
        })

    obslib["vertical_velocity"] = dict({
        "fun": vertical_velocity,
        "abbrv": "W",
        "unit_symbol": "Pa/s",
        "label": "Vertical velocity",
        "cmap": "coolwarm",
        "cmin": None,
        "cmax": None,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["meridional_velocity"] = dict({
        "fun": meridional_velocity,
        "abbrv": "V",
        "unit_symbol": "m/s",
        "label": "Meridional velocity",
        "cmap": "coolwarm",
        "cmin": None,
        "cmax": None,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["zonal_velocity"] = dict({
        "fun": zonal_velocity,
        "abbrv": "U",
        "unit_symbol": "m/s",
        "label": "Zonal velocity",
        "cmap": "coolwarm",
        "cmin": None,
        "cmax": None,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["exprec_scaling"] = dict({
        "fun": exprec_scaling_wrapper,
        "abbrv": "XPS",
        "unit_symbol": "mm/day",
        "label": "Extreme precip. scaling",
        "cmap": "Blues",
        "cmin": 0.0,
        "cmax": 64.0,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["convection_rain"] = dict({
        "fun": convection_rain,
        "abbrv": "Rconv",
        "unit_symbol": "mm/day",
        "label": "Convection rain",
        "cmap": "Blues",
        "cmin": 0.0, 
        "cmax": 64.0,
        "clo": "gray", 
        "chi": "yellow",
        })
    obslib["condensation_rain"] = dict({
        "fun": condensation_rain,
        "abbrv": "Rcond",
        "unit_symbol": "mm/day",
        "label": "Condensation rain",
        "cmap": "Blues",
        "cmin": 0.0, 
        "cmax": 64.0,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["total_rain"] = dict({
        "fun": total_rain,
        "abbrv": "Rtot",
        "unit_symbol": "mm/day",
        "label": "Rain rate",
        "cmap": "Blues",
        "cmin": 0.0, 
        "cmax": 64.0,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["specific_humidity"] = dict({
        "fun": specific_humidity,
        "abbrv": "Q",
        "unit_symbol": "kg/kg",
        "label": "Specific humidity",
        "cmap": "Blues",
        "cmin": None,
        "cmax": None,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["temperature"] = dict({
        "fun": temperature,
        "abbrv": "T",
        "unit_symbol": "K",
        "label": "Temperature",
        "cmap": "Reds",
        "cmin": 210.0, 
        "cmax": 350.0,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["column_water_vapor"] = dict({
        "fun": column_water_vapor,
        "abbrv": "CWV",
        "unit_symbol": r"kg m$^{-2}$",
        "label": "Column water vapor",
        "cmap": "Blues",
        "cmin": 0.0, 
        "cmax": 7.0,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["column_relative_humidity"] = dict({
        "fun": column_relative_humidity,
        "abbrv": "CRH",
        "unit_symbol": r"fraction",
        "label": "Column relative humidity",
        "cmap": "Blues",
        "cmin": 0.0, 
        "cmax": 1.0,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["water_vapor_convergence"] = dict({
        "fun": water_vapor_convergence,
        "abbrv": "QCON",
        "unit_symbol": r"kg m$^{-2}$s$^{-1}$",
        "label": "Water vapor convergence",
        "cmap": "coolwarm_r",
        "cmin": None, 
        "cmax": None,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["vorticity"] = dict({
        "fun": vorticity,
        "abbrv": "VOR",
        "unit_symbol": r"s$^{-1}$",
        "label": "Vorticity",
        "cmap": "coolwarm",
        "cmin": None, 
        "cmax": None,
        "clo": "gray",
        "chi": "yellow",
        })
    obslib["surface_pressure"] = dict({
        "fun": surface_pressure,
        "abbrv": "PS",
        "unit_symbol": r"Pa",
        "label": "Surface pressure",
        "cmap": "rainbow",
        "cmin": 96.0e3, 
        "cmax": 103.0e3,
        "clo": "gray",
        "chi": "yellow",
        })
    return obslib

def observable_from_name_vertprof(ds, field_name, latlonsel, t_runavg=1):
    # TODO: make a class or a dictionary for this
    # All fields are daily averaged unless specified otherwise
    # This function is to recover vertical profiles
    if field_name == "convection_rain":
        da = convection_rain(ds.sel(latlonsel))
    elif field_name == "condensation_rain":
        da = condensation_rain(ds.sel(latlonsel))
    elif field_name == "total_rain":
        da = total_rain(ds.sel(latlonsel))
    elif field_name == "vert_deriv_sat_sphum":
        da = vert_deriv_sat_sphum(ds.sel(latlonsel))
    elif field_name == "fn":
        da = fn(ds).sel(latlonsel)
    elif field_name == "specific_humidity":
        da = specific_humidity(ds.sel(latlonsel))
    elif field_name == "temperature":
        da = temperature(ds.sel(latlonsel))
    elif field_name == "vertical_velocity":
        da = vertical_velocity(ds.sel(latlonsel))
    elif field_name == "zonal_velocity":
        da = zonal_velocity(ds.sel(latlonsel))
    elif field_name == "meridional_velocity":
        da = meridional_velocity(ds.sel(latlonsel))
    elif field_name == "surface_pressure":
        da = surface_pressure(ds.sel(latlonsel))
    elif field_name == "column_water_vapor":
        da = column_water_vapor(ds.sel(latlonsel))
    elif field_name == "column_relative_humidity":
        da = column_relative_humidity(ds.sel(latlonsel))
    elif field_name == "water_vapor_convergence":
        da = water_vapor_convergence(ds.sel(latlonsel))
    elif field_name == "vorticity":
        da = vorticity(ds.sel(latlonsel))
    elif field_name == "exprec_scaling":
        da = exprec_scaling_wrapper(ds.sel(latlonsel))
    elif field_name == "effective_static_stability":
        da = eff_stat_stab(ds.sel(latlonsel))

    # Average over the area
    cosine_weight = np.cos(np.deg2rad(da["lat"]))
    cosine_weight *= 1.0/(cosine_weight.sum()*da["lon"].size)
    da = (da*cosine_weight).sum(dim=["lat","lon"])
    # Average over a time horizon if necessary
    if t_runavg != 1:
        da = time_avg(da, t_runavg)
    return da

def observable_from_name(ds, field_name, t_runavg=1, pfull_target=None):
    if pfull_target is None:
        pfull_target = 500
    # TODO: make a class or a dictionary for this
    # All fields are daily averaged unless specified otherwise
    # This function is to recover 2d maps (functions of lat and lon)
    if field_name == "convection_rain":
        da = convection_rain(ds)
    elif field_name == "condensation_rain":
        da = condensation_rain(ds)
    elif field_name == "total_rain":
        da = total_rain(ds)
    elif field_name == "vert_deriv_sat_sphum":
        da = vert_deriv_sat_sphum(ds)
    elif field_name == "fn":
        da = fn(ds)
    elif field_name == "temperature":
        da = temperature(selnear(ds, "pfull", pfull_target))
    elif field_name == "specific_humidity":
        da = specific_humidity(selnear(ds, "pfull", pfull_target))
    elif field_name == "vertical_velocity":
        da = vertical_velocity(selnear(ds, "pfull", pfull_target))
    elif field_name == "zonal_velocity":
        da = zonal_velocity(selnear(ds, "pfull", pfull_target))
    elif field_name == "meridional_velocity":
        da = meridional_velocity(selnear(ds, "pfull", pfull_target))
    elif field_name == "surface_pressure":
        da = surface_pressure(ds)
    elif field_name == "column_water_vapor":
        da = column_water_vapor(ds)
    elif field_name == "column_relative_humidity":
        da = column_relative_humidity(ds)
    elif field_name == "water_vapor_convergence":
        da = water_vapor_convergence(ds)
    elif field_name == "vorticity":
        da = vorticity(selnear(ds, "pfull", pfull_target))
    elif field_name == "exprec_scaling":
        da = exprec_scaling_wrapper(ds)
    elif field_name == "effective_static_stability":
        da = selnear(eff_stat_stab(ds), "pfull", 500)

    # Average over a time horizon if necessary
    if t_runavg != 1:
        da = time_avg(da, t_runavg)
    return da


def combine_histograms(ds, fields2hist, durations2hist):
    # Each entry in field_dict must specify the name of the field, the minimum and maximum, and the number of bins.
    hist_dict = dict()
    for fkey in list(fields2hist.keys()):
        for dur in durations2hist:
            da = observable_from_name(ds, fkey, dur)
            hist_dict[f"{dur}day-{fkey}"] = histogram_over_days(da, fields2hist[fkey]["fmin"], fields2hist[fkey]["fmax"], fields2hist[fkey]["nbins"]).compute()
    return hist_dict

def moments_from_histogram(hist, mom=None):
    # Given a histogram where one of the dimensions is "bin_lower" (and regularly spaced!), compute the quantiles
    if mom is None:
        mom = np.arange(1,4)
    mom_coords = dict()
    for key in hist.dims:
        if key == "bin_lower":
            mom_coords["moment"] = mom
        else:
            mom_coords[key] = hist.coords[key]
    mom = xr.DataArray(coords=mom_coords, dims=list(mom_coords.keys())) 
    bin_width = np.diff(hist["bin_lower"][:2]).item()
    bin_center = hist["bin_lower"] + 0.5*bin_width
    for k in mom["moment"].data:
        mom.loc[dict(moment=k)] = (hist*bin_center**k).sum(dim="bin_lower")/(hist.sum(dim="bin_lower"))
    return mom

def quantiles_from_histogram(hist, qlevels=None):
    # Given a histogram where one of the dimensions is "bin_lower" (and regularly spaced!), compute the quantiles
    if qlevels is None:
        qlevels = np.array([0.25, 0.5, 0.75])
    quant_coords = dict()
    for key in hist.dims:
        if key == "bin_lower":
            quant_coords["quantile"] = qlevels
        else:
            quant_coords[key] = hist.coords[key]
    quant = xr.DataArray(coords=quant_coords, dims=list(quant_coords.keys())) 
    bin_lower = hist["bin_lower"]
    bin_width = np.diff(bin_lower[:2]).item()
    cdf = hist.cumsum(dim="bin_lower")
    normalizer = cdf.isel(bin_lower=-1)
    cdf *= 1.0/normalizer
    for q in qlevels:
        q_lower = (bin_lower.where(cdf >= q)).min(dim="bin_lower")
        print(f"For level {q}, q_lower = {q_lower}")
        q_upper = q_lower + bin_width
        cdf_lower = cdf.where(bin_lower == q_lower).mean(dim="bin_lower")
        cdf_upper = cdf_lower + hist.where(bin_lower == q_lower).mean(dim="bin_lower")/normalizer  # Add only the bit of mass in the bin
        quant.loc[dict(quantile=q)] = q_lower #+ bin_width*(q - cdf_lower)/(cdf_upper - cdf_lower)
    return quant


def zonal_mean_histogram(hist_dict):
    # Aggregate over all points at a given latitude circle and convert to a density.
    # TODO: enable aggregation over a finite-width band
    zm_hist = dict()
    for fkey in list(hist_dict.keys()):
        zm_hist[fkey] = (1.0*hist_dict[fkey]).sum(dim="lon")
        print(f"zm_hist[{fkey}] dims = {zm_hist[fkey].dims}")
        bin_width = np.diff(hist_dict[fkey]["bin_lower"][:2].to_numpy())[0]
        integral = zm_hist[fkey].sum(dim="bin_lower")*bin_width
        zm_hist[fkey] *= 1.0/integral
    return zm_hist

def compute_temp_quantiles(ds):
    durations = np.array([1,3,5])
    quantiles = np.array([0.5,0.99,0.999,0.9999])
    temp_keys = ["temperature"]
    data_vars = dict()
    for key in temp_keys:
        data_vars[key] = xr.DataArray(
                coords={"duration": durations, "quantile": quantiles, "lat": ds["lat"]},
                dims=["duration","quantile","lat"],
                data=np.nan
                )
    for dur in durations:
        print(f"Duration {dur}")
        for key in temp_keys:
            da = observable_from_name(ds, key, dur)
            data_vars[key].loc[dict(duration=dur)] = (
                    da.chunk(dict(time=-1))
                    .quantile(quantiles,dim=["time","lon"])
                    ).compute() 
    temp_quantiles = xr.Dataset(data_vars=data_vars)
    return temp_quantiles

def compute_precip_quantiles(ds, durations=None, quantiles=None):
    if durations is None:
        durations = np.array([1,3,5,7,9,11])
    if quantiles is None:
        quantiles = np.array([0.99,0.999,0.9999])
    rain_keys = ["convection_rain","condensation_rain","total_rain"]
    data_vars = dict()
    for key in rain_keys:
        data_vars[key] = xr.DataArray(
                coords={"duration": durations, "quantile": quantiles, "lat": ds["lat"]},
                dims=["duration","quantile","lat"],
                data=np.nan
                )
    for dur in durations:
        print(f"Duration {dur}")
        for key in rain_keys:
            da = observable_from_name(ds, key, dur)
            data_vars[key].loc[dict(duration=dur)] = (
                    da.chunk(dict(time=-1))
                    .quantile(quantiles,dim=["time","lon"])
                    ).compute() 
    precip_quantiles = xr.Dataset(data_vars=data_vars)
    return precip_quantiles

def compute_observable_quantiles(da, durations=None, quantiles=None):
    if durations is None:
        durations = np.array([1,3,5,7,9,11])
    if quantiles is None:
        quantiles = np.array([0.99,0.999,0.9999])
    q = xr.DataArray(
            coords={"lat": da["lat"], "duration": durations, "quantile": quantiles},
            dims=["lat","duration","quantile"],
            data=0.0
            )
    for dur in durations:
        print(f"Duration {dur}")
        da_avg = da.copy()
        for lag in range(1,dur):
            da_avg += da.shift(time=1)
        da_avg *= 1.0/dur
        q.loc[dict(duration=dur)] = (
                da_avg.quantile(quantiles,dim=["time","lon"])
                ).transpose("lat","quantile")
    return q

def compute_precip_histograms(ds,savefolder):
    fields2hist = dict({
        "convection_rain": dict({"fmin": 0, "fmax": 70, "nbins": 150,}),
        "condensation_rain": dict({"fmin": 0, "fmax": 70, "nbins": 150}),
        "total_rain": dict({"fmin": 0, "fmax": 70, "nbins": 150}),
        })
    durations2hist = np.array([1,3,5])
    hist_dict = dict()
    for fkey in list(fields2hist.keys()):
        for dur in durations2hist:
            da = observable_from_name(ds, fkey, dur)
            hist_dict[f"{dur}day-{fkey}"] = histogram_over_days(da, fields2hist[fkey]["fmin"], fields2hist[fkey]["fmax"], fields2hist[fkey]["nbins"]).compute()
    for fkey in list(hist_dict.keys()):
        print(f"About to save the histogram for {fkey}")
        hist_dict[fkey].to_netcdf(join(savefolder,f"hist_{fkey}.nc"))
        print(f"Just saved the histogram for {fkey}")
    return hist_dict

def ingest_history_aggregated(histdir, pattern="history*.nc"):
    files2open = glob.glob(f"{histdir}/{pattern}")
    ds = xr.open_mfdataset(files2open, decode_times=False)
    return ds

def ingest_history(histdir,pattern="d*h00",verbose=False):
    freqs = ["1xday","4xday"]
    # Return a Dask DataArray from all the netcdfs (1xday and 4xday) under histdir
    chunk_dirs = glob.glob(f"{histdir}/{pattern}")
    if verbose:
        print(f"chunk_dirs = {chunk_dirs}")
    ds = ingest_timechunks(chunk_dirs)
    return ds

def ingest_timechunks(chunk_dirs):
    freqs = ["1xday","4xday"]
    file_list = dict({freq: [] for freq in freqs})
    for chd in chunk_dirs:
        for freq in ["1xday","4xday"]:
            file_list[freq] += glob.glob(f"{chd}/*{freq}*.nc*")
    ds = dict()
    for freq in freqs:
        ds[freq] = xr.open_mfdataset(file_list[freq], decode_times=False, preprocess=preprocess, parallel=True)
    return ds

def ingest_multiple_histories(histdir_list, pattern="d*h00"):
    freqs = ["1xday","4xday"]
    # Return a Dask DataArray from all the netcdfs (1xday and 4xday) under histdir
    file_list = dict({freq: [] for freq in freqs})
    for histdir in histdir_list:
        chunk_dirs = glob.glob(f"{histdir}/{pattern}")
        for chd in chunk_dirs:
            for freq in ["1xday","4xday"]:
                file_list[freq] += glob.glob(f"{chd}/*{freq}*.nc*")
    ds = dict()
    for freq in freqs:
        ds[freq] = xr.open_mfdataset(file_list[freq], decode_times=False, preprocess=preprocess)
    return ds

def process_history_folder(rundir, fields2comp, overwrite=False):
    # rundir/history is the name of the histdir
    # rundir/features is the name of the feature output
    histdir = join(rundir,"history")
    print(f"does histdir exist? {exists(histdir)}")
    savefolder = join(rundir,"analysis","features")
    os.makedirs(savefolder, exist_ok=True)
    ds = ingest_history(histdir)
    precompute_features(ds, fields2comp, savefolder, overwrite=overwrite)
    return


if __name__ == "__main__":
    fields2comp = dict({
        "1xday": [
            "total_rain"
        ],
        "4xday": [
            #"exprec_scaling",
            "vertical_velocity",
            "temperature",
            "zonal_velocity",
            "meridional_velocity",
            "surface_pressure",
            "vorticity",
            "column_relative_humidity",
            "effective_static_stability",
            "water_vapor_convergence", 
        ],
    })
    rundir = "/pool001/ju26596/fms_archive/2022-12-13/3/ctrl_bmconv_21x100_8proc/abs1.0_smooth/mem_ctrl"
    process_history_folder(rundir, fields2comp)

def estimate_return_time_dns(ds, func2max, block_size):
    # Estimate return period and return level from a long simulation, using the modified block maximum method of Lestang et al 2018
    # TODO: determine whether the function should be evaluated before or after dividing into blocks
    fds = func2max(ds)
    num_blocks = int((fds["time"][-1] - fds["time"][0])/block_size)
    print(f"num_blocks = {num_blocks}")
    block_dividers = np.linspace(0, len(fds["time"])-1, num_blocks+1).astype(int)
    block_maxima = np.zeros(num_blocks)
    block_sizes = np.zeros(num_blocks)
    for i_block in range(num_blocks):
        bm = fds.isel(time=slice(block_dividers[i_block], block_dividers[i_block+1]))
        dt = bm["time"][1].item() - bm["time"][0].item()
        block_sizes[i_block] = np.isfinite(bm).sum().compute().item() * dt
        block_maxima[i_block] = bm.max(dim="time").compute().item()
    # Delete any blocks with shorter block sizes due to time integrating
    max_block_size = np.max(block_sizes)
    idx = np.where(block_sizes == max_block_size)[0]
    num_full_blocks = len(idx)
    block_maxima = block_maxima[idx]
    exrt = np.nan*np.ones(len(idx))
    order = np.argsort(block_maxima)
    thresh_list = block_maxima[order]
    prob_exceedance = np.arange(1, num_full_blocks+1)[::-1] / num_full_blocks
    idx_nontrivial = np.where((prob_exceedance > 0) * (prob_exceedance < 1))
    exrt[idx_nontrivial] = -max_block_size / np.log(1 - prob_exceedance[idx_nontrivial])
    return thresh_list,exrt        




