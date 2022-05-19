#!/usr/bin/env python3
import PISM
import numpy as np
from scipy import ndimage
from netCDF4 import Dataset as NC
from funcs import *

def get_nc_data(file, var, time):
    ds = NC(file)
    avail_vars = [nc_var for nc_var in ds.variables]
    if var not in avail_vars:
        raise ValueError('variable not found; must be in {}'.format(avail_vars))
    else:
        var_data = ds[var][time][:]
    return var_data

ctx = PISM.Context()

def create_pism(input_file, options):
    """Allocate and initialize PISM, using bed topography 'topg' in the
       `input_file` to set the horizontal grid.

    `options` is a dictionary containing command-line options.
    """

    # set command line options
    opt = PISM.PETSc.Options()
    for key, value in options.items():
        opt.setValue(key, value)

    PISM.set_config_from_options(ctx.unit_system, ctx.config)

    # get horizontal grid parameters from the input file
    grid_parameters = PISM.GridParameters(ctx.ctx, input_file, "topg", PISM.CELL_CENTER)

    # get vertical grid parameters from command line options
    grid_parameters.vertical_grid_from_options(ctx.config)
    grid_parameters.ownership_ranges_from_options(ctx.ctx.size())

    # allocate the computational grid
    grid = PISM.IceGrid(ctx.ctx, grid_parameters)

    model = PISM.IceModel(grid, ctx.ctx)
    model.init()

    return model

def run_pism(pism, dt_years, bed_elevation, ice_thickness, yield_stress):
    """Run PISM for `dt_years` years using provided `bed_elevation` and `ice_thickness`.

    Here `bed_elevation` and `ice_thickness` are NumPy arrays
    containing the local (sub-domain) parts of these fields,
    *including* the two grid cell wide border or ghosts.

    returns a tuple containing updated (ice_thickness, mask, u_surface, v_surface)

    """

    # Copy ice thickness and bed elevation into PISM's storage
    #
    # Note that this abuses "const" access to various parts of PISM.

    # bed deformation models are in charge of bed elevation
    pism.bed_deformation_model().bed_elevation().local_part()[:] = bed_elevation
    
    # we also need to update this copy of bed elevation (for consistency)
    pism.geometry().bed_elevation.local_part()[:] = bed_elevation
    
    # pism.geometry() stores ice thickness
    pism.geometry().ice_thickness.local_part()[:] = ice_thickness
    
    H_min = ctx.config.get_number("geometry.ice_free_thickness_standard")
    pism.geometry().ensure_consistency(H_min)

    # set basal yield stress
    pism.basal_yield_stress_model().basal_material_yield_stress().local_part()[:] = yield_stress

    dt = PISM.util.convert(dt_years, "year", "second")
    pism.run_to(ctx.time.current() + dt)

    # get new ice thickness from PISM:
    H    = pism.geometry().ice_thickness.local_part()
    H    = np.array(H, copy=True)

    # get new cell type mask from PISM:
    mask = pism.geometry().cell_type.local_part()
    mask = np.array(mask, copy=True)

    # get basal yield stress from PISM:
    tauc = pism.basal_yield_stress_model().basal_material_yield_stress().local_part()
    tauc = np.array(tauc, copy=True)

    # compute surface velocity:
    stress_balance_model = pism.stress_balance()
    diags                = stress_balance_model.diagnostics()
    velsurf              = diags["velsurf"].compute().local_part()

    # extract U and V components and paste them into arrays with
    # "ghosts" to ensure that all arrays have the same shape:

    # stencil width:
    w = 2
    u_surface = np.zeros_like(H)
    u_surface[w:-w, w:-w] = velsurf[:, :, 0]

    v_surface = np.zeros_like(H)
    v_surface[w:-w, w:-w] = velsurf[:, :, 1]

    return (H, mask, u_surface, v_surface, tauc)

def iteration_old(bed, usurf, yield_stress, mask, dh_ref, dt, beta, bw, options):
    
    h_old = usurf - bed

    model = create_pism('input.nc', options)
    
    (thk_mod, mask_iter, u_rec, v_rec, tauc) = run_pism(model, dt, bed, h_old, yield_stress)

    dh_rec = (thk_mod - h_old)/dt
    
    misfit = dh_rec - dh_ref        

    B_rec = bed - beta * misfit
    S_rec = usurf #+ beta * 0.01 * misfit

    ### buffer ###
    k = np.ones((bw, bw))
    buffer = ndimage.convolve(mask_iter, k)/(bw)**2 
    criterion = np.logical_and(np.logical_and(buffer > 0, buffer != 1), mask==1)
    B_rec[criterion] = 0
    ### buffer end ###

    B_rec[mask==0] = bed[mask==0]
    S_rec[mask==0] = usurf[mask==0]
    B_rec[B_rec>S_rec] = S_rec[B_rec>S_rec]

    return B_rec, S_rec
    
def iteration(model, bed, usurf, yield_stress, mask, dh_ref, vel_ref, dt, beta, bw, update_friction):
        
    h_old = usurf - bed

    # run PISM forward for dt years
    (h_rec, mask_iter, u_rec, v_rec, tauc_rec) = run_pism(model, dt, bed, h_old, yield_stress)

    # calculate modelled dh/dt
    dh_rec = (h_rec - h_old)/dt
    
    # calculate dh/dt misfit and shift it
    misfit = dh_rec - dh_ref
    misfit = shift(misfit, u_rec, v_rec, mask, .3)
    
    # apply bed and surface corrections
    B_rec = bed - beta * misfit
    S_rec = usurf + beta * 0.03 * misfit
    
    # interpolate around ice margin
    k = np.ones((bw, bw))
    buffer = ndimage.convolve(mask_iter, k)/(bw)**2 
    criterion = np.logical_and(np.logical_and(buffer > 0, buffer != 2), mask == 1)
    B_rec[criterion]=0
    
    # mask out 
    B_rec[mask==0] = bed[mask==0]
    S_rec[mask==0] = usurf[mask==0]
    B_rec[B_rec>S_rec] = S_rec[B_rec>S_rec]

    if update_friction == 'yes':   

        vel_rec = np.sqrt(u_rec**2+v_rec**2)
        vel_mismatch = np.maximum(np.minimum((vel_rec - vel_ref)/vel_ref, 0.5), -0.5)
        vel_mismatch[mask==0]=np.nan
        vel_mismatch =  gauss_filter(vel_mismatch, .6,2)
        vel_mismatch[np.isnan(vel_mismatch)]=0
        tauc_rec += vel_mismatch * tauc_rec  
    
    return B_rec, S_rec, tauc_rec, misfit

if __name__ == "__main__":
    input_file = PISM.OptionString("-i", "input file name")

    if not input_file.is_set():
        raise RuntimeError("-i is required")

    options = {
        "-Mz": 21,
        "-Lz": 4000,
        "-z_spacing": "equal",
        "-surface" : "given",
        "-surface_given_file": input_file.value(),
        "-i": input_file.value(),
        "-bootstrap": "",
        "-energy": "none",
        "-sia_flow_law": "isothermal_glen",
        "-ssa_flow_law": "isothermal_glen",
        "-stress_balance": "ssa+sia",
        "-yield_stress": "constant"
    }

    pism = create_pism("input.nc", options)

    H   = np.array(pism.geometry().ice_thickness.local_part(), copy=True)
    bed = np.array(pism.geometry().bed_elevation.local_part(), copy=True)

    tauc = np.array(pism.basal_yield_stress_model().basal_material_yield_stress().local_part(),
                    copy=True)

    tauc[10:-10,:] *= 0.95

    bed[10:-10,:] -= 1

    dt_years = 1
    (H, mask, u_surface, v_surface, tauc) = run_pism(pism, dt_years, bed, H, tauc)

    u_surface[mask == 0] = np.nan
    u_surface[mask == 4] = np.nan
    v_surface[mask == 0] = np.nan
    v_surface[mask == 4] = np.nan

    import pylab as plt

    for each in (H, mask, u_surface, v_surface, tauc):
        plt.figure()
        plt.imshow(each)
        plt.colorbar()
    plt.show()
