import numpy as np
from bed_inversion import *
from Kronebreen_set_topography import *
import PISM
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    ctx = PISM.Context()
    dt = .1
    
    setup_file = "Kronebreen_initial_setup.nc"
    output_file = "Kronebreen_build_output.nc"
    
    setup = set_topography()
    write_setup_to_nc(setup, setup_file)
    options = {
        "-Mz": 30,
        "-Lz": 5000,
        #"-z_spacing": "equal",
        "-surface" : "given",
        "-atmosphere.given.file": setup_file,
        "-surface.given.file": setup_file,
        "-ocean.given.file": setup_file,
         "-i": setup_file,
        "-bootstrap": "",
        "-energy": "none",
        "-sia_flow_law": "isothermal_glen",
        "-ssa_flow_law": "isothermal_glen",
        "-stress_balance": "ssa+sia",
        "-yield_stress": "constant",
        "-pseudo_plastic": "",
        "-pseudo_plastic_q": 0.333333,
        "-pseudo_plastic_uthreshold": 3.1556926e7,
        "-yield_stress": "constant",
        "geometry.update.use_basal_melt_rate": "no",
         "stress_balance.ssa.compute_surface_gradient_inward": "no",
         "flow_law.isothermal_Glen.ice_softness": 1.2597213016951452e-24,
         "constants.ice.density": 900.,
       "constants.sea_water.density": 1000.,
         "bootstrapping.defaults.geothermal_flux": 0.0,
         "stress_balance.ssa.Glen_exponent": 3.,
         "constants.standard_gravity": 9.81,
         "ocean.sub_shelf_heat_flux_into_ice": 0.0,
         "stress_balance.sia.bed_smoother.range": 0.0,
        "sea_level.constant.value": -1e4,
        "-bed_def": "iso",
        "bed_deformation.mantle_density": 1e20,
        "-o": output_file,
        "output.size": "big"
    }
    
    model = create_pism(setup_file, options)

    topg_with_ghosts = model.bed_deformation_model().bed_elevation().local_part()[:]
    thk_with_ghosts =  model.geometry().ice_thickness.local_part()[:]
    tauc_with_ghosts = model.basal_yield_stress_model().basal_material_yield_stress().local_part()[:]

    (H, mask, u_surface, v_surface, tauc) = run_pism(model, dt, topg_with_ghosts, thk_with_ghosts, tauc_with_ghosts)

    model.save_results()
