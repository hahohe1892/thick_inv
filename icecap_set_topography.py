import numpy as np
from netCDF4 import Dataset as NC

def set_topography():

    """ creates standard setup on the standard grid

    returns: 
      dictionary containing:
        2D numpy arrays of:
          - topg
          - tauc
          - ice surface temperature
          - smb
          - thickness
          - surface
       1D numpy arrays of x and y
    """
    
    Lx = 2 * 25e3  # in m
    Ly = 2 * 25e3  # in m

    dx, dy = 1e3,1e3

    # grid size: # of boxes
    ny = int(np.floor(Lx / dx) + 1)  # make it an odd number
    nx = int(np.floor(Ly / dy) + 1)  # make it an odd number

    # grid size: extent in km's, origin (0,0) in the center of the domain
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)

    nxcenter = int(np.floor(0.5 * nx))
    nycenter = int(np.floor(0.5 * ny))

    surf = np.ones((ny, nx))
    topg = np.zeros((ny, nx))
    tauc = np.zeros((ny, nx))
    ice_surface_temp = np.ones((ny, nx))*268.15
    smb = np.zeros((ny, nx))

    for i in range(0, nx):
        for j in range(0, ny):
            dist = ((x[i])**2+(y[j])**2)**0.5
            dist2a=((x[i]-Lx/5)**2+(y[j]-Ly/5)**2)**0.5
            dist2b=((x[i]+Lx/5)**2+(y[j]+Ly/5)**2)**0.5
            dist2c = ((2*(x[i])+Lx/5)**2+(2*(y[j])-Ly/5)**2)**0.5
            dist2d = ((2*(x[i])-Lx/5)**2+(2*(y[j])+Ly/5)**2)**0.5
            topg[i, j] = (np.maximum(500*(1-dist2a*5/Lx),0)+np.maximum(500*(1-dist2b*5/Lx),0)+np.maximum(500*(1-dist2c*5/Lx),0)+np.maximum(500*(1-dist2d*5/Lx),0))
            smb[i, j] = 5 *(1-(dist*2/Lx))
            surf[i,j] = topg[i,j]+1

    tauc = np.ones((ny,nx))*5e7
    for i in range(0,nx):
        for j in range(0, ny):
            if j<=24:
                tauc[i,j] -= 4e7*np.exp((-(x[i])**2)/(2*10000**2))
            dist = ((x[i])**2+(y[j]-15000)**2)**0.5

    thk = np.ones((ny,nx))
    surf = topg+thk
    tauc = np.ones_like(topg)*1e10
    mask = np.zeros_like(topg)
    setup = {'topg': topg, 'usurf': surf, 'mask': mask, 'tauc': tauc, 'velsurf_mag': np.zeros_like(topg), 'ice_surface_temp': ice_surface_temp, 'smb': smb, 'thk': thk, 'x': x, 'y': y}

    return setup
    
def write_setup_to_nc(setup, output_file):
    """ takes dictionary with required input fields 
    (i.e. topg, tauc, ice_surface_temp, smb, thk, x, y)
    and writes an nc file that can be used to initialize PISM """
    
    vars = {'y':    ['m',
                 'y-coordinate in Cartesian system',
                 'projection_y_coordinate',
                 None,
                setup['y']],
        'x':    ['m',
                 'x-coordinate in Cartesian system',
                 'projection_x_coordinate',
                 None,
                 setup['x']],
        'thk':  ['m',
                 'floating ice shelf thickness',
                 'land_ice_thickness',
                 None,
                 setup['thk']],
        'topg': ['m',
                 'bedrock surface elevation',
                 'bedrock_altitude',
                 None,
                 setup['topg']],
        'ice_surface_temp': ['K',
                             'annual mean air temperature at ice surface',
                             'surface_temperature',
                             None,
                             setup['ice_surface_temp']],
        'climatic_mass_balance': ['kg m-2 year-1',
                                  'mean annual net ice equivalent accumulation rate',
                                  'land_ice_surface_specific_mass_balance_flux',
                                  None,
                                  setup['smb'] * 900],
        'tauc': ['Pa',
                 'yield stress for basal till (plastic or pseudo-plastic model)',
                 'yield stress',
                 None,
                 setup['tauc']],
        'mask': ['',
                 'ice-type (ice-free/grounded/floating/ocean) integer mask',
                 'mask',
                 None,
                 setup['mask']],
        'usurf': ['m',
                 'ice top surface elevation',
                 'surface_altitude',
                 None,
                  setup['usurf']],
        'velsurf_mag': ['m year-1',
                        'magnitude of horizontal velocity of ice at ice surface',
                        'surface velocity',
                        None,
                        setup['velsurf_mag']]
        }
    
    ncfile = NC(output_file, 'w', format='NETCDF3_CLASSIC')
    xdim = ncfile.createDimension('x', int(setup['x'].shape[0]))
    ydim = ncfile.createDimension('y', int(setup['y'].shape[0]))
    
    for name in list(vars.keys()):
        [_, _, _, fill_value, data] = vars[name]
        if name in ['x', 'y']:
            var = ncfile.createVariable(name, 'f4', (name,))
        else:
            var = ncfile.createVariable(name, 'f4', ('y', 'x'), fill_value=fill_value)
        for each in zip(['units', 'long_name', 'standard_name'], vars[name]):
            if each[1]:
                setattr(var, each[0], each[1])
        var[:] = data
    
    # finish up
    ncfile.close()
    print("NetCDF file ", output_file, " created")
    print('')
