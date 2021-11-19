### inpaint_nans function from matlab ported to Python ###
import scipy.sparse as sps
from scipy.sparse import linalg
import numpy as np
from numpy import matlib
from copy import copy
from netCDF4 import Dataset as NC
from scipy.interpolate import griddata

def identify_neighbors(n,m, nan_list, talks_to):
    nan_count = np.shape(nan_list)[0]
    talk_count = np.shape(talks_to)[0]
    nn = np.zeros((nan_count*talk_count,2), dtype='int')
    j=[0, nan_count]
    for i in range(talk_count):
        nn[j[0]:j[1],:]=nan_list[:,1:3]+matlib.repmat(talks_to[i], nan_count,1)
        j[0]+=nan_count
        j[1]+=nan_count
    L = np.logical_or(np.logical_or((nn[:,0]<0), (nn[:,0]>n)), np.logical_or((nn[:,1]<0), (nn[:,1]>m)))
    nn = nn[~L]
    neighbors_list = np.zeros((np.shape(nn)[0],3), dtype='int')
    neighbors_list[:,0] = np.ravel_multi_index((nn[:,0],nn[:,1]), (n, m))
    neighbors_list[:,1] = nn[:,0]
    neighbors_list[:,2] = nn[:,1]
    neighbors_list = np.unique(neighbors_list, axis=0)
    nrows, ncols = neighbors_list.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [neighbors_list.dtype]}

    neighbors_list_final = np.setdiff1d(neighbors_list.view(dtype), nan_list.view(dtype))
    neighbors_list_final = neighbors_list_final.view(neighbors_list.dtype).reshape(-1, ncols)
    return neighbors_list_final

def inpaint_nans(B_rec):
    n, m = np.shape(B_rec)
    A=B_rec.flatten()
    nm=n*m
    k = np.isnan(A)
    nan_list_unrolled = np.where(k)[0]
    known_list = np.where(~np.isnan(A))[0]
    nan_count = len(nan_list_unrolled)
    nr = np.argwhere(np.isnan(B_rec))[:,0]
    nc = np.argwhere(np.isnan(B_rec))[:,1]
    nan_list = np.zeros((nan_count,3), dtype='int')
    nan_list[:,0] = nan_list_unrolled
    nan_list[:,1] = nr
    nan_list[:,2] = nc

    talks_to = [[-1,0], [0,-1],[1,0], [0,1]]
    neighbors_list = identify_neighbors(n, m, nan_list, talks_to)
    all_list = np.concatenate((nan_list, neighbors_list), axis=0)
    L = np.where(np.logical_and(all_list[:,1]>0, all_list[:,1]<n))[0]
    nl=np.shape(L)[0]
    if nl>0:
        down = sps.csr_matrix((np.ones(nl), (all_list[L,0], all_list[L,0]-1)), shape=(nm,nm))
        middle = sps.csr_matrix((np.ones(nl)*-2, (all_list[L,0], all_list[L,0])), shape=(nm,nm))
        up = sps.csr_matrix((np.ones(nl), (all_list[L,0], all_list[L,0]+1)), shape=(nm,nm))
        fda = down+middle+up
    else:
        fda = sps.csr_matrix((np.zeros(nm), (np.arange(nm), np.arange(nm))))
    L = np.where(np.logical_and(all_list[:,2]>0, all_list[:,2]<m))[0]
    nl=np.shape(L)[0]
    if nl>0:
        down = sps.csr_matrix((np.ones(nl), (all_list[L,0], np.maximum(0, all_list[L,0]-m))), shape=(nm,nm))
        middle = sps.csr_matrix((np.ones(nl)*-2, (all_list[L,0], all_list[L,0])), shape=(nm,nm))
        up = sps.csr_matrix((np.ones(nl), (all_list[L,0], all_list[L,0]+m)), shape=(nm,nm))
        fda+=down+middle+up

    rhs = -fda[:,known_list]*A[known_list]
    k=np.argwhere(np.any(fda[:,nan_list[:,0]]))[:,0]

    B=copy(A)
    fda = fda[:,nan_list[:,0]]
    B[nan_list[:,0]] = sps.linalg.lsqr(fda[k], rhs[k])[0]
    B = np.reshape(B, (n,m))
    return B


def get_nc_data(file, var, time):
    ds = NC(file)
    avail_vars = [nc_var for nc_var in ds.variables]
    if var not in avail_vars:
        raise ValueError('variable not found; must be in {}'.format(avail_vars))
    else:
        var_data = ds[var][time][:]
    return var_data

def shift(data, u, v, dx):
    x_shift, y_shift = np.meshgrid(range(nx), range(ny))
    uv_mag = np.ones_like(u)
    uv_mag[np.logical_or(u!=0, v!=0)] = np.sqrt(u[np.logical_or(u!=0, v!=0)]**2+v[np.logical_or(u!=0, v!=0)]**2)
    x_shift = x_shift+(u/uv_mag)*dx
    y_shift = y_shift+(v/uv_mag)*dx

    points = np.zeros((np.shape(u)[0]*np.shape(u)[1],2))
    points[:,0] = x_shift.flatten()
    points[:,1]=y_shift.flatten()
    xi, yi = np.meshgrid(range(nx), range(ny))

    newgrid = griddata(points, data.flatten(), (xi.flatten(), yi.flatten())).reshape(np.shape(u))
    return newgrid

from IPython.display import display, Javascript
import time
import hashlib

def save_and_push(notebook_path, branch_name, nc_file, commit_message):
        
    current_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode('ascii').strip()
    if current_branch != branch_name:
        raise ValueError('not on correct branch')
                        
    start_md5 = hashlib.md5(open(notebook_path,'rb').read()).hexdigest()
    display(Javascript('IPython.notebook.save_checkpoint();'))
    current_md5 = start_md5
                                        
    while start_md5 == current_md5:
        time.sleep(1)
        current_md5 = hashlib.md5(open(notebook_path,'rb').read()).hexdigest()
                                                             
    hashmark =  subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
                                                                                    
    save_model = ["cp", "".format(nc_file), "./models/{}.nc".format(hashmark)]
    stage = ["git", "add", "{}".format(notebook_path)]
    commit = ["git", "commit", "-m", commit_message]
    try:
        proc = subprocess.check_output(stage, stderr=subprocess.STDOUT)
        proc = subprocess.check_output(commit, stderr=subprocess.STDOUT)
        proc = subprocess.check_output(save_model, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise ValueError('something went wrong')
