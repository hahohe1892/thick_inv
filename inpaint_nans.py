### inpaint_nans function from matlab ported to Python ###
import scipy.sparse as sps
from scipy.sparse import linalg
import numpy as np
from numpy import matlib
from copy import copy
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
