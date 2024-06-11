import numpy as np
from scipy.sparse import csr_matrix
import math
from joblib import Parallel, delayed

#####################
# Auxiliary functions
#####################

def bound_index(x,Nx):
  # restrict coordinate x to matrix limit [0,Nx-1]
  if x<0:
    x = 0
  if x>(Nx-1):
    x = Nx-1
  return x

def bound_weight(w,x,y,Nx,Ny):
  # null weights outside the FOV
  if x<0 or x> (Nx-1) or y<0 or y> (Ny-1):
    w = 0
  return w  

def lin_index(x,y,Nx):
  # get 1D linear index from 2D index
  return int(x + Nx*y)

#####################
# Key function !!!
#####################

# get_sparse_motion_matrix takes a [Nx,Ny,2] flow field (i.e no underlying mesh)
# and creates the corresponding [Nx*Ny Nx*Ny] sparse motion matrix

def get_sparse_motion_matrix(flow_field):
# creates a sparse motion matrix corresponding to the motion in the flow field
# assuming linear interpolation
  def __get_index_and_value(x, y, Nx, Ny):
    ux = flow_field[x,y,0]
    uy = flow_field[x,y,1]
    y1 = math.floor(y+uy)
    y2 = math.floor(y+uy+1)
    x1 = math.floor(x+ux)
    x2 = math.floor(x+ux+1)
    # interpolants for linear interpolation
    wx = ux-math.floor(ux)
    wy = uy-math.floor(uy)
    w11 = bound_weight((1-wx)*(1-wy),x1,y1,Nx,Ny)
    w12 = bound_weight((1-wx)*wy,x1,y2,Nx,Ny)
    w21 = bound_weight(wx*(1-wy),x2,y1,Nx,Ny)
    w22 = bound_weight(wx*wy,x2,y2,Nx,Ny)
    # avoiding out of FOV issues
    y1 = bound_index(y1,Ny)
    y2 = bound_index(y2,Ny)
    x1 = bound_index(x1,Nx)
    x2 = bound_index(x2,Nx)
    # loading sparse matrix indexes
    li = int(lin_index(x,y,Nx))
    x1y1 = int(lin_index(x1,y1,Nx))
    x1y2 = int(lin_index(x1,y2,Nx))
    x2y1 = int(lin_index(x2,y1,Nx))
    x2y2 = int(lin_index(x2,y2,Nx))
    row, col, val = [], [], []
    # assigning weights to sparse matrix
    if w11 != 0:
      row.append(li)
      col.append(x1y1)
      val.append(w11)
    if w12 != 0:
      row.append(li)
      col.append(x1y2)
      val.append(w12)
    if w21 != 0:
      row.append(li)
      col.append(x2y1)
      val.append(w21)
    if w22 != 0:
      row.append(li)
      col.append(x2y2)
      val.append(w22)
    return (np.array(row), np.array(col), np.array(val))

  Ny = np.shape(flow_field)[1]
  Nx = np.shape(flow_field)[0]

  r = Parallel(n_jobs=-1)(delayed(__get_index_and_value)(x, y, Nx, Ny) for x in range(np.shape(flow_field)[0]) for y in range(np.shape(flow_field)[1]))
  rows, cols, vals = zip(*r)
  rows, cols, vals = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
  sparse_mot = csr_matrix((vals, (rows, cols)), shape=(Ny*Nx,Ny*Nx))

  return sparse_mot

#####################
# Key function !!!
#####################

# apply_sparse_motion takes a [Nx,Ny] image and a [Nx*Ny,Nx*Ny] spr_mat and applies
# the corresponding motion. adj_flag determines if the forward or transpose motion
# is applied (tranpose should be a good approximation of the inverse)
def apply_sparse_motion(img,spr_mat,adj_flag):
    # applies a motion field via the sparse representation
    Ny = np.shape(img)[1]
    Nx = np.shape(img)[0]
    img = np.reshape(img,(Nx*Ny,1),order='F')
    # flag == 1 means we apply transpose (i.e. inverse) motion
    if adj_flag == 1:
        spr_mat = spr_mat.transpose()
    # apply motion
    img_r = spr_mat * np.real(img)
    img_i = spr_mat * np.imag(img)
    # apply correction pertaining to errors in discrete interpolations with large jacobians
    m_norm = np.ones((Nx*Ny,1))
    m_norm = spr_mat * m_norm
    np.seterr(divide='ignore', invalid='ignore')
    img_r = np.divide(img_r,m_norm)
    img_i = np.divide(img_i,m_norm)
    # mind nans
    img_r = np.nan_to_num(img_r)
    img_i = np.nan_to_num(img_i)
    img = np.vectorize(complex)(img_r[:,0], img_i[:,0])
    img = np.reshape(img,(Nx,Ny),order='F')
    return img