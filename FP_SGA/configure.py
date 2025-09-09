import numpy as np
import torch
import scipy.io as scio
import torch.nn as nn
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

problem =  'OU'
#'Burgers'
# 'PDE_divide'
#  'PDE_compound'
#  'chafee-infante'
#  'Kdv'
#  'OU'
#  'DW'
#  'CIR'



seed = 13
device = torch.device('cuda:0')
# device = torch.device('cpu')

# AIC hyperparameter
aic_ratio = 0.5  # lower this ratio, less important is the number of elements to AIC value



def smooth_pdf(u_orig, spatial_sigma=1.0, temp_sigma=1.0):
    """2D Gaussian smoothing in both spatial and temporal directions"""
    # Apply 2D Gaussian smoothing using separable 1D filters
    u_smooth = gaussian_filter(u_orig, sigma=(spatial_sigma, temp_sigma), mode='mirror', axes=(0, 1))
    
    return u_smooth

def truncate_edges(u, x, t, percent=0.05):
    """
    Truncate `percent` from each edge in BOTH spatial (axis 0) and temporal (axis 1).
    Returns:
      - u_trunc:  truncated 2D array
      - x_trunc:  truncated spatial vector
      - t_trunc:  truncated temporal vector
    """
    n_x, n_t = u.shape
    cut_x = int(n_x * percent)
    cut_t = int(n_t * percent)
    
    u_trunc = u[cut_x:-cut_x, cut_t:-cut_t]
    x_trunc = x[cut_x:-cut_x]
    t_trunc = t[cut_t:-cut_t]
    return u_trunc, x_trunc, t_trunc


if problem == 'OU':
    data = np.load('./data/ou_pde_standard.npz')#ou_pde_sde: 1e5 sample; ou_pde_sde2: 1e6 sample; ou_pde: analytic; ou_pde_standard: all coef = 1
    u = data['usol']
    x = data['x']
    t = data['t']
    u  = smooth_pdf(u, 1, 9)
    u,x,t = truncate_edges(u, x, t, 0.05)
    
if problem == 'DW': #dX=(aX-bX^3)dt+cdW. 1:(1,0.1,1);2:(1.5,1,0.5);3:(1,1,1);4:(1,1,0.1);5:(0.1,1,0.1);6:(0.7,1,0.5)
    data = np.load('./data/sde_double_well_3.npz')
    u = data['pdf']
    x = data['x']
    t = data['t']
    u  = smooth_pdf(u, 1, 5)
    u,x,t = truncate_edges(u, x, t, 0.05)

if problem == 'CIR': #dX=a(b-x)dt+c*sqrt(x)dW. 1:(1,1,0.3);2:(1,1,1);3:(1,1,2);4(1,1,1.2);5:(1,1,3);6:(1,1,1)inti:(3,0.5);7:(0,0,1)
    data = np.load('./data/sde_CIR_5.npz')
    u = data['pdf']
    x = data['x']
    t = data['t']
    u  = smooth_pdf(u, 5, 5)
    u,x,t = truncate_edges(u, x, t, 0.05)

if problem == 'Null':
    nx = 150
    nt = 200
    x=np.linspace(1,2,nx)
    t=np.linspace(0,1,nt)
    u=np.zeros((nx, nt))

# PDE-1: Ut= -Ux/x + 0.25Uxx
if problem == 'PDE_divide':
    u=np.load("./data/PDE_divide.npy").T
    nx = 100
    nt = 251
    x=np.linspace(1,2,nx)
    t=np.linspace(0,1,nt)

# PDE-3: Ut= d(uux)(x)
if problem == 'PDE_compound':
    u=np.load("./data/PDE_compound.npy").T
    nx = 100
    nt = 251
    x=np.linspace(1,2,nx)
    t=np.linspace(0,0.5,nt)    
    
# Burgers -u*ux+0.1*uxx
if problem == 'Burgers':
    data = scio.loadmat('./data/burgers.mat')
    u=data.get("usol")
    x=np.squeeze(data.get("x"))
    t=np.squeeze(data.get("t").reshape(1,201))

# # Kdv -0.0025uxxx-uux
if problem == 'Kdv':
    data = scio.loadmat('./data/Kdv.mat')
    u=data.get("uu")
    x=np.squeeze(data.get("x"))
    t=np.squeeze(data.get("tt").reshape(1,201))

# chafee-infante   u_t=u_xx-u+u**3
if problem == 'chafee-infante': # 301*200的新数据
    u = np.load("./data/chafee_infante_CI.npy")
    x = np.load("./data/chafee_infante_x.npy")
    t = np.load("./data/chafee_infante_t.npy")



# plt.contourf(t, x, u)
# plt.colorbar()
# plt.xlabel('Time (t)')
# plt.ylabel('Position (x)')
# plt.show()