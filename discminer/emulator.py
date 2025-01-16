import numpy as np
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append('../unet_emu')

from create_model import create_nnmodel
import torch
from params import params
from loader import generate_ict_128x128_disc_tri
from loader import get_testset
params['device']  = 'cpu'

###################################

#parameters that were in params
import numpy as np
import losses
import torch
name = 'tri'
################### Normalization functions ###################################
def scaleandlog(data, scale):
    data = np.nan_to_num(data)
    return np.log10(1 + data/scale)

def nonorm(data, scale):
    return data/scale

def norm_labels(labels):
    #['PlanetMass', 'AspectRatio', 'Alpha', 'InvStokes1', 'FlaringIndex']
    max = np.array([1e-2, 0.1, 0.01, 1e3, 0.35])
    min = np.array([1e-5, 0.03, 1e-4, 10, 0])
    for i in [0, 2, 3]:
        labels[:, i] = np.log10(labels[:,i])
        max[i] = np.log10(max[i])
        min[i] = np.log10(min[i])
    labels = 2*(labels-min)/(max-min) - 1
    return labels

def norm_labels_gas(labels):
    #['PlanetMass', 'AspectRatio', 'Alpha',  'FlaringIndex']
    max = np.array([1e-2, 0.1, 0.01,  0.35])
    min = np.array([1e-5, 0.03, 1e-4,0])
    for i in [0, 2]:
        labels[:, i] = np.log10(labels[:,i])
        max[i] = np.log10(max[i])
        min[i] = np.log10(min[i])
    labels = 2*(labels-min)/(max-min) - 1
    return labels

def norm_cube_log(data, scale):
    shape = [1,3,1,1] if len(data.shape)==4 else [3,1,1]
    scale = np.array([scale, 1, 1]).reshape(shape)
    data = np.nan_to_num(data)
    return data/scale

def generate_ict_128x128_disc(slopes, nonorm=False):
    #generating initial conditions
    x = np.linspace(-3, 3, 128)
    y = np.linspace(-3, 3, 128)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    ict = np.float32(r**(-slopes.reshape(-1,1,1))*((r<3) & (r>0.4)))
    
    if not nonorm:
        ict = params['norm'](np.float32(ict), 1)
    ict = np.expand_dims(ict, axis=1)
    return ict

def generate_ict_128x128_disc_tri(slopes):
    x = np.linspace(-3, 3, 128)
    y = np.linspace(-3, 3, 128)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    vaz_ict = np.float32(r**(-0.5)*((r<3) & (r>0.4)))
    vaz_ict = np.expand_dims(np.repeat(np.expand_dims(vaz_ict, 0), len(slopes),axis=0), 1)
    vr_ict = np.zeros(vaz_ict.shape)
    dens_ict = generate_ict_128x128_disc(slopes, nonorm=True)
    ict = np.concatenate([dens_ict, vaz_ict, vr_ict], axis=1)
    return np.float32(params['norm'](ict, 1))

######################################################################
params = {
    'name': name,  
    'device': 'cuda', 
    'nepochs': 801,
    'lr': 1e-4,
    'save_model': True,
    'savedir': f'../outputs/{name}',
    'datadir': f'../data/gas_tri/',
    'mode': '128x128_disc_tri',
    'Override': True,
    'savefreq': 20,
    'cond': True,
    'lr_decay': False,
    'resume': False,
    'periodic_bound_x': False,
    'sample_freq': 10, 
    'batch_size': 32,
    'rotaugm': False,
    'image_size': 128,
    'logima_freq': 20,
    'loss': torch.nn.MSELoss(),
    'unc': False,
    'norm': norm_cube_log,
    'scale': 1e-3,
    'norm_labels': norm_labels_gas,
    'n_test_log_images': 50,
    'num_channels': 96,
    'channel_mult': "1, 1, 2, 3, 4",
    'num_res_blocks': 3,
    'pretrain': False,
    'n_param' : 4,
    'infer_labels': ['PlanetMass', 'AspectRatio', 'Alpha', 'FlaringIndex'],
    'n_pretrain': 10000 #note: it must be <101,000
}


###################################


class Emulator():
    
    def __init__(self, device='cpu'):
        self.emulator = create_nnmodel(n_param=params['n_param'], image_size=params['image_size'], num_channels=params['num_channels'],
                                        num_res_blocks=params['num_res_blocks'], channel_mult=params['channel_mult'],
                                        mode=params['mode'], unc=params['unc']).to(device=torch.device(device))
        dataem = torch.load(f'{params["savedir"]}/model__epoch_{ep}_test_{params["name"]}.pth',  map_location=torch.device('cpu'))
        self.emulator.load_state_dict(dataem)

    #emulate with norm params
    def __emulate(self,ic, params):
        #for i, test_im in tqdm(enumerate(xtest), desc='element of testset'):
        emulation = self.emulator(ic, params)
        return emulation
    
    def emulate(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        ic = generate_ict_128x128_disc_tri(slopes=np.array(sigmaSlope))
        params = np.array([planetMass, h, alpha, flaringIndex]).reshape(1,4)
        norm_params = params['norm_labels'](params)
        return self.__emulate(ic, norm_params)
    
    def emulate_dens(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex)[:,0]
    
    def emulate_vphi(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex)[:,1]
    
    def emulate_vr(self, alpha, h, planetMass, sigmaSlope, flaringIndex):
        return self.emulate(alpha, h, planetMass, sigmaSlope, flaringIndex)[:,2]
        
    