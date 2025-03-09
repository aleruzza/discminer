from pathlib import Path
from random import randint
import torch as th
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import torch
from params import params
    
image_size = params['image_size']

############ functions that read numpy array data ##############################

def get_image_files_narray(base_path):
    image_files = np.load(f'{base_path}')
    return image_files


def get_labels_narray(base_path, labels):
    dataframe = pd.read_csv(f'{base_path}', index_col=0)
    labels = np.array(dataframe[labels])
    #initial conditions
    slopes = np.array(dataframe['SigmaSlope'])
    return np.float32(params['norm_labels'](labels)), slopes


def get_pretraining_data(base_path, n=10):
    dataset = np.load(f'{base_path}/swe_data.npy')
    dataset = dataset.reshape(-1, image_size, image_size)
    np.random.shuffle(dataset)
    return dataset[0:n]


def generate_ict(slopes, mode):
    if mode=='mdeco':
        return generate_ict_mdeco(slopes=slopes)
    elif mode=='128x128_disc':
        return generate_ict_128x128_disc(slopes=slopes)
    elif mode=='cyl':
        return generate_ict_cyl(slopes=slopes)
    elif mode=='128x128_disc_tri':
        return generate_ict_128x128_disc_tri(slopes=slopes)
        
    
    
########### Initial conditions function generators ##################

def generate_ict_mdeco(slopes):
    r = np.logspace(np.log10(0.3), np.log10(3), image_size)
    t = np.linspace(0, 2*np.pi, 512)
    rr, _ = np.meshgrid(r, t)
    ict = np.float32(params['scale']*rr**(-slopes.reshape(-1,1,1)))
    ft = np.fft.rfft(ict, axis=1)
    #remove the last k to make the input data divisible by 2 multiple times -> (1000x256x128)
    ft = ft[:,:-1,:]
    #put imag and real parts in different channels
    real = np.expand_dims(ft.real, axis=1)
    imag = np.expand_dims(ft.imag, axis=1)
    ict = np.concatenate([real, imag], axis=1)
    ict = params['norm'](np.float32(ict),1e-5)
    return ict

def generate_ict_128x128_disc(slopes, nonorm=False):
    #generating initial conditions
    x = np.linspace(-3, 3, image_size)
    y = np.linspace(-3, 3, image_size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    ict = np.float32(r**(-slopes.reshape(-1,1,1))*((r<3) & (r>0.4)))
    
    if not nonorm:
        ict = params['norm'](np.float32(ict), 1)
    ict = np.expand_dims(ict, axis=1)
    return ict

def generate_ict_128x128_disc_tri(slopes):
    x = np.linspace(-3, 3, image_size)
    y = np.linspace(-3, 3, image_size)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2+yy**2)
    vaz_ict = np.float32(r**(-0.5)*((r<3) & (r>0.4)))
    vaz_ict = np.expand_dims(np.repeat(np.expand_dims(vaz_ict, 0), len(slopes),axis=0), 1)
    vr_ict = np.zeros(vaz_ict.shape)
    dens_ict = generate_ict_128x128_disc(slopes, nonorm=True)
    ict = np.concatenate([dens_ict, vaz_ict, vr_ict], axis=1)
    return np.float32(params['norm'](ict, 1))

def generate_ict_cyl(slopes, nr=image_size, ntheta=512):
    r = np.logspace(np.log10(0.3), np.log10(3), nr)
    t = np.linspace(0, 2*np.pi, ntheta)
    rr, _ = np.meshgrid(r, t)
    ict = np.float32(1e-5*rr**(-slopes.reshape(-1,1,1)))
    return np.expand_dims(ict, axis=1)
        
        

#################################################################################

#######################Test set##################################################

def get_testset(params):    
    lab, slopes = get_labels_narray(f'{params["datadir"]}/testpara.csv', labels=params['infer_labels'])
    ict = generate_ict(slopes, mode=params['mode'])
    testparam = torch.tensor(lab).to(params['device'])
    xtest = params['norm'](np.load(f'{params["datadir"]}/datatest.npy'),params['scale'])
    
    if not params['mode']=='mdeco':
        xtest = np.expand_dims(xtest, axis=1)
    xtest = torch.tensor(xtest).to(params['device'])

    ict = torch.tensor(ict).to(device=params['device'])
    
    return ict, testparam, xtest


###################### Datasets #################################################

class PretrainDataset(Dataset):
 
    def __init__(
        self,
        folder="",
        image_size=128,
        shuffle=False,
        n_param =5,
        n_pretrain=100
    ):
        """Init

        Parameters
        ----------
        folder : str, optional
            folder where the .npy files are stored, by default ""
        image_size : int, optional
            pixel dimension of the images in the dataset, by default 128
        shuffle : bool, optional
            enables shuffling of the dataset, by default False
        n_param : int, optional
            number of conditional parameters for which the model is built.
            Note that during pretraining all these parameters
            will be set to 0, by default 6
        """        
        super().__init__()
        folder = Path(folder)
        self.data = get_pretraining_data(folder, n=n_pretrain)
        self.shuffle = shuffle
        self.prefix = folder
        self.image_size = image_size
        self.n_param = n_param

    def __len__(self):
        return len(self.data)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        tokens = np.float32(np.array([0]).repeat(self.n_param)) #array([0,0]) for uncondional training
        original_image = np.float32(self.data[ind])
        arr = np.expand_dims(original_image,axis=0) # only one channel
        return th.tensor(arr),th.tensor(np.float32(tokens))

    

class TextImageDataset(Dataset):
    def __init__(
        self,
        folder="",
        labels_file = "run4.csv",
        data_file="data.npy",
        image_size=64,
        shuffle=False,
        rotaugm=False, 
        mode='cyl',
        device='cpu',
        inf_channel=None
    ):
        super().__init__()
        folder = Path(folder)
        self.data = get_image_files_narray(f'{folder}/{data_file}')
        if inf_channel is not None:
            self.data = self.data[:,[inf_channel],:,:]
        self.labels, self.slopes = get_labels_narray(f'{folder}/{labels_file}', labels=params['infer_labels'])
        self.ics = generate_ict(self.slopes, mode=mode )
        self.rotaugm = rotaugm
        self.shuffle = shuffle
        self.prefix = folder
        self.mode = mode
        self.image_size = image_size
        
        if rotaugm:
            self.transform = T.RandomRotation(90)

    def __len__(self):
        return len(self.labels)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)


    def __getitem__(self, ind):
        original_image = np.float32(self.data[ind])
        if self.mode!='mdeco':
            arr = params['norm'](original_image, params['scale'])
        else:
            arr = params['norm'](original_image, params['scale'])
        arr = th.tensor(np.float32(arr))
        if self.rotaugm:
            arr = self.transform(arr)
            
        return arr, th.tensor(np.float32(self.labels[ind])), th.tensor(self.ics[ind])
    
