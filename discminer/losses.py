import torch.nn as nn
import torch

class MSEandFFT(nn.Module):
    def __init__(self, wmse=1, wfft=1, c_kea=1, gamma_kea=0, device='cuda'):
        super(MSEandFFT, self).__init__()
        self.wmse = wmse
        self.wfft = wfft
        self.moving_w = 1
        self.gamma_kea = gamma_kea
        self.c_kea = c_kea
        self.device = device

    def forward(self, inputs, targets):
        mse = (inputs-targets)**2
        mse = mse.mean()
        in_fft = torch.abs(torch.fft.rfft(inputs, axis=2))
        tar_fft = torch.abs(torch.fft.rfft(targets, axis=2))
        fftaw = self.c_kea*torch.exp(-self.gamma_kea*torch.range(0, tar_fft.shape[2]-1)).to(self.device)
        fft = (((in_fft-tar_fft)**2).mean(axis=-1)*fftaw).mean()
        
        return mse*self.wmse + fft*self.wfft
    
class MSEUnc(nn.Module):
    def  forward(self, inputs, targets):
        mse = (inputs[:,0,:,:]-targets[:,0,:,:])**2/torch.exp(2*inputs[:,1,:,:]) + inputs[:,1,:,:]/2
        return mse.mean()
        
