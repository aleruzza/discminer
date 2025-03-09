import os
import wandb
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from params import params
from loader import TextImageDataset, PretrainDataset, get_testset
from create_model import create_nnmodel


def image_from_mdeco(mdeco):
    '''
        Go from the m-modes decomposition of a disc image back to the disc
        I/O shapes: (N, 2, X/2, Y) -> (N, 1, X, Y) 
    '''
    #mdeco = mdeco*1e-5
    fft = mdeco[:,0,:,:]+1j*mdeco[:,1,:,:]
    fft = np.pad(fft, pad_width=((0,0),(0,1),(0,0)))
    images = np.fft.irfft(fft, axis=1)
    return images
    
    
def train(params, model):
    # initialize the dataset
    #if pretrain load the pretraining dataset
    if params['pretrain']:
        dataset = PretrainDataset(
            folder=params['datadir'],
            image_size=128,
            shuffle=True,
            n_param=params['n_param'],
            n_pretrain = params['n_pretrain']
        )
        test_param = None
    else:
        #targets
        dataset = TextImageDataset(
                folder=params['datadir'],
                image_size=params['image_size'],
                shuffle=True,
                rotaugm=params['rotaugm'],
                mode=params['mode'],
                device=params['device']
            )

    # dataloader setup
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    #get test set
    ict, testparam, xtest = get_testset(params)
    
    # number of UNET parameters to be trained
    params_to_optimize = [
            {'params': model.parameters()}
        ]
    number_of_params = sum(x.numel() for x in model.parameters())
    print(f"Number of parameters for unet: {number_of_params}")
    
    #define the loss function
    loss = params['loss']
    
    #initialize optimizer
    optim = torch.optim.Adam(params_to_optimize, lr=params['lr'])

    #loop
    wandb.watch(model, criterion=loss, log_freq=10)
    for ep in range(params['nepochs']):
        print(f'epoch {ep}')
        model.train() #setting model in training mode
        
        #linear lr decay
        if params['lr_decay']:
            optim.param_groups[0]['lr'] = params['lr']*(1-ep/params['nepochs'])
            
        pbar = tqdm(dataloader)
        mean_loss = np.array([])
        for i, (x, p, ic) in enumerate(pbar):
            model.train()
            model.zero_grad()
            optim.zero_grad() #reset the gradients
            x = x.to(params['device'])
            p = p.to(params['device'])
            ic = ic.to(params['device'])
            x_pred = model(ic, p)
            
            lossv = loss(x_pred, x)
            lossv.backward()
            mean_loss = np.append(mean_loss, [lossv.item()])
            #mean_mse = np.append(mean_mse, [getmse(x, x_pred)])
            pbar.set_description(f'loss: {lossv.item():.4f}')
            #wandb.log({'mse_train': mean_mse, 'epoch': ep})
            optim.step()
            
        if params['save_model']:
            if ep%params['savefreq']==0:
                torch.save(model.state_dict(), params['savedir'] + f"/model__epoch_{ep}_test_{params['name']}.pth")
        
        model.eval()
        with torch.inference_mode():
            x_pred_t = model(ict, testparam)
            mse_test = getmse(x_pred_t.cpu(), xtest.cpu())
            wandb.log({'loss': mean_loss.mean(), 'epoch': ep, 'mse_test': mse_test})
            
            if params['mode']=='mdeco':
                im_pred = image_from_mdeco(x_pred_t.cpu())
                im_test = image_from_mdeco(xtest.cpu())
            else:
                im_pred = x_pred_t.cpu()
                im_test = xtest.cpu()
                
            mse_test_image = getmse(im_pred, im_test)
            l1_test_image = getl1(im_pred, im_test)
            wandb.log({'mse_test_image':mse_test_image, 'l1_test_image':l1_test_image, 'epoch':ep})
                
            #log some test images
            if ep%params['logima_freq']==0:
                images = []
                for i in range(params['n_test_log_images']):
                    image = wandb.Image(torch.tensor(np.float32(im_pred[i])), mode='F')
                    images.append(image)
                wandb.log({"testset_emulations": images})
            if ep==0:
                images = []
                for i in range(params['n_test_log_images']):
                    image = wandb.Image(torch.tensor(np.float32(im_test[i])), mode='F')
                    images.append(image)
                wandb.log({"testset_simulations": images})
            
            del x_pred_t
            torch.cuda.empty_cache()
        
                
        
        
def getmse(im1, im2):
    #xx, yy = np.meshgrid(x,y)
    #rr = np.sqrt(xx**2+yy**2)
    #return (((im1-im2)**2)*((rr<3) & (rr>0.3))).mean()
    return ((im1-im2)**2).mean().to('cpu')

def getl1(im1, im2):
    return torch.abs(im1-im2).mean().to('cpu')
    

if __name__ == "__main__":
    
    #checking if exists and creating output directory if it does not
    if os.path.exists(params['savedir']):
        if params['Override']:
            print('Saving directory exists, overriding old data as instructed.')
        else:
            print('WARNING! -> saving directory already exists, please run with Override=True')
            exit()
    else:
        os.mkdir(params['savedir'])

    #checking file with parameter history and adding this run
    if os.path.exists('parahist.csv'):
        oldpara = pd.read_csv('parahist.csv', index_col=0)
        params['index'] = oldpara.index[-1]+1
        newparafile = pd.concat([oldpara, pd.DataFrame([params]).set_index('index')])
    else:
        params['index'] = 0
        newparafile = pd.DataFrame([params]).set_index('index')
    newparafile.to_csv('parahist.csv')
    
    with wandb.init(project='emulator_unet', config=params, name=params['name']):
        #begin train
        if params['resume']:
            if not os.path.exists(params['resume_from']):
                print('Error! the model wich you want to resume from does not exist!\n Exiting...')
                exit()
            else:
                #TODO: implement possibility to resume
                exit()
        else:
            emulator = create_nnmodel(n_param=params['n_param'], image_size=params['image_size'], num_channels=params['num_channels'],
                                      num_res_blocks=params['num_res_blocks'], channel_mult=params['channel_mult'],
                                      mode=params['mode'], unc=params['unc']).to(device=params['device'])
            

        train(params=params, model=emulator)
