#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import torch.nn as nn
import h5py
import numpy.lib.recfunctions as rf
from numpy import linalg as LA
import math
import  os

def optimize_generator(dop,G,z,params,train_epoch=1):
  with torch.autograd.set_detect_anomaly(True):    
    pathname =  params['filename'].replace('.mat','_'+str(params['gen_base_size'])+'d/weights_GENplusLAT'+str(params['coilEst'])+'_')
    pathname =      pathname+str(params['slice'])+'_'+str(params['nintlPerFrame'])+'arms_'+str(params['siz_l'])+'latVec'+str(params['nFramesDesired'])+'frms'
    if not(os.path.exists(pathname)):
      os.makedirs(pathname)
    
    lr_g = params['lr_g']
    lr_z = params['lr_z']
    gpu = params['device']
    batch_sz = params['nBatch']
    legendstring = np.array2string(np.arange(params["siz_l"]))
    legendstring = legendstring[1:-1]
    #optimizer = optim.SGD([
    #{'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_g},
    #{'params': z.z_, 'lr': lr_z}
    #], momentum=(0.9))
    optimizer = optim.Adam([
    {'params': filter(lambda p: p.requires_grad, G.parameters()), 'lr': lr_g},
    {'params': z.z_, 'lr': lr_z}
    ], betas=(0.4, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=30, verbose=True, min_lr=1e-6)
    
    if isinstance(params['slice'], int):
        nslc = 1
    else:
        nslc = len(params['slice'])
   
    train_hist = {}
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    loss = nn.MSELoss(reduction='sum')
    
    G_oldi = G.state_dict()
    z_oldi = z.z_.data
    divergence_counter = 0 
    
    print('training start!')
    start_time = time.time()
    G_losses = []
    SER = np.zeros(train_epoch)
    nbatches = params['nFramesDesired']//batch_sz  #180
    nindices = nbatches*batch_sz  #180*5=900
    # Epoch loop
    for epoch in range(train_epoch):
        indices=np.arange(0,params['nFramesDesired'])
        random.shuffle(indices)
        #indices = np.random.randint(0,params['nFramesDesired'],params['nFramesDesired'])
        indices = np.reshape(indices[0:nindices],(nbatches,batch_sz))
        epoch_start_time = time.time()
        batch_loss = 0
        # Batch loop
        #-----------
        
        for batch in range(nbatches):
            G_loss = 0
            optimizer.zero_grad()
            
            #Slice loop
            #----------
            
            for slc in range(nslc):
                G_result = G(z.z_[indices[batch],...,slc])[...,slc]
                G_result_projected = dop.Psub(G_result,indices[batch],slc)
                
                if(params["fastMode"]):
                    G_loss += loss(torch.view_as_real(G_result_projected),torch.view_as_real(dop.Atb[indices[batch],...,slc]))
                else:
                    G_loss += loss(torch.view_as_real(G_result_projected),torch.view_as_real(dop.Atb[indices[batch],...,slc]).to(gpu))
                    
                G_loss += dop.image_energy_sub(G_result,slc)  # image regularization to zero out regions outside maks
                #print('z.KLloss=%.3f, %.1f slice'%(z.KLloss(slc),slc))
                G_loss += z.KLloss(slc)  # K_L divergence loss per slice 

            #Slice loop end
            #---------------
                
            G_loss +=  G.weightl1norm()    # Netowrk regularization
            G_loss += z.Reg()     # latent variable regularization
            
            
            G_loss.backward()
            #print('before batchloss=%.3f'%batch_loss)
            batch_loss += G_loss.detach()
            optimizer.step()
            #print('After batchloss=%.3f'%batch_loss)
        # Batch loop end
        #---------------
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        G_losses.append(batch_loss.item())
        #print(G_losses)
        # If cost increases, load an old state and decrease step size
        if(epoch >10):
            if((batch_loss.item() > 1.15*train_hist['G_losses'][-1])): # higher cost
                G.load_state_dict(G_oldi)
                z.z_.data = z_oldi
                print('loading old state; reducing stp siz')
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.98
                divergence_counter = divergence_counter+1
                
            else:       # lower cost; converging
                divergence_counter = divergence_counter-1
                if((divergence_counter<0)):
                    divergence_counter=0
                G_oldi = G.state_dict()
                z_oldi = z.z_.data    
                train_hist['G_losses'].append(batch_loss.item())
                path = os.path.join(pathname, 'net_{}_Zreg_epch{}_gloss{}.pth'.format('GENplusLAT', epoch,torch.mean(torch.FloatTensor(G_losses))))
                torch.save({'G_oldi':G.state_dict(),'z_oldi':z.z_}, path)
        else:
            G_oldi = G.state_dict()
            z_oldi = z.z_.data 
            train_hist['G_losses'].append(batch_loss.item())
            path = os.path.join(pathname, 'net_{}_Zreg_epch{}_gloss{}.pth'.format('GENplusLAT', epoch,torch.mean(torch.FloatTensor(G_losses))))
            torch.save({'G_oldi':G.state_dict(),'z_oldi':z.z_}, path) 
            
        # If diverges, exit
        if(divergence_counter>=1):
            print('Optimization diverging; exiting')
            return G,z,train_hist,SER,epoch
        
        #G_oldi = G.state_dict()
        #z_oldi = z.z_.data
        
        #path = os.path.join(pathname, 'net_{}_Zreg_epch{}_gloss{}.pth'.format('GENplusLAT', epoch,torch.mean(torch.FloatTensor(G_losses))))
        #torch.save({'G_oldi':G.state_dict(),'z_oldi':z.z_}, path)
        # Epoch loop end
        #---------------
        
        #Display results
        print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(G_losses))))

        if(np.mod(epoch,30)==0):
          fig,ax = plt.subplots(nslc,2)   
          if(nslc==1):
            ax = np.expand_dims(ax,0)
          for sl in range(nslc):
              G_result = G(z.z_[indices[batch],...,sl])[...,sl]

              test_image1 = G_result[-1].squeeze(0).cpu().data.numpy()
              
              ax[sl,0].imshow(abs(test_image1),cmap='gray')
              temp = z.z_[...,sl].data.squeeze().cpu().numpy()
              ax[sl,1].plot(temp)
              #ax[sl,1].legend(legendstring,loc='best')          
          plt.pause(0.00001)
          print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(G_losses))))
    print('Optimization done in %d seconds' %(time.time()-start_time))
    print('[%d/%d] - ptime: %.2f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(G_losses))))
    return G,z,train_hist,SER,epoch
