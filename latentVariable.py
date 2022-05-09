#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
from torch.autograd import Variable

class latentVariableNew():
    def __init__(self,params,z_in=1,init='interpolate',alpha=[0,0],klreg=0):
        
        self.gpu=torch.device(params['device'])
        self.klreg = klreg
        self.alpha = alpha
        
        if isinstance(params['slice'], int):
            self.nsl = 1
        else:
            self.nsl = len(params['slice'])
            
        device = params['device']
        nFrames = params['nFramesDesired']
        siz_l = params['siz_l']
        if(init=='ones'):
            zpnew = 0.5*np.ones((nFrames,siz_l,1,1,self.nsl))
            for i in range(siz_l):
                zpnew[:,i]*= np.random.rand()

        elif(init=='random'):
            zpnew = 0.15*np.random.randn(nFrames,siz_l,1,1,self.nsl)
        elif(init=='zeros'):
            zpnew = np.zeros((nFrames,siz_l,1,1,self.nsl))
        else:
            zinput = z_in.z_.data.cpu().numpy()
            nin = np.size(zinput,0)
            if(nFrames == nin):
                zpnew = zinput
            else:
                x = np.arange(0,nFrames)
                nf = np.size(zinput,0)          
                xp = np.arange(0,nf)*nFrames/nf
                zpnew = np.zeros((nFrames,np.size(zinput,1),1,1,self.nsl))
                for i in range(np.size(zinput,1)):
                    zpnew[:,i,:] = np.interp(x, xp, zinput[:,i,:])                   
        
        z_in = torch.tensor(zpnew,dtype=torch.float32)
        z_in = z_in.cuda(device)
        self.z_ = torch.zeros((nFrames, siz_l,1,1,self.nsl),dtype=torch.float32)
        self.z_ = Variable(self.z_.cuda(device), requires_grad=True)
        self.z_.data = z_in
        
        
        
    def Reg(self):
        zsmoothness = self.z_[1:,:,:,:]-self.z_[:-1,:,:,:]
        zsmoothness = torch.sum(zsmoothness*zsmoothness,axis=(0,-1)).squeeze()
        zsmoothness = torch.sum(self.alpha*zsmoothness,axis=0)
        #kl=self.KL_loss()
        return(zsmoothness)
   
    def KLloss(self,slc):
        Nsamples = self.z_.shape[0]
        mn = torch.mean(self.z_[:,:,0,0,slc],0)
        meansub = self.z_[:,:,0,0,slc] - mn
        Sigma = meansub.T@meansub/Nsamples

        tr = torch.trace(Sigma)
        if(tr> 0.001):
          loss = 0.5*(mn@mn.T + tr - self.z_.shape[1] - torch.logdet(Sigma))
          if(math.isnan(loss)):
            loss = 0.5*(mn@mn.T)
        else:
          loss = 0.5*(mn@mn.T)  
        return(self.klreg*loss)
    
    #def KLloss(self):
      #Nsamples = self.z_.shape[0]
      #mn = torch.mean(self.z_[:,:,0,0],0)
     # meansub = self.z_[:,:,0,0] - mn
     # Sigma = meansub.T@meansub/Nsamples

      #tr = torch.trace(Sigma)
     # if(tr> 0.001):
      #  loss = 0.5*(mn@mn.T + tr - self.z_.shape[1] - torch.logdet(Sigma))
     # else:
        #loss = 0.5*(mn@mn.T)  
      #return(self.klreg*loss)
 
  #  def KL_loss(self):
      
  #      Nsamples = self.z_.shape[0]
  #      mn = torch.mean(self.z_[:,:,0,0],0)

  #      meansub=torch.zeros((self.z_.shape[0],self.z_.shape[1],self.z_.shape[-1]),dtype=torch.float32).to(self.gpu)
   #     Sigma=torch.zeros((self.z_.shape[1],self.z_.shape[1],self.z_.shape[-1]),dtype=torch.float32).to(self.gpu)
   #     tr=torch.zeros((1,self.z_.shape[-1]),dtype=torch.float32).to(self.gpu)
   #     lossk=torch.zeros((self.z_.shape[-1]),dtype=torch.float32).to(self.gpu)

   #     for i in np.arange(self.z_.shape[-1]):
    #        meansub[...,i] = self.z_[:,:,0,0,i] - mn[:,i]
    #        A=meansub[...,i].T.contiguous()
    #        Sigma[...,i] = A@meansub[...,i]/Nsamples
     #       tr[...,i] = torch.trace(Sigma[...,i])
    #        B=mn[...,i].T.contiguous()
     #       if(tr[...,i]> 0.001):
     #           lossk[i] = 0.5*(mn[...,i]@B + tr[...,i] - self.z_.shape[1] - torch.logdet(Sigma[...,i]))
    #        else:
    #            lossk[i] = 0.5*(mn[...,i]@B)  
        #del meansub, Sigma, tr, B
    #    lossk = torch.sum(lossk,axis=0)
    #    return(self.klreg*lossk)
   
