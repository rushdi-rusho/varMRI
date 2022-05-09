#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import path
import numpy as np
import torch
import torchkbnufft as tkbn

from espirit.espirit import espirit, fft
import pickle
import mat73
import tqdm
import matplotlib.pyplot as plt
import scipy as scipy
import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl

#%% Reads the data and operators

class dataAndOperators:

  # function to change the number of frames
    def changeNumFrames(self,params):
        gpu = self.gpu
        self.smapT = torch.tensor(self.smap,dtype=torch.complex64)
        self.smapT = torch.tile(self.smapT,[params["nBatch"],1,1,1,1])

        ktrajSorted = torch.reshape(self.ktraj,(1,2,params["nFramesDesired"], params["nintlPerFrame"]*self.nkpts))  
        ktrajSorted = ktrajSorted.permute(2,1,3,0).squeeze().contiguous()

        kdataSorted = torch.reshape(self.kdata,(1,self.nch,params["nFramesDesired"],  params["nintlPerFrame"]*self.nkpts,self.nsl))
        kdataSorted = kdataSorted.permute(2,1,3,0,4).squeeze(3).contiguous()

        sz = (params["nFramesDesired"],1,self.im_size[0],self.im_size[1],self.nsl)
    
        if(self.params["verbose"]):
            print("Precomputing Atb ..")
    
       
        if(self.params["fastMode"]):
            self.Atb = torch.zeros(sz,dtype=torch.complex64)
            for i in range(self.nsl):
                kdata1 = kdataSorted[:,:,:,i].to(gpu)
                ktraj1 = ktrajSorted.to(gpu)
                temp3 = torch.tensor(self.smap[...,i],dtype=torch.complex64)
                smap1 = torch.tile(temp3,[params["nFramesDesired"],1,1,1]).to(gpu)
                self.Atb[...,i] = self.adjnufft_ob(kdata1,ktraj1,smaps=smap1) # adjoin_kdata1 shape: (batch,coil, klength)
            del kdata1, ktraj1, temp3, smap1  
        else:
            self.Atb = torch.zeros(sz,dtype=torch.complex64)
            for i in range(self.nsl):
                for j in range(params["nFramesDesired"]):
                    temp1 = kdataSorted[j:j+1,:,:,i].to(gpu) 
                    temp2 = ktrajSorted[j:j+1].to(gpu)
                    temp3 = self.smapT[0:1,:,:,:,i].to(gpu)
                    self.Atb[j:j+1,0,:,:,i]=self.adjnufft_ob(temp1,temp2,smaps=temp3).cpu() 
            del temp1, temp2, temp3   
    

        if(self.params["verbose"]):
            print("Precomputing Toeplitz kernels ..")
    
    
        dcfSorted = torch.reshape(self.dcf,(1,params["nFramesDesired"],params["nintlPerFrame"]*self.nkpts))
        dcfSorted = dcfSorted.permute(1,0,2).contiguous()
        sz = (params["nFramesDesired"],2*self.im_size[0],2*self.im_size[1])
        self.dcomp_kernel = torch.zeros(sz,dtype=torch.complex64)
    
        for i in range(params["nFramesDesired"]):
            self.dcomp_kernel[i] = tkbn.calc_toeplitz_kernel(ktrajSorted[i].to(gpu),tuple(self.im_size),dcfSorted[i].unsqueeze(0).to(gpu)).cpu() 
    
        self.dcomp_kernel = self.dcomp_kernel.unsqueeze(1)

        maxvalue = torch.view_as_real(self.Atb).max()
        self.Atb = self.Atb/maxvalue/2
    
        temp1 = self.Atb[0:1,...,0].to(gpu)
        temp2 = self.dcomp_kernel[0:1].to(gpu)
        temp3 = self.smapT[0:1,:,:,:,0].to(gpu)
            
        temp = self.toep_ob(temp1,temp2,smaps=temp3.to(gpu)).cpu()
        maxvalue = torch.view_as_real(temp).max()
        self.dcomp_kernel = self.dcomp_kernel/maxvalue
        del temp1, temp2, temp3
    
        self.mask = self.Atb[0:self.batch_size].abs() == 0.00
        if self.params["fastMode"]:
            print("Moving to GPU")
            self.Atb = self.Atb.to(gpu)
            self.dcomp_kernel = self.dcomp_kernel.to(gpu)
            self.smapT = self.smapT.to(gpu)
    
        torch.cuda.empty_cache()
    
        if self.params["verbose"]:
            print("Done initializing data Object !!")
            max_memory = torch.cuda.max_memory_allocated(device=gpu)*1e-9
            print("Max GPU utilization",max_memory," GB ")
            current_memory = torch.cuda.memory_allocated(device=gpu)*1e-9
            print("Current GPU utilization",current_memory," GB ")
            available_memory = torch.cuda.get_device_properties(device=gpu).total_memory*1e-9
            print("Total GPU memory available",available_memory," GB ")
            if(current_memory > available_memory*0.5):
                print("You may have to switch off fastMode to conserve GPU memory")
        return(self)
    
#-----------------------------------------------------------------------------

  # Initialization  
  #---------------------------
    def __init__(self,params):

        self.params = params
        self.im_size = params["im_size"]
        self.batch_size = params["nBatch"]
        dtype = params['dtype']
        gpu=torch.device(params['device'])
        self.gpu = gpu

        if(self.params["verbose"]):
            print("Reading data ..")
    
        # Reading h data from mat file
        #----------------------------------------------
        if(params['filename'][-3:-1] =='ma'):  # mat file
            extension = '_'+str(params['slice'])+'.pickle'
            fnamepickle = params['filename'].replace('.mat',extension)    
            if(not(path.exists(fnamepickle))):
                data_dict = mat73.loadmat(params['filename'])
                kdata = data_dict['kdata']
                kdata = np.squeeze(kdata[:,:,:,params['slice']])
                ktraj=np.asarray(data_dict['k'])    
                dcf=np.asarray(data_dict['dcf'])

            # save with pickle for fast reading
                with open(fnamepickle, 'wb') as f:
                    pickle.dump([kdata,ktraj,dcf],f,protocol=4)
            else:
                with open(fnamepickle, 'rb') as f:
                    [kdata,ktraj,dcf] = pickle.load(f)
        
        else: # read pickle file
            fname = params['filename']  
            with open(fname, 'rb') as f:
                [kdata,ktraj,dcf] = pickle.load(f)
    
        #Reshaping the variables
        #----------------------------------------------
        kdata = kdata.astype(np.complex64)
        ktraj=ktraj.astype(np.complex64)

        if isinstance(params['slice'], int):
            kdata = np.expand_dims(kdata,3)
    
        #kdata=np.transpose(kdata,(1,2,0)) 
        kdata=np.transpose(kdata,(1,2,0,3)) 
        dcf = np.transpose(dcf,(1,0))
        ktraj = np.transpose(ktraj,(1,0))

        # Reducing the image size if factor < 1
        #----------------------------------------------

        im_size = np.int_(np.divide(params["im_size"],params["factor"]))
        ktraj=np.squeeze(ktraj)*2*np.pi

        # Deleting initial interleaves to achieve steady state
        #------------------------------------------------------
        self.nch = np.size(kdata,0)
        self.nkpts = np.size(kdata,2)

        nintlvsNeeded = params["nintlPerFrame"]*params["nFramesDesired"]
        nintlvs = np.size(kdata,1)
        nintlvsLeft = nintlvs - params["nintlvsToDelete"]
        if(nintlvsNeeded > nintlvsLeft):
            print("Too few interleaves. Reduce nFramesDesired or nintlvsToDelete")
    
        self.nsl = kdata.shape[3]



        # Reconstructing coil images 
        #---------------------------
    
        if(self.params["verbose"]):
            print("Reconstruction of coil images ..")
        
        self.nch = kdata.shape[0]
        

        kdata = np.reshape(kdata[:,params["nintlvsToDelete"]:nintlvsNeeded+params["nintlvsToDelete"]],(self.nch,nintlvsNeeded*self.nkpts,self.nsl)) 

        ktraj=ktraj[params["nintlvsToDelete"]:nintlvsNeeded+params["nintlvsToDelete"],:]
        dcf = dcf[params["nintlvsToDelete"]:nintlvsNeeded+params["nintlvsToDelete"],:]
        dcf = dcf/params["nintlPerFrame"]/params["nintlPerFrame"]

        ktraj = np.reshape(ktraj,(1,nintlvsNeeded*self.nkpts))
        ktraj = np.stack((np.real(ktraj), np.imag(ktraj)),axis=1)
        dcf = np.reshape(dcf,(1,nintlvsNeeded*self.nkpts)) 
        
        for i in range(self.nsl):
            for j in range(self.nch):
                kdata[j,:,i] = kdata[j,:,i] * dcf

        self.kdata = torch.tensor(kdata,dtype=torch.complex64).unsqueeze(0)
        
        self.ktraj = torch.tensor(ktraj,dtype=torch.float)
        self.dcf = torch.tensor(dcf,dtype=torch.float)

        self.adjnufft_ob=tkbn.KbNufftAdjoint(im_size=im_size,grid_size=im_size,device=gpu)
        coilimages = np.zeros((self.nch,im_size[0],im_size[1],self.nsl)).astype(complex)
    
        for i in range(self.nsl):
            coilimages[...,i] = self.adjnufft_ob(self.kdata[...,i].to(gpu),self.ktraj.to(gpu)).squeeze(0).cpu()
        
        self.kdata = self.kdata.squeeze(3)
        #np.save('Coilimagggg.npy', coilimages)
        # FOIVR coil combination
        #------------------------------------------------------
    
        if(self.params["verbose"]):
            print("Coil combination ..")
        
        nCoils = params["virtual_coils"]
        x = np.arange(im_size[0])-im_size[0]/2
        y = np.arange(im_size[1])-im_size[1]/2
        x,y = np.meshgrid(x, y)
        mask = x**2 + y**2 < params["mask_size"]*im_size[0]*im_size[1]/4

        signal = coilimages*mask[None,:,:,None]
        noise = coilimages*np.logical_not(mask[None,:,:,None])

        signal = np.reshape(signal,(self.nch,im_size[0]*im_size[1]*self.nsl))
        noise = np.reshape(noise,(self.nch,im_size[0]*im_size[1]*self.nsl))
    

        A = np.real(signal@np.transpose(np.conj(signal)))
        B = np.real(noise@np.transpose(np.conj(noise)))
        [D,W] = scipy.linalg.eig(A,B);
        ind=np.flipud(np.argsort(D))
        W=W[:,ind[0:nCoils]]

        coilimages = W.T@np.reshape(coilimages,(self.nch,im_size[0]*im_size[1]*self.nsl))
        coilimages = np.reshape(coilimages,(nCoils,im_size[0],im_size[1],self.nsl))
        coilimages = np.expand_dims(coilimages,0)

        kdata = W.T@np.reshape(kdata,(self.nch,params["nFramesDesired"]*params["nintlPerFrame"]*self.nkpts*self.nsl))
        self.kdata = torch.tensor(kdata,dtype=torch.complex64).unsqueeze(0)
        self.nch = nCoils

        
        self.coilimages = coilimages

    # Coil sensitivity estimation
    #-----------------------------               
            
        if(self.params["coilEst"]=='espirit'):
            if(self.params["verbose"]):
                print("Coil sensitivity estimation using Espirit ..")
            
            x_f = fft(coilimages, (2, 3))
            #print(x_f.shape)
            smap=np.zeros((im_size[0],im_size[1],self.nch,self.nsl)).astype(complex)
            
            for nslc in range(self.nsl):
                x_f1=x_f[:,:,:,:,nslc]
                x_f1 = np.transpose(x_f1, (2, 3, 0, 1))  # 6, 24
                smap1 = espirit(x_f1, 6, 24, 0.02, 0.95)#0.14, 0.8925)
                smap1 = smap1[:, :, 0, :, 0]
                smap[:,:,:,nslc]=smap1
            np.save('ESPIRIT_def.npy',smap)  
            mp_s= np.absolute(np.transpose(smap,(2,0,1,3)))
            pl.ImagePlot(np.squeeze(mp_s[:,:,:,0]),z=0,mode='m',colormap='jet',title='Log magnitude of ESPIRIT maps')
            print("Coil sensitivity estimation using Espirit Done!")
            del mp_s
            #return
            #print(smap.shape)
            smap = np.transpose(smap, (2, 0, 1,3))  # 6, 24
            self.smap=np.expand_dims(smap,axis=0)
            #print(self.smap.shape)
     
            del x_f1, coilimages

        else:
            if(self.params["verbose"]):
                print("Coil sensitivity estimation using Jsense ..")
            smap = giveJsenseCoilsensitivities(coilimages.squeeze(0),0.03)
            mp_s= np.absolute(smap)
            pl.ImagePlot(np.squeeze(mp_s[:,:,:,0]),z=0,mode='m',colormap='jet',title='Log magnitude of JSENSE maps')
            print("Coil sensitivity estimation using Jsense Done!")
            del mp_s
            #print(smap.shape) ##(8, 168, 168, 4)
            np.save('JSENSE_def.npy',smap) 
           
            self.smap=np.expand_dims(smap,axis=0)
        

        self.toep_ob = tkbn.ToepNufft().to(gpu)
    
        self.changeNumFrames(params)

# Define operators        
        
    # Projection operator
    def Psub(self,x,indices,slc):
        out = 0*x
        if(self.params["fastMode"]):
            out = self.toep_ob(x, self.dcomp_kernel[indices],smaps=self.smapT[...,slc])
        else:
            for i in range(self.nsl):
                out = self.toep_ob(x, self.dcomp_kernel[indices].to(self.gpu),smaps=self.smapT[...,slc].to(self.gpu))
        return out

    # Image energy
    #--------------
    def image_energy_sub(self,x,slc):
        return torch.norm(x*self.mask[...,slc].to(self.gpu),'fro')

    

def giveJsenseCoilsensitivities(coilimages,threshold=0.05):
    nsl = coilimages.shape[3]
    sos = np.sqrt(np.sum(np.abs(coilimages)**2,0))
    mask = sos > threshold*np.max(sos)
    mps = np.zeros(coilimages.shape).astype(complex)
    
    for i in range(nsl):
        maskslc = scipy.ndimage.morphology.binary_closing(mask[...,i],iterations=20)
        maskslc = np.expand_dims(maskslc,0)
        test = sp.fft(coilimages[...,i],axes=(1,2))
        mpslc = mr.app.JsenseRecon(test,mps_ker_width=12,ksp_calib_width=48,max_iter=10,lamda=0.0,show_pbar=False).run()
        mps[:,:,:,i] = mpslc*maskslc.astype(complex)

    return(mps)