## Accelerated pseudo-3D dynamic speech MRI at 3T 
Reference Paper:   
[1] Rusho, R.Z., Zou, Q., Alam, W., Erattakulangara, S., Jacob, M., Lingala, S.G. (2022). Accelerated Pseudo 3D Dynamic Speech MR Imaging at 3T Using Unsupervised Deep Variational Manifold Learning. In: Wang, L., Dou, Q., Fletcher, P.T., Speidel, S., Li, S. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2022. MICCAI 2022. Lecture Notes in Computer Science, vol 13436. Springer, Cham.

Paper link: [click here](https://doi.org/10.1007/978-3-031-16446-0_66)

Relevant Paper: Q. Zou, A. H. Ahmed, P. Nagpal, S. Priya, R. F. Schulte and M. Jacob, "Variational Manifold Learning From Incomplete Data: Application to Multislice Dynamic MRI," in IEEE Transactions on Medical Imaging, vol. 41, no. 12, pp. 3552-3561, Dec. 2022, doi: 10.1109/TMI.2022.3189905.

 
### The framework of proposed pseudo-3D dynamic speech MRI
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/Proposed%20pseudo-3D%20speech%20MRI.png" width=80% height=80%>
Fig. 1: The proposed pseudo-3D variational manifold speech MRI reconstrruct scheme.     
  
  
We propose to recover time aligned multi-slice (or pseudo-3D) dynamic image time series from sequentially acquired sparsly sampled spiral k-t space data of multiple 2D slices. This is done by jointly learning low dimensional latent vectors and the CNN-based generator parameters from the undersampled scanner measurements. The network optimizes the following loss function,  
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/Loss%20functions.png" width=60% height=60%>

### Main advantages of our method
1. In contrast to existing models, this method uses implicit motion resolved reconstruction strategy by exploiting the smoothness of the image time series on a manifold. Distant images that share same speech posture are mapped as neighbors on the manifold.  
2. In contrast to current 2D dynamic speech MRI approaches that reconstructs slices independently, resulting in full vocal tract motion to be out of synchrony across slices, we propose to jointly recover all the slices as a time aligned multi-slice 2D (or pseudo-3D) dataset.  
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/Video-2-2D-CS.gif" width=70% height=70%>  
Fig. 2: 2D temporal TV (asynchronous in time)   
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/Video-1-pseudo3D(1).gif" width=70% height=70%>   
Fig. 3: Proposed pseudo-3D method (synchronized in time)  


This ensures interpretation of full vocal tract organ motion in 3D, and allows for quantitative extraction of vocal tract area functions to characterize speech patterns.  
The following output GIF is showing representative vocal tract area functions quantitating the vocal tract motion in 3D from a 10-slice time aligned reconstructions of speech task of uttering the repeated phrase 'za-na-za-loo-lee-laa' using our proposed method. (Temporal resolution= 18 ms)  
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/Vocal_area_function.gif" width=55% height=55%>  
3. In contrast to existing model-based  deep learning MRI reconstruction schemes that are reliant on fully sampled training datasets, our approach does not need to rely on training data, and reconstructs the image time series only from the measured under-sampled k-t data.  

### Output of speech MR reconstruction
We reconstructed two different speech tasks from two different speakers at 18 ms temporal resolution:  
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/count1to5.gif" width=70% height=70%>  
Fig. 4: 5-slice time aligned reconstruction of the speech task: repetition of counting from 1 to 5 by subject 1.   
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/loo_lee_la.gif" width=70% height=70%>  
Fig. 5: 5-slice time aligned reconstruction of the speech task: repetition of the phrase 'za-na-za-loo-lee-laa' by subject 2.    

### Dataset
An open source speech MR dataset has been published: https://doi.org/10.6084/m9.figshare.20468559.v1 

### File description
`Variational_manifold_code.ipynb` : This is the main file to run the optimization.  
`dataOpNewKbnufft.py`: This file preprocesses the raw MR data, calculates coil sensitivites, coil turncation and non-uniform FFTs.  
`generator.py`: Contains CNN-based gererator architecture, and generatior loss function.  
`latentVariable.py`: Contains low dimensional latent vector space, and latent vector regularization.  
`optimize_gen_sub.py`: Contains optimizaion pocesses with epochs, batch size, loss calculations and so on.   
`requirements.txt` : Contains depedencies of the code.  

### How to run the code

Ensure that all the dependencies of the `requirements.txt` are met. After that, specify the parametes and data path in the `Variational_manifold_code.ipynb` file and run it accordingly. Feel free to play with generator and latent vector configurations in the respective files.


###### Contact
The code is meant for reproducible research. In case of any difficulty, please open an issue or directly email me at rushdizahid-rusho@uiowa.edu
