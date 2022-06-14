## Accelerated pseudo-3D dynamic speech MRI
Reference Paper: R. Z. Rusho, Q. Zou, W. Alam, S. Erattakulangara, M. Jacob and S. G. Lingala, "Accelerated pseudo 3D dynamic speech MR imaging at 3T using unsupervised deep variational manifold learning," accepted to MICCAI 2022.

Paper link: < will be uploaded soon! > 
 
### The framework of proposed pseudo-3D dynamic speech MRI
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/Proposed%20pseudo-3D%20speech%20MRI.png" width=70% height=70%>
Figure 1: The proposed pseudo-3D variational manifold speech MRI reconstrruct scheme.     
  
  
We propose to recover time aligned multi-slice (or pseudo-3D) dynamic image time series from sequentially acquired sparsly sampled spiral k-t space data of multiple 2D slices. This is done by jointly learning low dimensional latent vectors and the CNN-based generator parameters from the undersampled scanner measurements. The network optimizes the following loss function,  
<img src="https://github.com/rushdi-rusho/variational_manifold_speech_MRI/blob/main/images/Loss%20functions.png" width=50% height=50%>

### Main advantages of our method
1. In contrast to existing models, this method uses implicit motion relolved reconstruction strategy by exploiting the smoothness of the image time series on a manifold. Distant images that share same speech posture are mapped as neighbors on the manifold.  
2. In contrast to current 2D dynamic speech MRI approaches that reconstructs slices independently, resulting in full vocal tract motion to be out of synchrony across slices, we propose to jointly recover all the slices as a time aligned multi-slice 2D (or pseudo-3D) dataset. This ensures interpretation of full vocal tract organ motion in 3D, and allows for quantitative extraction of vocal tract area functions to characterize speech patterns.  
3. In contrast to existing model-based  deep learning MRI reconstruction schemes that are reliant on fully sampled training datasets, our approach does not need to rely on training data, and reconstructs the image time series only from the measured under-sampled k-t data.
### Output of speech MR reconstruction

### Dataset
An open source speech MR dataset will be provided soon! 

### File description
`Variational_manifold_code.ipynb` : This is the main file to run the optimization.  
`dataOpNewKbnufft.py`: This file preprocesses the raw MR data, calculates coil sensitivites, coil turncation and non-uniform FFTs.  
`generator.py`: Contains CNN-based gererator architecture, and generatior loss function.  
`latentVariable.py`: Contains low dimensional latent vector space, and latent vector regularization.  
`optimize_gen_sub.py`: Contains optimizaion pocesses with epochs, batch size, loss calculations and so on.   
`requirements.txt` : Contains depedencies of the code.  

### How to run the code

Ensure that all the dependencies of the requirement.txt are met. After that, specify the parametes and data path in the `Variational_manifold_code.ipynb` file and run it accordingly. Feel free to play with generator and latent vector configurations in the respective files.


###### Contact
The code is meant for reproducible research. In case of any difficulty, please open an issue or directly email me at rushdizahid-rusho@uiowa.edu
