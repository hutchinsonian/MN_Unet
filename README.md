# nat
nat network with multi-kernels  
# Experimental details  
natten CUDA compilation see https://github.com/SHI-Labs/Neighborhood-Attention-Transformer  
The proposed MN\_Unet is implemented in Pytorch and trained with 4 NVIDIA Titan Tesla V100 GPUs (each has 32GB memory). We adopt the Adamw\cite{loshchilov2018fixing} optimizer to train the model. In COVID-19 and LITS dataset，the original input dimensions of the image are (512, 512)，the initial learning rate is set to 3e-3, and the scheduler strategy is used to dynamically adjust the learning rate. In BraTS dataset, we crop the image input as (240, 240) to (224, 224) and use 0 padding at the last inference. The initial learning rate is set to 1e-3.
The following data augmentation techniques are applied: (1) random mirror flipping across the axial, coronal and sagittal planes by a probability of 0.5; (2) random rotate the image between [-20,20] by a probability of 0.5; (3) random enlarge the length and width of the picture by 1.25 or 1.5 times planes by a probability of 0.5
