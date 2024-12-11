import numba
from numba import cuda
import torch
import numpy as np

@cuda.jit
def lif_maxpool_kernel(membrane, synaptic, input_signal, output, pool_size, tau_mem_inv, tau_syn_inv, v_reset, v_th, height, width):
    #combine LIF and MaxPool2D

    #parameters from LIFCELL:
        # tau_syn_inv
        # tau_mem_inv
        # v_reset
        # v_th threshold
        #v_reset

        #membrane & synaptic are managed internally at layer
        #just like norse implementation, we update them here
    
    #ix = threadIdx.x + blockIdx.x * blockDim.x
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    block_dim_x = cuda.blockDim.x
    block_dim_y = cuda.blockDim.y
    gx = bx * block_dim_x + tx
    gy = by*block_dim_y + ty

    index = gy*width + gx

    pool_height, pool_width = pool_size #(tuple), like in MaxPool2D

    #Do the LIF stuff in GMEM
    if gx <width & gy < height:
        i = synaptic[index]
        v = membrane[index]
        i += input_signal[index]
        i *= tau_syn_inv
        v += tau_mem_inv * (i-v)

        if v > v_th:
            v = v_reset
        
        synaptic[index] = i
        membrane[index] = v


    #MaxPool2d
    #Do this in SMEM for efficiency
    #following lecture 12


    #different syntax in numba
    #__shared__ float tile[block_dim_y][BDIMX]
    
    # smem = cuda.shared.array(shape=(block_dim_x + 2,block_dim_y), dtype=numba.float32)
    smem = cuda.shared.array(shape=(64,64), dtype=numba.float32)

    if gx < width and gy<height:
        smem[ty, tx] = membrane[index]
    

    cuda.syncthreads()
    
    #THIS MADE IT MUCH FASTER
    #POOLING HAPPENS ACROSS ELEMENTS. so only one thread has to do pooling per block
    #so skip this step for most threads
    #like in norse implementation, default stride value is same as pool kernel size
    if ty%pool_height == 0 and tx % pool_width == 0:
        max_val = smem[ty,tx]
        if ty+1 < block_dim_y:
            max_val = max(max_val, smem[ty+1, tx]) #specifically optimized for (2,1) pooling
            #max_val = max(max_val, smem[ty+pool_height, tx]) ... (however many number of combinations)
        
        pooled_row = gy//pool_height
        pooled_col = gx //pool_width
        pooled_width = width // pool_width

        if pooled_row < height // pool_height and pooled_col < pooled_width:
            pooled_idx = pooled_row * pooled_width + pooled_col
            output[pooled_idx] = max_val        

        #output is 1d but smem is 2d

