import torch
from numba import cuda
from lif_maxpool_kernel import lif_maxpool_kernel

class LIFMaxPool2d(torch.nn.Module):
    def __init__(self, lif_params, pool_size):
        super().__init__()#self, lif_params, pool_size)
        self.lif_params = lif_params
        self.pool_size = pool_size

        #norse manages these internally, so necessary to do here as well
        self.membrane = None
        self.synaptic = None
    
    
    def reset_state(self, batch_size, height, width, device):
        self.membrane = torch.zeros((batch_size, height, width), device=device)
        self.synaptic = torch.zeros((batch_size, height, width), device=device)

    
    def forward(self, input_signal): # membrane, synaptic, input_signal):

        batch_size, channels, height, width = input_signal.shape

        #for any reason any issues, just hard reset (prevent crashes)
        if self.membrane is None or self.synaptic is None or self.membrane.shape[-2] != (height, width):
            self.reset_state(batch_size, height, width, input_signal.device)

        input_signal = input_signal.detach()
        # input_signal.requires_grad_(False)

        #make things flat bc output in cuda is 1d

        flat_size = height*width

        input_flat = input_signal.view(-1).cuda()
        membrane_flat = self.membrane.view(-1).cuda()
        synaptic_flat = self.synaptic.view(-1).cuda()
        output_flat = torch.zeros(flat_size // (self.pool_size[0] * self.pool_size[1]), device="cuda")

        tau_mem_inv = self.lif_params.tau_mem_inv.item()
        tau_syn_inv = self.lif_params.tau_syn_inv.item()
        v_reset = self.lif_params.v_reset.item()
        v_th = self.lif_params.v_th.item()

        block_dim = (32,32) #arbitrary size
        grid_dim = ((width + block_dim[0]-1)//block_dim[0], (height + block_dim[1]-1)//block_dim[1])

        #this is how you launch a kernel in numba
        lif_maxpool_kernel[grid_dim, block_dim](
            membrane_flat, synaptic_flat, input_flat, output_flat,
            self.pool_size, tau_mem_inv, tau_syn_inv,v_reset, v_th, height, width
        )

        pooled_height = (height // self.pool_size[0]) * self.pool_size[0]
        pooled_width = (width // self.pool_size[1]) * self.pool_size[1]
        expected_size = batch_size* channels* pooled_height*pooled_width

        #just in case there is an error
        if output_flat.numel() > expected_size:
            output_flat = output_flat[:expected_size]
        elif output_flat.numel() < expected_size:
            padding = torch.zeros(expected_size-output_flat.numel(), device=output_flat.device)
            output_flat = torch.cat((output_flat, padding))

        return output_flat.view(batch_size, channels, pooled_height, pooled_width)
