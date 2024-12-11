import torch
from numba import cuda

# print("hello world")
#print(torch.cuda.is_available())

@cuda.jit
def add_kernel(a,b,c,size):
    idx = cuda.grid(1)
    if idx<size:
        c[idx] = a[idx] + b[idx]

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,a,b):
        size = a.numel()

        a_d = cuda.as_cuda_array(a)
        b_d = cuda.as_cuda_array(b)
        c_d = cuda.device_array_like(a_d)
        #c = torch.empty_like(a)

        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block -1) // threads_per_block

        add_kernel[blocks_per_grid, threads_per_block](
            a_d, b_d, c_d, size
        )
        return torch.tensor(c_d.copy_to_host(), device=a.device)
    
if __name__ == "__main__":
    model = MyModel()

    size = 1024
    a = torch.rand(size, device="cuda")
    b = torch.rand(size, device="cuda")

    c = model(a,b)

    e = a+b
    if torch.allclose(c,e):
        print("worked")
    else:
        print("no good")