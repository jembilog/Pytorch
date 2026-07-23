import torch
import time

device = torch.device("cuda")
print('GPU', torch.cuda.get_device_name(0))

x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)

torch.cuda.synchronize()
start = time.time()

z = torch.matmul(x,y)

torch.cuda.synchronize()
end = time.time()
print(f"Time: {end - start:.3f} seconds")
