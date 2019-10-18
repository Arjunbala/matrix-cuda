import torch

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
# Initialise cuda tensors here. E.g.:
A = torch.rand(1000, 1000, device = 'cuda:0')
B = torch.rand(1000, 1000, device = 'cuda:0')
# Wait for the above tensors to initialise.
torch.cuda.synchronize()
with torch.cuda.stream(s1):
    C = torch.mm(A, A)
with torch.cuda.stream(s2):
    D = torch.mm(B, B)
# Wait for C and D to be computed.
#torch.cuda.synchronize()
# Do stuff with C and D.
