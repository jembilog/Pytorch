import torch

#scalar
x = torch.tensor(5)

#array liek numpy (vector)
y = torch.tensor([1,2,3])

#matrix just liek numpy arrays
z = torch.tensor([
    [1,2],
    [3,4]
])

print(x)
print(y)
print(z)

#shape
print(z.shape)

#random tensors

#uniform distribution
print(torch.rand(3,4)) #rows , columns

#normal (Gaussian) distribution
print(torch.randn(3,4)) #rows, columns

#range
print(torch.arange(10))

#reshape
x = torch.arange(12)
x = x.reshape(3,4) #rows , columns
print(x)

#indexing
# x[0] #---> first row
# x[:,1] #---> second column
# x[-1] #---> last row

#arithmetic
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
print(a+b)
print(a-b)
print(a*b)
print(a/b)

#matrix multiplication
A = torch.rand(2,3)
B = torch.rand(3,4)
C = A @ B
print(C)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
