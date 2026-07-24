from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import torch.nn as nn

class MyDataset(Dataset):
    def __init__(self):#runs once -> load the data
        self.X = torch.tensor([
            [1.0],
            [2.0],
            [3.0],
            [4.0]
        ])
        self.Y = torch.tensor([
            [2.0],
            [4.0],
            [6.0],
            [8.0]
        ])
    def __len__(self):#return dataset size
        return len(self.X)

    def __getitem__(self, index): #returns one sample
        return self.X[index], self.Y[index]
#creating dataset
dataset = MyDataset()
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        self.linear = nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters() , lr=0.01)

#training
epochs = 100
for epoch in range(epochs):
    for X_batch, Y_batch in loader:
        prediction = model(X_batch)

        loss = criterion(prediction, Y_batch)
        #reset gradient
        optimizer.zero_grad()
        #backprop
        loss.backward()
        #update
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("\nTraining Finished! Starting Testing...")

#evaluation mode
model.eval()

test = torch.tensor([
    [7.0],
    [8.0],
    [9.0],
    [10.0]
])
with torch.no_grad():
    prediction = model(test)
print(prediction)
