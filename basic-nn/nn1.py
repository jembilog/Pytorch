import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,10)
        self.layer2 = nn.Linear(10,1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


model = MyModel()

x = torch.tensor([[5.9]])
target = torch.tensor([[10.0]])

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

prediction = model(x)

loss = criterion(prediction,target)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("Prediction:",prediction)
print("Loss:",loss)
