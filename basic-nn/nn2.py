import torch
import torch.nn as nn

#dataset
X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0]
])
Y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0]
])
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #hidden layer1 (1 feature -> 10 neurons)
        self.layer1 = nn.Linear(1,10)

        #hiddden layer2 (10 neurons -> 5 output)
        self.layer2 = nn.Linear(10,5)

        #output layer
        self.layer3 = nn.Linear(5,1)

    def forward(self, x ):

        #pass tru the first layer
        x = self.layer1(x)

        #apply the relu function
        x = torch.relu(x)

        #pass tru the output layer
        x = self.layer2(x)
        x = torch.relu(x)

        x = self.layer3(x)
        
        return x

#creati g mdoel
model = NeuralNetwork()

#mse loss
criterion = nn.MSELoss()

#for the optimize we can use SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training
epochs = 1000

for epoch in range(epochs):

    #forward pass
    prediction = model(X)

    #computer loss
    loss = criterion(prediction, Y)

    #clear old gradients
    optimizer.zero_grad()

    loss.backward()

    #update werights
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")


#testing
print("\nTraining Finished!")

test = torch.tensor([[5.0]])
prediction = model(test)
print(f"Input: {test.item()}")
print(f"Prediction: {prediction.item():.4f}")
