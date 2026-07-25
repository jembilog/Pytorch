import torch
import torch.nn as nn

X = torch.tensor([
    [80., 22., 25.],
    [85., 24., 28.],
    [90., 25., 30.],
    [100., 27., 35.],
    [120., 30., 40.],
    [140., 32., 45.],
    [160., 35., 50.],
    [180., 38., 55.]
], dtype=torch.float32)

#labels must be float for BCEWithLogitsLoss
Y = torch.tensor([
    [0.],
    [0.],
    [0.],
    [0.],
    [1.],
    [1.],
    [1.],
    [1.]
], dtype=torch.float32)

class DiabetesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3,8)
        self.fc2 = nn.Linear(8,4)
        self.fc3 = nn.Linear(4,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = DiabetesModel()

#loss
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 1000
for epoch in range(epochs):
    outputs= model(X)
    loss = criterion(outputs, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

model.eval()
test = torch.tensor([
    [130., 31., 43.]
], dtype=torch.float32)

with torch.no_grad():
    logits = model(test)
    probability  = torch.sigmoid(logits)
    prediction = (probability >= 0.5).float()

print("\n===== Prediction =====")

print("Logit:")
print(logits)

print("\nProbability:")
print(probability)

print("\nPredicted Class:")
print(int(prediction.item()))

if prediction.item() == 0:
    print("Result: No Diabetes")
else:
    print("Result: Diabetes")
