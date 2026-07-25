import torch
import torch.nn as nn

X = torch.tensor([
    [2., 70., 5., 60.],
    [3., 75., 6., 65.],
    [4., 80., 6., 70.],
    [5., 85., 7., 75.],
    [6., 90., 7., 80.],
    [7., 92., 8., 85.],
    [8., 95., 8., 90.],
    [9., 98., 8., 95.]
], dtype=torch.float32)

# Labels (Classes)
# 0 = Fail
# 1 = Pass
Y = torch.tensor([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
], dtype=torch.long)


class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,2)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = StudentClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),lr=0.01
)
epochs = 1000

for epoch in range(epochs):
    predictions = model(X)
    loss = criterion(predictions, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")

model.eval()

test = torch.tensor([
    [6.,88.,7.,82.]
], dtype=torch.float32)

with torch.no_grad():
    output = model(test)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(output,dim=1)
print("\n========== Prediction ==========")
print("Raw Output (Logits):")
print(output)

print("\nProbabilities:")
print(probabilities)

print("\nPredicted Class:")
print(prediction.item())

if prediction.item() == 0:
    print("Result: FAIL")
else:
    print("Result: PASS")
