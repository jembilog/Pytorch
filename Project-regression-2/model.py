import torch
import torch.nn as nn

X = torch.tensor([
    [2., 80., 6., 5., 75.],
    [3., 82., 7., 6., 78.],
    [4., 85., 6., 7., 80.],
    [5., 88., 7., 8., 84.],
    [6., 90., 7., 8., 86.],
    [7., 92., 8., 9., 89.],
    [8., 95., 8., 10., 92.],
    [9., 97., 8., 10., 95.]
], dtype=torch.float32)

Y = torch.tensor([
    [76.],
    [79.],
    [82.],
    [85.],
    [88.],
    [91.],
    [95.],
    [98.]
], dtype=torch.float32)

X_mean = X.mean(dim=0, keepdim=True)
X_std = X.std(dim=0, keepdim=True)
X_scaled = (X - X_mean) /(X_std + 1e-8)

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(5, 10)
        self.hidden2 = nn.Linear(10, 20)
        self.hidden3 = nn.Linear(20, 5)
        self.output = nn.Linear(5,1)
    def forward(self, x):
        x = self.hidden1(x)
        x = torch.relu(x)

        x = self.hidden2(x)
        x = torch.relu(x)

        x = self.hidden3(x)
        x = torch.relu(x)

        x = self.output(x)
        return x
model = StudentModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr= 1.0)

epochs = 1000

for epoch in range(epochs):
    prediction = model(X_scaled)

    loss= criterion(prediction,Y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

print("\nTraining Finished")

model.eval()
new_student = torch.tensor([
    [6., 91., 7., 8., 87.]
])

with torch.no_grad():
    new_student_scaled= (new_student - X_mean) / (X_std + 1e-8)
    test_prediction = model(new_student_scaled)

    final_predictions = model(X_scaled)
    print("\n========== Prediction ==========")
    print(f"Predicted Grade: {test_prediction.item():.2f}")

    mae = torch.mean(torch.abs(Y - final_predictions))
    mape = torch.mean(torch.abs((Y - final_predictions)) / Y)
    accuracy = (1.0 - mape)  * 100
    print(f"Final Model MAE: {mae.item():.4f} points")
    print(f"Final Model Accuracy: {accuracy.item():.2f}%")

    print("\nTarget vs Prediction")
    for i in range(len(Y)):
        print(f"Actual: {Y[i].item():.1f} | Predicted: {final_predictions[i].item():.1f}")
