import torch
import torch.nn as nn
import pandas as pd

input_features = 7
class HeartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_features,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,2)
    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = HeartModel()
model.load_state_dict(torch.load("heart_model.pth"))
model.eval()

sample = torch.tensor([[
    0.35,
    0.60,
    0.55,
    -0.40,
    0.80,
    0.0,
    1.0
]],dtype=torch.float32)

with torch.no_grad():
    output = model(sample)
    probabilities  = torch.softmax(output, dim=1).squeeze(0)
    prediction = torch.argmax(output,dim=1)
print(f"Probability of No Heart Disease: {probabilities[0].item():.2%}")
print(f"Probability of Heart Disease: {probabilities[1].item():.2%}")
print("\nPrediction")
if prediction.item() == 1:
    print("Result: HEART DISEASE DETECTED")
else:
    print("Result: No heart disease detected")
