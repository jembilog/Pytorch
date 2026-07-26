import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset,DataLoader

df= pd.read_csv("heart_disease.csv")
# print(df.head())
X = df.drop("HeartDisease" , axis=1)
Y = df["HeartDisease"]

numerical_columns = [
    "Age",
    "BloodPressure",
    "Cholesterol",
    "MaxHeartRate",
    "BMI"
]

categorical_coluimns = [
    "Smoking"
]

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_coluimns)
])

#split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
)
X_train  = preprocessor.fit_transform(X_train)
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()

X_train = torch.tensor(X_train,dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.long)

train_dataset = TensorDataset(X_train, Y_train)
train_loader= DataLoader(train_dataset,batch_size=8,shuffle=True)

input_features = X_train.shape[1]

class HeartModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(input_features,16)
        self.fc2 = nn.Linear(16,8)
        self.fc3 = nn.Linear(8,2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = HeartModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

#train
epochs = 200
for epoch in range(epochs):
    model.train()
    runnning_loss=0
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs,Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
    if(epoch+1) % 20 == 0:
        print(f"Epoch{epoch+1}: Loss = {runnning_loss:.4f}")

#save
torch.save(model.state_dict(), "heart_model.pth")
print("\nModel Saved Successfully!")
