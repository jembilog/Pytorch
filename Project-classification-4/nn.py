import pandas as pd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import(
    accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
)
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("heart_disease.csv")
X = df.drop("HeartDisease", axis=1)
Y = df["HeartDisease"]

numerical_columns = [
    "Age",
    "BloodPressure",
    "Cholesterol",
    "MaxHeartRate",
    "BMI"
]

categorical_columns = [
    "Smoking"
]

numerical_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline,categorical_columns)
])

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

Y_train_np = Y_train.to_numpy()
Y_test_np = Y_test.to_numpy()

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.long)
Y_test = torch.tensor(Y_test.values, dtype=torch.long)

#dataset and sdataloader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset  = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=8, shuffle=False)

input_features= X_train.shape[1]

class HeartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(input_features,16)
        self.hidden2 = nn.Linear(16,8)
        self.output = nn.Linear(8,2)
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x
model = HeartModel()
criterion = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if(epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss : {running_loss/len(train_loader):.4f}")


#evaluation
model.eval()
predictions = []
actual = []
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        outputs = model(X_batch)
        predicted = torch.argmax(outputs, dim=1)
        predictions.extend(predicted.numpy())
        actual.extend(Y_batch.numpy())

print("Accuracy")
print(accuracy_score(actual, predictions))
print("\nPrecision")
print(precision_score(actual,predictions))
print("\nRecall")
print(recall_score(actual,predictions))
print("\nF1 score")
print(f1_score(actual, predictions))
print("\nConfusion Matrix")
print(confusion_matrix(actual, predictions))
print("\nClassification Report")
print(classification_report(actual, predictions))

#testing
new_patient = pd.DataFrame({
    "Age":[47],
    "BloodPressure":[145],
    "Cholesterol":[240],
    "MaxHeartRate":[144],
    "BMI":[30.0],
    "Smoking":["Yes"]
})
new_patient = preprocessor.transform(new_patient)

if hasattr(new_patient, "toarray"):
    new_patient = new_patient.toarray()

new_patient = torch.tensor(
    new_patient, dtype=torch.float32
)

model.eval()
with torch.no_grad():
    output = model(new_patient)
    probabilities = torch.softmax(output, dim=1).squeeze(0)
    prediction = torch.argmax(output, dim=1)
print(f"Probability of No Heart Disease: {probabilities[0].item():.2%}")
print(f"Probability of Heart Disease: {probabilities[1].item():.2%}")
print("\nPrediction")
if prediction.item() == 1:
    print("Result: HEART DISEASE DETECTED")
else:
    print("Result: No heart disease detected")
