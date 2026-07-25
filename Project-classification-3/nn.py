import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv("students.csv")

X = df.drop("Passed", axis=1)
Y = df["Passed"]

nuinumerical_columns = [
    "StudyHours",
    "Attendance",
    "SleepHours",
    "PreviousGrade",
    "AssignmentsCompleted"
]
categorical_columns = [
    "InternetAccess",
    "Extracurricular"
]

#creating pipeline
numerical_pipeline  = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num",numerical_pipeline, nuinumerical_columns),
    ("cat",categorical_pipeline, categorical_columns)
])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#convert sparse matrix to dense array
X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

#convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = torch.tensor(Y_train.values, dtype=torch.long)
Y_test = torch.tensor(Y_test.values, dtype=torch.long)

#dataset and dataloader
train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(
    train_dataset, batch_size=8,shuffle=True
)

test_loader = DataLoader(
    test_dataset, batch_size=8,shuffle=False
)

input_features = X_train.shape[1]

class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(input_features, 16)
        self.hidden2 = nn.Linear(16,8)
        self.output = nn.Linear(8,2)
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = StudentClassifier()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#training
epochs = 200

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion (outputs, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {running_loss:.4f}"
        )

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        outputs = model(X_batch)
        predictions = torch.argmax(outputs, dim=1)
        total += Y_batch.size(0)
        correct += (predictions == Y_batch).sum().item()
accuracy = 100 * correct / total
print(f"\nAccuracy: {accuracy:.2f}%")


#new student prediction
new_student = pd.DataFrame({
    "StudyHours":[6],
    "Attendance":[91],
    "SleepHours":[7],
    "PreviousGrade":[85],
    "AssignmentsCompleted":[8],
    "InternetAccess":["Yes"],
    "Extracurricular":["Yes"]
})

new_student = preprocessor.transform(new_student)
new_student = new_student.toarray() if hasattr(new_student, "toarray") else new_student

new_student = torch.tensor(
    new_student, dtype=torch.float32
)

model.eval()
with torch.no_grad():
    output = model(new_student)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(output, dim=1)

print("\n========== Prediction ==========")

print("Probabilities:")
print(probabilities)

print(f"\nPredicted Class: {prediction.item()}")

if prediction.item() == 0:
    print("Result: FAIL")
else:
    print("Result: PASS")
