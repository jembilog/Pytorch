import pandas as pd
import torch 
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv("students.csv")
X = df.drop("Passed", axis=1)
Y = df["Passed"]

numerical_columns = [
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

numerical_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])


#train and test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#train and validation split
X_train,X_val, Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=42)

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_val = X_val.toarray()
    X_test = X_test.toarray()

X_train = torch.tensor(X_train,dtype=torch.float32)
Y_train = torch.tensor(Y_train.values,dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val.values, dtype=torch.long)

X_test = torch.tensor(X_test,dtype=torch.float32)
Y_test  = torch.tensor(Y_test.values,dtype=torch.long)

#dataset
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

#dataloader 
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

#model
input_features = X_train.shape[1]
class StudentModel(nn.Module):
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
model = StudentModel()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#early stopping
best_loss = float("inf")
patience = 20
counter = 0
epochs = 500

#training
for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        #calculate the training acc
        predicted = torch.argmax(outputs,dim=1)
        train_correct += (predicted == Y_batch).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

#validation
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            validation_loss += loss.item() * X_batch.size(0)
    validation_loss /= len(val_loader.dataset)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {validation_loss:.4f}"
    )

    if validation_loss < best_loss:
        best_loss = validation_loss
        counter = 0  
        torch.save(
            model.state_dict(),
            "best_model.pth"
        )
        print("Best model updated!")
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

#load best model
model.load_state_dict(torch.load("best_model.pth"))

#final testing
model.eval()
predictions = []
actual = []

with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        outputs = model(X_batch)
        predicted = torch.argmax(outputs, dim=1)
        predictions.extend(predicted.numpy())
        actual.extend(Y_batch.numpy())

accuracy = accuracy_score(actual,predictions)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")

new_student = pd.DataFrame({

    "StudyHours":[6],
    "Attendance":[90],
    "SleepHours":[7],
    "PreviousGrade":[85],
    "AssignmentsCompleted":[8],
    "InternetAccess":["Yes"],
    "Extracurricular":["Yes"]

})

new_student = preprocessor.transform(new_student)

if hasattr(new_student,"toarray"):
    new_student = new_student.toarray()
new_student = torch.tensor(new_student,dtype=torch.float32)

model.eval()
with torch.no_grad():
    output = model(new_student)
    probabilities = torch.softmax(output, dim=1).squeeze(0)
    prediction = torch.argmax(output, dim=1)#it returns 0
print(f"Probability of FAILING: {probabilities[0].item():.2%}")
print(f"Probability of PASSING: {probabilities[1].item():.2%}")
print("\nPrediction")
if prediction.item() == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
