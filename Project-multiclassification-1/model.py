import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv("students.csv")

X = df.drop("Performance",axis=1)
Y = df["Performance"]
label_encoder = LabelEncoder()
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
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

Y = label_encoder.fit_transform(Y)
X_train , X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)
X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=42,stratify=Y_train)

X_train=  preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_val = X_val.toarray()
    X_test = X_test.toarray()

#convert all of them to tensors
X_train = torch.tensor(X_train,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.long)

X_test = torch.tensor(X_test,dtype=torch.float32)
Y_test  = torch.tensor(Y_test,dtype=torch.long)

#dataset and dataloader
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)

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

input_features = X_train.shape[1]

class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(input_features,32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.3)
        self.hidden2 = nn.Linear(32,16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(16,4)
    def forward(self,x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.hidden2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x

model = StudentModel()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

#early stopping vars
best_loss = float("inf")
counter = 0
patience = 25
epochs =  300

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
        train_loss += loss.item()

        #calculate the training accuracy
        predicted = torch.argmax(outputs,dim=1)
        train_correct += (predicted == Y_batch).sum().item()
    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / len(train_loader.dataset)

    #validate inside the loop
    model.eval() 
    validation_loss = 0
    val_correct = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            validation_loss += loss.item() * X_batch.size(0)

            #calculate the validation accuracy
            predicted = torch.argmax(outputs,dim=1)
            val_correct += (predicted == Y_batch).sum().item()
        validation_loss /= len(val_loader.dataset)
        validation_accuracy = val_correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train loss: {train_loss:.4f} | Train acc: {train_accuracy*100:.2f} | "
            f"Validation loss: {validation_loss:.4f} | Val acc: {validation_accuracy*100:.2f}"
        )

        if validation_loss < best_loss:
            best_loss =  validation_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
print("\nTraining Finished!")


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
accuracy =  accuracy_score(actual, predictions)
print(f"Test Accuracy : {accuracy*100:.2f}%")
print("\nConfusion Matrix")
print(confusion_matrix(actual, predictions))
print("\nClassification Report")

print(
    classification_report(
        actual,
        predictions,
        target_names=label_encoder.classes_
    )

)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")
