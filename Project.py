
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("DataSets/train.csv")
test = pd.read_csv("DataSets/test.csv")

print(train.head())
print(train.info())
print(train.describe())


sns.countplot(x="Survived", data=train)
plt.show()

# Gender vs Survival
sns.countplot(x="Sex", hue="Survived", data=train)
plt.show()

# Pclass vs Survival
sns.countplot(x="Pclass", hue="Survived", data=train)
plt.show()

# Check missing values
train.isnull().sum()

# So the exploratory data analysis (EDA) is Done 


# Now we will do Data preprocessing and Feature Engineering


train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
test["Fare"] = test["Fare"].fillna(test["Fare"].median())


train["Sex"] = train["Sex"].map({"male":0, "female":1})
test["Sex"] = test["Sex"].map({"male":0, "female":1})


train = pd.get_dummies(train, columns=["Embarked"], drop_first=True)
test = pd.get_dummies(test, columns=["Embarked"], drop_first=True)

# Feature Engineering
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

train["IsAlone"] = 1
train.loc[train["FamilySize"] > 1, "IsAlone"] = 0

test["IsAlone"] = 1
test.loc[test["FamilySize"] > 1, "IsAlone"] = 0

# Select features
#features = ["Pclass","Sex","Age","Fare","FamilySize","IsAlone",
#            "Embarked_Q","Embarked_S"]
#X = train[features]
#y = train["Survived"]

for col in ["Embarked_Q", "Embarked_S"]:
    if col not in train.columns:
        train[col] = 0
    if col not in test.columns:
        test[col] = 0

features = ["Pclass","Sex","Age","Fare","FamilySize","IsAlone",
            "Embarked_Q","Embarked_S"]
X = train[features]
y = train["Survived"]

#TRAIN/VALIDATION SPLIT

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL BUILDING

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

# EVALUATION


print("Accuracy:", accuracy_score(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))


test_X = test[features]
test_preds = model.predict(test_X)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("submission.csv", index=False)
print("✅ Submission file created!")