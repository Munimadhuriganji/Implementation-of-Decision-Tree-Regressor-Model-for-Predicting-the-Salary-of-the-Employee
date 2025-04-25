# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by:GANJI MUNI MADHURI
RegisterNumber: 212223230060
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Load and preview data
data = pd.read_csv("/content/Salary (1).csv")
print(data.head())
print(data.info())
print(data.isnull().sum())

# Encode categorical variable
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

# Prepare features and target
x = data[["Position", "Level"]]
y = data["Salary"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train the model
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Predict and evaluate
y_pred = dt.predict(x_test)
r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Show position encoding
print("Position encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# Predict new value
print("Prediction for [5, 6]:", dt.predict([[5, 6]]))
```
## Output:
![image](https://github.com/user-attachments/assets/d662ad78-580e-46c2-90ac-7a7c72114ebf)
![image](https://github.com/user-attachments/assets/f17b99c2-84ec-4f6b-83b0-092a11ec2375)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
