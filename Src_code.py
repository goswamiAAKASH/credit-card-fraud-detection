import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset using raw string literal for Windows path
df = pd.read_csv(r'D:\project\fraud_detection.csv')
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Get the shape of the dataframe
print(df.shape)

# Check distribution of transaction types
print(df.type.value_counts())

# Visualize the distribution of transaction types using a pie chart
type = df['type'].value_counts()
transactions = type.index
quantity = type.values

# Drop missing values
df = df.dropna()
print(df)

# Map isFraud values from numeric to string
df['isFraud'] = df['isFraud'].map({0: 'No Fraud', 1: 'Fraud'})
print(df)

# Check unique transaction types
print(df['type'].unique())

# Map transaction types to numerical values
df['type'] = df['type'].map({'PAYMENT': 1, 'TRANSFER': 4, 'CASH_OUT': 2, 'DEBIT': 5, 'CASH_IN': 3})

# Check the value counts after mapping
print(df['type'].value_counts())
print(df)

x = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = df.iloc[:, -2]

# Display the target variable
print(y)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Print the model score on the test set
print(model.score(xtest, ytest))  # Model completed
'''
# Save the trained model as a .pkl file
with open(r"D:\project\fraud_detection_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Load the model to verify it was saved correctly
with open(r"D:\project\fraud_detection_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Test prediction (using reshaped input)
print(loaded_model.predict([[2, 9800, 170136, 160296]]))

# Print statistical summary of numerical columns in the dataset
print("\nStatistical Summary of Numerical Columns:")
print(df.describe())

# Print column names and their data types
print("\nColumn Names and Dtypes:")
print(df.dtypes)
'''