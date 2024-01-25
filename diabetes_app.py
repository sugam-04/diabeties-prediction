import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv.xls')

# Separating the data labels
x = df.drop(columns='Outcome')
y = df['Outcome']

# Standardizing the data
scaler = StandardScaler()
scaler.fit(x)
standard_values = scaler.transform(x)
x = standard_values

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=2)

# Get column names from the original DataFrame before standardization
feature_columns = df.columns[:-1]

# Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Function to predict diabetes based on user input
def predict_diabetes(input_data):
    input_in_numpy = np.array(input_data).reshape(1, -1)
    std_data = scaler.transform(input_in_numpy)
    prediction = classifier.predict(std_data)
    return prediction[0]

# Streamlit app
st.title("Diabetes Prediction App")

# User input
input_data = []
for feature in feature_columns:
    val = st.slider(f"Enter {feature}", float(df[feature].min()), float(df[feature].max()))
    input_data.append(val)

if st.button("Predict"):
    prediction = predict_diabetes(input_data)
    if prediction == 0:
        st.success("You are not Diabetic!")
    else:
        st.error("You are Diabetic!")

# Visualizations
st.subheader("Data Visualizations")
st.pyplot(sns.pairplot(df, hue='Outcome', diag_kind='kde'))

# Boxplot
plt.figure(figsize=(15, 10))
for i, column in enumerate(feature_columns, 1):
    plt.subplot(3, 3, i)
    sns.boxplot(x='Outcome', y=column, data=df)
plt.tight_layout()
st.pyplot(plt)

# Correlation matrix heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
st.pyplot(plt)
