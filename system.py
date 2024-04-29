import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
@st.cache
def load_data():
    dataset = pd.read_csv("course_recommendation_dataset.csv")
    return dataset

# Preprocess the dataset
def preprocess_data(dataset):
    X = dataset.drop('course', axis=1)
    y = dataset['Course']

    # Convert grades to numerical values using label encoding
    label_encoder = LabelEncoder()
    for column in X.columns:
        X[column] = label_encoder.fit_transform(X[column])

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, label_encoder

# Train the KNN model
def train_model(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def main():
    st.title("Course Recommendation System")

    # Load the dataset
    dataset = load_data()

    # Preprocess the data
    X_train, y_train, label_encoder = preprocess_data(dataset)

    # Train the model
    knn = train_model(X_train, y_train)

    # Ask the user to enter their grades
    st.write("Please enter your grades for the following subjects (A, B, C, D or F):")
    grades = {}
    for column in dataset.columns[:-1]:
        grade = st.selectbox(f"What is your grade in {column}?", ['A', 'B', 'C', 'D','F'])
        grades[column] = label_encoder.transform([grade])[0]

    # Predict the field of interest
    user_data = pd.DataFrame([grades])
    predicted_field = knn.predict(user_data)
    st.write("\nBased on your grades, the predicted field of interest is:", predicted_field[0])

if __name__ == "__main__":
    main()

