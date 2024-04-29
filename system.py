import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv("course_recommendation_data.csv")

dataset = load_data()

# Split features (grades) and target (field of interest)
X = dataset.drop(['Field of Interest', 'Course'], axis=1)
y = dataset['Field of Interest']

# Convert grades to numerical values using label encoding
label_encoder = LabelEncoder()
for column in X.columns:
    X[column] = label_encoder.fit_transform(X[column])

# Initialize the KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# Train the KNN model
knn.fit(X, y)

# Define function to get user input
def get_user_input():
    grades = {}
    for column in X.columns:
        while True:
            grade = st.selectbox(f"What is your grade in {column}?", options=["A", "B", "C", "D", "F"])
            grades[column] = label_encoder.transform([grade])[0]
            break
    field_of_interest = st.selectbox("What is your field of interest?", options=dataset['Field of Interest'].unique())
    return grades, field_of_interest

# Predict the field of interest
def predict_field(grades):
    user_data = pd.DataFrame([grades])
    predicted_field = knn.predict(user_data)
    return predicted_field[0]

# Define function to get recommended course
def get_recommended_course(predicted_field):
    recommended_courses = {
        "Domestic and Industrial Electricity": "Electrical and Industrial Automation Engineering",
        "Computer": "Electrical and Computer Engineering",
        "Electronics": "Electronics and Telecommunication Engineering",
        "Networks": "Electrical and Computer Engineering",
        "Solar Electricity": "Electrical and Renewable Energy",
        "Unknown": "Unknown"
    }
    return recommended_courses.get(predicted_field, "Unknown")

# Streamlit UI
def main():
    st.title("Course Recommendation System")

    st.write("Welcome to the Course Recommendation System!")
    st.write("Please enter your grades and field of interest to get a course recommendation.")

    grades, field_of_interest = get_user_input()

    st.write("Predicting the field of interest based on your grades...")
    predicted_field = predict_field(grades)
    st.write("Predicted field of interest:", predicted_field)

    st.write("Determining the recommended course...")
    recommended_course = get_recommended_course(predicted_field)
    
    if recommended_course != "Unknown":
        st.success(f"Based on your grades and field of interest, the recommended course is: {recommended_course}")
    else:
        st.warning("Based on your grades and field of interest, the recommended course is unknown.")

if __name__ == "__main__":
    main()
