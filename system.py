import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

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

# Function to convert alphabet letter to corresponding index
def letter_to_index(letter):
    return ord(letter.upper()) - ord('A')

# Function to convert index to corresponding alphabet letter
def index_to_letter(index):
    return chr(index + ord('A'))

# Main Streamlit app
def main():
    st.title("Course Recommendation System")

    st.write("Welcome to the Course Recommendation System!")
    st.write("Please enter your grades for the following subjects (A, B, C, D, or F):")

    grades = {}
    for column in X.columns:
        grade = st.text_input(f"What is your grade in {column}?", "")
        if grade.upper() in ["A", "B", "C", "D", "F"]:
            grades[column] = label_encoder.transform([grade.upper()])[0]
        else:
            st.warning("Invalid grade. Please enter A, B, C, D, or F.")

    st.write("Please select your field of interest:")
    field_options = dataset['Field of Interest'].unique()
    choice = st.selectbox("Select your field of interest", field_options)

    # Predict the course based on grades and field of interest
    user_data = pd.DataFrame([grades])
    predicted_course = knn.predict(user_data)

    # Map predicted course to recommended course
    recommended_courses = {
        "Domestic and Industrial Electricity": "Electrical and Industrial Automation Engineering",
        "Computer": "Electrical and Computer Engineering",
        "Electronics": "Electronics and Telecommunication Engineering",
        "Networks": "Electrical and Computer Engineering",
        "Solar Electricity": "Electrical and Renewable Energy",
        "Unknown": "Unknown"
    }

    recommended_course = recommended_courses.get(choice, "Unknown")

    if recommended_course != "Unknown":
        st.success(f"Based on your grades and field of interest, the recommended course is: {recommended_course}")
    else:
        st.warning("Based on your grades and field of interest, the recommended course is unknown.")

if __name__ == "__main__":
    main()
